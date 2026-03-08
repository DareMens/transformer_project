import sys
import os

# Add the parent directory to the system path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import torch
from modelling.model import TransformerModel as Transformer
from modelling.functional import TransformerLRScheduler, create_optimizer, evaluate_model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from modelling.tokenizer import CustomBPETokenizer
import hydra
from omegaconf import DictConfig
import logging
from torch.utils.tensorboard import SummaryWriter

# Define your Transformer model
from modelling.dataset import TranslationDataset, load_and_preprocess_dataset

# Define constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Training loop function
@hydra.main(config_path="../conf", config_name="config.yaml")
def train(cfg: DictConfig):
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=cfg.log_dir)

    # Initialize random seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    # Prepare tokenizer
    tokenizer = CustomBPETokenizer.from_pretrained(cfg.tokenizer_dir)

    # Create datasets
    datasets = load_and_preprocess_dataset(
    output_dir=cfg.data_dir,
    tokenizer=tokenizer,
    max_length=cfg.max_len,
    # num_proc=4
    )
    train_dataset = TranslationDataset(datasets["train"])
    val_dataset = TranslationDataset(datasets["validation"])
    test_dataset = TranslationDataset(datasets["test"])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # Initialize model, optimizer, and scheduler
    logger.info("Positional encoding type: %s", cfg.pe)
    model = Transformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
        pe=cfg.pe,
    ).to(DEVICE)
    # model parameters
    logger.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    optimizer = create_optimizer(model, cfg.lr, cfg.weight_decay)
    num_training_steps = len(train_loader) * cfg.epochs
    num_warmup_steps = 0.1 * num_training_steps
    scheduler = TransformerLRScheduler(optimizer, cfg.d_model, num_warmup_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=cfg.label_smoothing)  # Assuming 0 is the padding token

    # Training variables
    best_val_loss = float("inf")
    best_bleu_score = 0
    patience = 5
    no_improve_epochs = 0
    os.makedirs(cfg.save_dir, exist_ok=True)

    # load the model if it exists
    model_path = os.path.join(cfg.save_dir, f"best_model_{cfg.pe}_{cfg.max_len}.pth")
    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs} [Training]")
        
        for batch in train_bar:
            src, tgt = batch["input_ids"].to(DEVICE), batch["labels"].to(DEVICE)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            optimizer.zero_grad()

            # Forward pass
            output = model(src, tgt_input)
            loss = criterion(output.view(-1, output.shape[-1]), tgt_output.reshape(-1))

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{cfg.epochs} - Training loss: {train_loss}")
        writer.add_scalar("Loss/train", train_loss, epoch)

        # Validation phase
        if (epoch + 1) % cfg.val_freq == 0:
            val_loss, bleu_score = evaluate_model(model, val_loader, tokenizer, DEVICE, criterion, cfg.max_len, print_output=True)
            if bleu_score > best_bleu_score:
                best_bleu_score = bleu_score
        else:
            val_loss, bleu_score = evaluate_model(model, val_loader, tokenizer, DEVICE, criterion, cfg.max_len, compute_bleu=False)

        logger.info(f"Epoch {epoch + 1}/{cfg.epochs} - Validation loss: {val_loss}, BLEU score: {best_bleu_score}")
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("BLEU/val", best_bleu_score, epoch)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), model_path)
            logger.info(f"New best model saved at epoch {epoch + 1}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # Test phase
    model.load_state_dict(torch.load(model_path))
    test_loss, bleu_score = evaluate_model(model, test_loader, tokenizer, DEVICE, criterion, cfg.max_len, print_output=True)
    logger.info(f"Test loss: {test_loss}, BLEU score: {bleu_score}")
    writer.add_scalar("Loss/test", test_loss)
    writer.add_scalar("BLEU/test", bleu_score)  

    writer.close()

# Run the training loop
if __name__ == "__main__":
    train()