from torch import nn
import torch
import os

class WordEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(WordEmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        # Initialize the encoding matrix
        encoding = torch.empty(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        
        # Compute sine and cosine positional encodings
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)  # Add batch dimension

        # Register encoding as a buffer
        self.register_buffer("encoding", encoding)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Use the registered buffer and move it to the correct device
        encoding = self.encoding[:, :x.size(1), :].to(x.device)
        return self.dropout(x + encoding)
    
class TrainablePositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000, dropout=0.1, init='uniform'):
        super(TrainablePositionalEncoding, self).__init__()
        # Learnable positional embeddings
        if init == 'uniform':
            self.positional_embeddings = nn.Parameter(torch.zeros(1, max_len, embedding_dim))
            nn.init.uniform_(self.positional_embeddings, -0.1, 0.1)  # Initialize embeddings
        elif init == 'normal':
            self.positional_embeddings = nn.Parameter(torch.zeros(1, max_len, embedding_dim))
            nn.init.normal_(self.positional_embeddings, mean=0, std=0.1)  # Initialize embeddings
        elif init == 'xavier':
            self.positional_embeddings = nn.Parameter(torch.zeros(1, max_len, embedding_dim))
            nn.init.xavier_uniform_(self.positional_embeddings)  # Initialize embeddings
        else: # Random initialization with mean 0 and std 1
            self.positional_embeddings = nn.Parameter(torch.randn(1, max_len, embedding_dim)) 
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Use positional embeddings up to the sequence length
        embeddings = self.positional_embeddings[:, :x.size(1), :].to(x.device)
        return self.dropout(x + embeddings)

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super(RotaryPositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        # Compute inverse frequency for RoPE
        inv_freq = 1.0 / (10000 ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """
        Apply rotary positional encoding to the input tensor `x`.
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
        Returns:
            Tensor of the same shape as input, with rotary encoding applied.
        """
        seq_len = x.size(1)
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)  # Shape: (seq_len, 1)
        angles = position * self.inv_freq  # Shape: (seq_len, embedding_dim / 2)

        # Calculate sin and cos embeddings
        sin_emb = torch.sin(angles)  # Shape: (seq_len, embedding_dim / 2)
        cos_emb = torch.cos(angles)  # Shape: (seq_len, embedding_dim / 2)

        # Repeat interleave the sin and cos embeddings to match input dimensions
        sin_emb = sin_emb.repeat_interleave(2, dim=-1)  # Shape: (seq_len, embedding_dim)
        cos_emb = cos_emb.repeat_interleave(2, dim=-1)  # Shape: (seq_len, embedding_dim)

        # Apply RoPE to the input tensor
        x1, x2 = x[..., ::2], x[..., 1::2]  # Split into even and odd dimensions
        rotated_x = torch.cat([-x2, x1], dim=-1)  # Rotate: (x_even, x_odd) -> (-x_odd, x_even)

        # Combine with sin and cos embeddings
        return (x * cos_emb) + (rotated_x * sin_emb)
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, feature_dim):
        super(PositionwiseFeedForward, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, d_model)
        )

    def forward(self, x):
        return self.feed_forward(x)
    
class TransformerLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        """
        Initialize the scheduler.
        :param optimizer: The optimizer being used for training.
        :param d_model: Dimensionality of the model embeddings.
        :param warmup_steps: Number of warm-up steps.
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        """Update the learning rate based on the current step."""
        self.step_num += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        """Return the current learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def _compute_lr(self):
        """Compute the learning rate using the formula from the paper."""
        return (
            self.d_model ** -0.5
            * min(self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5)
        )
    
def create_optimizer(model, lr, weight_decay):
    """
    Initialize the AdamW optimizer with decoupled weight decay.
    Exclude bias and layer norm parameters from weight decay.
    :param model: The Transformer model.
    :param lr: Learning rate.
    :param weight_decay: Weight decay coefficient.
    """
    no_decay = ['bias', 'LayerNorm.weight']
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        },
    ]
    return torch.optim.AdamW(param_groups, lr=lr)

# Save model checkpoint
def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"transformer_epoch_{epoch}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.step_num,
            "epoch": epoch,
            "loss": loss,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved at {checkpoint_path}")


def translate(model, src_sentences, tokenizer, device, max_len=64):
    """
    Generate translations for a batch of source sentences.
    Args:
        model: The trained Transformer model.
        src_sentences: Tensor of the source sentences (shape: [batch_size, seq_len]).
        tokenizer: Tokenizer object to decode the generated tokens.
        device: Device to run the model on.
        max_len: Maximum length of the generated output sequences.
    Returns:
        List of generated sentences as lists of words.
    """
    model.eval()
    model.to(device)
    
    # Prepare the source sentences (add batch dimension if necessary)
    if len(src_sentences.size()) == 1:
        src_sentences = src_sentences.unsqueeze(0)
    src_sentences = src_sentences.to(device)  # Move to device
    
    # Create the initial target sequences with the [BOS] token
    batch_size = src_sentences.size(0)
    tgt_sentences = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
    # Pass source sentences through the encoder
    memory = model.encode(src_sentences)
    
    with torch.no_grad():
        for _ in range(max_len):
            # Generate the output from the model
            output = model.generate(memory, tgt_sentences)
            
            # Get the token with the highest probability for each sentence in the batch
            next_tokens = output.argmax(dim=-1)[:, -1]
            
            # Append the tokens to the target sequences
            tgt_sentences = torch.cat([tgt_sentences, next_tokens.unsqueeze(1)], dim=1)
            
            # Stop if all sentences have generated the [EOS] token
            if (next_tokens == tokenizer.eos_token_id).all():
                break
    
    # Decode the generated tokens to words    
    return tokenizer.batch_decode(tgt_sentences, skip_special_tokens=True)

import json
from tqdm import tqdm
import torch
from evaluate import load

def evaluate_model(
    model, 
    data_loader, 
    tokenizer, 
    device, 
    criterion, 
    max_len, 
    compute_bleu=True, 
    print_output=False
):
    """
    Evaluates a model on a given dataset.

    Parameters:
        model: The model to be evaluated.
        data_loader: DataLoader providing input and target data.
        tokenizer: Tokenizer for decoding sequences.
        device: Device (CPU/GPU) for computation.
        criterion: Loss function to calculate model performance.
        max_len: Maximum sequence length for translation.
        compute_bleu (bool): Whether to compute BLEU score.
        print_output (bool): Whether to print sample predictions and references.

    Returns:
        avg_loss: Average loss over the dataset.
        bleu_score: Computed BLEU score (if enabled).
    """
    model.eval()
    model.to(device)
    total_loss = 0
    references = []
    hypotheses = []

    bar_desc = "Testing" if compute_bleu else "Validation"
    bar = tqdm(data_loader, desc=bar_desc)

    with torch.no_grad():
        for batch in bar:
            src = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)

            tgt_input = tgt[:, :-1]  # Remove the last token for input
            tgt_output = tgt[:, 1:]  # Remove the first token for target

            # Forward pass
            output = model(src, tgt_input)
            loss = criterion(output.view(-1, output.shape[-1]), tgt_output.reshape(-1))
            total_loss += loss.item()

            if compute_bleu:
                # Decode references and model-generated outputs
                references.extend(tokenizer.batch_decode(tgt_output, skip_special_tokens=True))
                generated = translate(model, src, tokenizer, device, max_len)
                hypotheses.extend(generated)

    avg_loss = total_loss / len(data_loader)

    bleu_score = 0.0
    if compute_bleu:
        if print_output:
            print("\nSample Predictions:")
            for i in range(min(5, len(hypotheses))):
                print(f"Example {i + 1}")
                print(f"Prediction: {hypotheses[i]}")
                print(f"Reference: {references[i]}\n")

        # Save test results to a JSON file
        with open("predictions.json", "w") as f:
            json.dump({"references": references, "hypotheses": hypotheses}, f, indent=4)

        bleu_metric = load("bleu")
        try:
            if all(len(pred.strip()) == 0 for pred in hypotheses):
                print("All predictions are empty.")
            else:
                bleu_score = bleu_metric.compute(
                    predictions=hypotheses,
                    references=[[ref] for ref in references]
                )["bleu"]
        except ZeroDivisionError:
            print("Error: Division by zero encountered in BLEU score computation.")

    model.train()  # Ensure model is set back to training mode after evaluation
    return avg_loss, bleu_score