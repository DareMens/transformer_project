import sys
import os

# Add the parent directory to the system path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from modelling.functional import PositionalEncoding, TrainablePositionalEncoding
from modelling.model import TransformerEncoderLayer
from torch.utils.tensorboard import SummaryWriter
import json

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Dataset Generation
class DatasetType:
    SHIFT = "shift"
    ODD_EVEN = "odd_even"

def generate_dataset(task_type, num_samples=10000, max_length=30, min_length=5, vocab_size=30):
    """
    Generate synthetic datasets for different sequence tasks
    
    Args:
        task_type: Type of task from DatasetType
        num_samples: Number of samples to generate
        max_length: Maximum sequence length
        min_length: Minimum sequence length
        vocab_size: Size of vocabulary
    """
    dataset = []
    
    if task_type == DatasetType.SHIFT:
        # Predict next token in sequence (original task)
        original_sequence = range(1, vocab_size + 1)
        for _ in range(num_samples):
            seq_length = random.randint(min_length, max_length)
            pivot = random.randint(0, vocab_size - seq_length)
            while pivot + seq_length > vocab_size:
                pivot = random.randint(0, vocab_size - seq_length)
            sequence = original_sequence[pivot:pivot + seq_length - 1]
            target = original_sequence[pivot + 1:pivot + seq_length]
            dataset.append((sequence, target))

    elif task_type == DatasetType.ODD_EVEN:
        # Generate sequences and label each token as odd (1) or even (0)
        for _ in range(num_samples):
            seq_length = random.randint(min_length, max_length)
            sequence = [random.randint(1, vocab_size) for _ in range(seq_length)]
            # Create binary labels: 1 for odd, 0 for even
            target = [1 if num % 2 == 1 else 0 for num in sequence]
            dataset.append((sequence, target))
            
    return dataset

# Positional Encodings
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)

class TrainablePositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000, dropout=0.0, init='uniform'):
        super(TrainablePositionalEncoding, self).__init__()
        # Learnable positional embeddings
        if init == 'uniform':
            self.positional_embeddings = nn.Parameter(torch.zeros(1, max_len, embedding_dim))
            nn.init.uniform_(self.positional_embeddings, -0.1, 0.1)  # Initialize embeddings
        elif init == 'normal':
            self.positional_embeddings = nn.Parameter(torch.zeros(1, max_len, embedding_dim))
            nn.init.normal_(self.positional_embeddings, mean=0, std=0.1)  # Initialize embeddings
        elif init == 'standard':
            self.positional_embeddings = nn.Parameter(torch.randn(1, max_len, embedding_dim))
        elif init == 'xavier':
            self.positional_embeddings = nn.Parameter(torch.zeros(1, max_len, embedding_dim))
            nn.init.xavier_uniform_(self.positional_embeddings)
        else:
            raise ValueError("Invalid initialization type")
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Use positional embeddings up to the sequence length
        embeddings = self.positional_embeddings[:, :x.size(1), :].to(x.device)
        return self.dropout(x + embeddings)

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, vocab_size, max_len, positional_encoding_type, init='uniform'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        if positional_encoding_type == "trainable":
            self.positional_encoding = TrainablePositionalEncoding(d_model, max_len, init=init)
        else: # default to sinusoidal positional encoding
            self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len)

        # encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.transformer = nn.ModuleList([
                TransformerEncoderLayer(d_model, nhead, feature_dim=2048, dropout=0.1)
                for _ in range(num_layers)
            ])
        self.fc = nn.Linear(d_model, vocab_size + 1)

    def forward(self, src):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        # src = self.transformer(src)
        for layer in self.transformer:
            src = layer(src)
        return self.fc(src)

# Training Loop
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src)
        output = output.view(-1, output.size(-1))
        tgt = tgt.view(-1)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation Function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src)
            output = output.view(-1, output.size(-1))
            tgt = tgt.view(-1)
            loss = criterion(output, tgt)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Accuracy Calculation
def calculate_accuracy(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_sequences = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src)
            predictions = output.argmax(dim=-1)

            for pred_seq, tgt_seq in zip(predictions, tgt):
                # chech if only 0s and 1s are present in the sequence
                if (torch.all((tgt_seq == 0.0) | (tgt_seq == 1.0)).item()):
                    tgt_seq = tgt_seq.cpu().numpy()
                else:
                    tgt_seq = tgt_seq[(tgt_seq != 0)].cpu().numpy()
                pred_seq_trimmed = pred_seq[:len(tgt_seq)].cpu().numpy()
                if np.array_equal(pred_seq_trimmed, tgt_seq):
                    total_correct += 1
                total_sequences += 1
    accuracy = total_correct / total_sequences * 100
    return accuracy

# Experiment Runner
class ExperimentRunner:
    def __init__(self, d_model, nhead, num_layers, max_len, batch_size, num_epochs, learning_rate):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []
        self.writer = SummaryWriter()

    def run(self, vocab_sizes, encoding_types, task_types):
        for vocab_size in vocab_sizes:
            for encoding_type in encoding_types:
                for task_type in task_types:
                    print(f"Running experiment with vocab_size={vocab_size}, encoding_type={encoding_type}, task_type={task_type}")
                    self.max_len = vocab_size # sequence length is equal to vocab size for this experiment
                    dataset = generate_dataset(task_type, 10000, max_length=self.max_len//2, vocab_size=vocab_size)
                    train_data, val_data = dataset[:8000], dataset[8000:]
                    test_data = generate_dataset(task_type, 2000, min_length=self.max_len//2, max_length=self.max_len, vocab_size=vocab_size)

                    def collate_fn(batch):
                        src, tgt = zip(*batch)
                        src = nn.utils.rnn.pad_sequence([torch.tensor(s) for s in src], batch_first=True, padding_value=0)
                        tgt = nn.utils.rnn.pad_sequence([torch.tensor(t) for t in tgt], batch_first=True, padding_value=0)
                        return src, tgt

                    train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True)
                    val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, collate_fn=collate_fn)
                    test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, collate_fn=collate_fn)

                    model = TransformerModel(self.d_model, self.nhead, self.num_layers, vocab_size, self.max_len, encoding_type.split("_")[0], init=encoding_type.split("_")[-1]).to(self.device)
                    total_params = sum(p.numel() for p in model.parameters())
                    print(f"Model has {total_params:,} trainable parameters")
                    if task_type == DatasetType.ODD_EVEN:
                        criterion = nn.CrossEntropyLoss()
                    else:
                        criterion = nn.CrossEntropyLoss(ignore_index=0)
                    optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)

                    best_val_accuracy = 0.0

                    for epoch in range(self.num_epochs):
                        train_loss = train_model(model, train_loader, criterion, optimizer, self.device)
                        val_loss = evaluate_model(model, val_loader, criterion, self.device)
                        val_accuracy = calculate_accuracy(model, val_loader, self.device)

                        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
                        self.writer.add_scalar(f"{encoding_type}/Val Accuracy", val_accuracy, epoch + 1)

                        if val_accuracy == 100.0 or epoch == self.num_epochs - 1:
                            epoch_reached = epoch + 1
                            best_val_accuracy = val_accuracy
                            break

                    del train_data, val_data, train_loader, val_loader
                    test_accuracy = calculate_accuracy(model, test_loader, self.device)
                    print(f"Test Accuracy for vocab_size={vocab_size}, encoding_type={encoding_type}, task_type={task_type}: {test_accuracy:.2f}%")
                    self.results.append({
                        "model_size": total_params,
                        "vocab_size": vocab_size,
                        "encoding_type": encoding_type,
                        "epoch_reached": epoch_reached,
                        "task_type": task_type,
                        "val_accuracy": best_val_accuracy,
                        "test_accuracy": test_accuracy,
                    })
                    # print(f"Results: {self.results}")
                    # clear GPU memory
                    del model, criterion, optimizer
                    torch.cuda.empty_cache()

        self.writer.close()
        self._save_results()

    def _save_results(self):
        results_file = f"results_{self.d_model}_{self.nhead}_{self.num_epochs}.json"
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                existing_results = json.load(f)
            existing_results.extend(self.results)
            with open(results_file, "w") as f:
                json.dump(existing_results, f, indent=4)
        else:
            with open(results_file, "w") as f:
                json.dump(self.results, f, indent=4)
        print(f"Results saved to results_{self.d_model}_{self.nhead}.json")

# Main Script
if __name__ == "__main__":
    base_model = {
        "d_model": 64,
        "nhead": 2,
        "num_layers": 2
    }
    runner = ExperimentRunner(
        d_model=base_model["d_model"], nhead=base_model["nhead"],
        num_layers=base_model["num_layers"], max_len=5000,
        batch_size=128, num_epochs=50, learning_rate=1e-4)
    encoding_types=["sinusoidal", "trainable_uniform", "trainable_normal", "trainable_standard", "trainable_xavier"]
    vocab_sizes = [2**i for i in range(6, 11)] # 64, 128, 256, 512, 1024
    task_types = [DatasetType.ODD_EVEN, DatasetType.SHIFT]
    runner.run(vocab_sizes=vocab_sizes, encoding_types=encoding_types, task_types=task_types)

    # clear GPU memory
    del runner
    torch.cuda.empty_cache()