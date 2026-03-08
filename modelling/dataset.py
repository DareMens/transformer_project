import re
import unicodedata
from typing import List, Dict
from datasets import Dataset as HFDataset, load_dataset
import json
import os
import torch
from transformers import PreTrainedTokenizerBase


# Clean text utility function
def clean_text(text: str, whitelist: str = "abcdefghijklmnopqrstuvwxyz ÄÖÜäöüß ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥") -> str:
    """
    Clean the input text by:
    1. Removing non-UTF8 characters
    2. Removing URLs and HTML tags
    3. Filtering to only include whitelisted characters
    4. Converting to lowercase
    
    Args:
        text (str): Input text to clean.
        whitelist (str): Allowed characters.
    
    Returns:
        str: Cleaned text.
    """
    text = unicodedata.normalize('NFC', text)  # Normalize Unicode
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = text.lower()  # Convert to lowercase
    return ''.join(char for char in text if char in whitelist).strip()


# Dataset cleaning function for Hugging Face `datasets`
def preprocess_dataset(dataset, min_length=5, max_length=64, max_length_ratio=1.5):
    """
    Preprocess and clean a Hugging Face dataset.

    Args:
        dataset (HFDataset): Input dataset object.
        min_length (int): Minimum acceptable sentence length.
        max_length (int): Maximum acceptable sentence length.
        max_length_ratio (float): Maximum source/target length ratio.

    Returns:
        HFDataset: Cleaned dataset.
    """
    def preprocess_example(example):
        source = clean_text(example["translation"]["de"])
        target = clean_text(example["translation"]["en"])
        
        # Length and ratio checks
        if (len(source.split()) < min_length or len(source.split()) > max_length or
            len(target.split()) < min_length or len(target.split()) > max_length or
            len(source) / len(target) > max_length_ratio or len(target) / len(source) > max_length_ratio):
            return {"de": source, "en": target, "valid": False}
        return {"de": source, "en": target, "valid": True}

    # Apply the cleaning function and filter out invalid examples
    cleaned_dataset = dataset.map(preprocess_example, remove_columns=dataset.column_names)
    return cleaned_dataset.filter(lambda x: x["valid"]).remove_columns("valid")


# Load and preprocess dataset
def load_and_preprocess_dataset(output_dir: str, tokenizer: PreTrainedTokenizerBase, max_length=64):
    """
    Load, preprocess, and tokenize the dataset.

    Args:
        output_dir (str): Directory to save cleaned datasets.
        tokenizer (PreTrainedTokenizerBase): Tokenizer object.
        max_length (int): Maximum token sequence length.
        num_proc (int): Number of processes for parallel execution.

    Returns:
        Dict[str, HFDataset]: Tokenized datasets for each split.
    """
    if os.path.exists(output_dir):
        print(f"Loading preprocessed datasets from {output_dir}")
        return {split: HFDataset.load_from_disk(os.path.join(output_dir, split)) for split in ["train", "validation", "test"]}

    # Load raw dataset
    raw_dataset = load_dataset("wmt17", "de-en")

    # Preprocess datasets
    processed_splits = {}
    for split in raw_dataset.keys():
        print(f"Preprocessing {split} dataset...")
        processed_splits[split] = preprocess_dataset(raw_dataset[split])

    # Tokenization function
    def tokenize_function(batch):
        return tokenizer(
            batch["de"], text_target=batch["en"], max_length=max_length,
            padding="max_length", truncation=True
        )

    # Tokenize datasets
    tokenized_splits = {
        split: processed_splits[split].map(
            tokenize_function,
            batched=True,
            remove_columns=processed_splits[split].column_names,
            desc=f"Tokenizing {split} dataset",
            # num_proc=num_proc
        )
        for split in processed_splits
    }

    # Save preprocessed datasets
    os.makedirs(output_dir, exist_ok=True)
    for split, dataset in tokenized_splits.items():
        dataset.save_to_disk(os.path.join(output_dir, split))

    return tokenized_splits


# PyTorch Dataset Wrapper
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset: HFDataset):
        """
        Initialize the PyTorch Dataset wrapper for a Hugging Face dataset.

        Args:
            hf_dataset (HFDataset): Tokenized Hugging Face dataset.
        """
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {key: torch.tensor(value) for key, value in self.dataset[idx].items()}
