import re
from collections import Counter
from typing import List, Union
import json
import os
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from transformers import PreTrainedTokenizerFast, GPT2Tokenizer

def causal_mask(size):
    """
    Creates a future mask to prevent the model from attending to future tokens.
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask

def pad_sequences(sequences, pad_token=0):
    """
    Pads a list of variable-length sequences with the pad_token (default 0).

    Args:
        sequences: A list of sequences (tensors) of varying lengths
        pad_token: The token to pad sequences with (default is 0)

    Returns:
        padded_sequences: Tensor of shape (batch_size, max_seq_length)
        mask: A binary mask of the same shape, where 1 indicates valid tokens and 0 indicates padding
    """
    batch_size = len(sequences)
    max_seq_length = max([seq.size(0) for seq in sequences])  # Find the longest sequence
    
    padded_sequences = torch.full((batch_size, max_seq_length), pad_token)  # Initialize tensor with pad_token
    mask = torch.zeros((batch_size, max_seq_length), dtype=torch.bool)  # Binary mask (1 for valid, 0 for padding)

    for i, seq in enumerate(sequences):
        length = seq.size(0)
        padded_sequences[i, :length] = seq  # Fill the sequence part with actual data
        mask[i, :length] = 1  # Mark the valid token positions in the mask

    return padded_sequences, mask

def render_token(token: bytes) -> str:
    """Helper function to render token bytes in a readable format"""
    try:
        return token.decode('utf-8')
    except UnicodeDecodeError:
        return ' '.join(f'{b:02x}' for b in token)

class CharLevelBPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.special_tokens = {
            '[PAD]': 0,
            '[BOS]': 1,
            '[EOS]': 2,
            '[UNK]': 3
        }
        self.special_tokens_reverse = {v: k for k, v in self.special_tokens.items()}
        
    def save(self, file_prefix: str) -> None:
        """
        Save tokenizer configuration and vocabulary
        
        Args:
            file_prefix: Path prefix for saving files
        """
        # Save model configuration
        model_file = file_prefix + ".model"
        config = {
            'version': 'bpe v1',
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'vocab': self.vocab
        }
        
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        # Save human-readable vocabulary
        vocab_file = file_prefix + ".vocab"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write("Special tokens:\n")
            for token, idx in self.special_tokens.items():
                f.write(f"{token}: {idx}\n")
            f.write("\nRegular vocabulary:\n")
            for token, idx in sorted(self.vocab.items()):
                if token not in self.special_tokens:
                    f.write(f"{token}: {idx}\n")

    def load(self, model_file: str) -> None:
        """
        Load tokenizer configuration from a saved file
        
        Args:
            model_file: Path to the saved model file
        """
        if not model_file.endswith(".model"):
            raise ValueError("Model file must have .model extension")
            
        with open(model_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        assert config['version'] == 'bpe v1', f"Incompatible version: {config['version']}"
        
        self.vocab_size = config['vocab_size']
        self.special_tokens = config['special_tokens']
        self.vocab = config['vocab']
        self.special_tokens_reverse = {v: k for k, v in self.special_tokens.items()}
        
    def _tokenize_by_characters(self, corpus):
        tokenized_corpus = [[' '.join(list(word)) for word in sentence.split()] for sentence in corpus]
        return [" ".join(sentence) for sentence in tokenized_corpus]

    def _get_stats(self, corpus):
        pairs = Counter()
        for word in corpus:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += 1
        return pairs

    def _merge_pairs(self, pair, corpus):
        bigram = ' '.join(pair)
        new_corpus = []
        for word in corpus:
            new_word = re.sub(r'\b{}\b'.format(bigram), ''.join(pair), word)
            new_corpus.append(new_word)
        return new_corpus

    def train(self, corpus):
        # Initialize vocabulary with special tokens
        corpus = self._tokenize_by_characters(corpus)
        vocab = Counter(' '.join(corpus).split())
        
        # Adjust vocab size to account for special tokens
        effective_vocab_size = self.vocab_size - len(self.special_tokens)
        
        max_len = len(vocab)
        while len(vocab) < effective_vocab_size:
            pairs = self._get_stats(corpus)
            if not pairs:
                break
            most_frequent = max(pairs, key=pairs.get)
            corpus = self._merge_pairs(most_frequent, corpus)
            vocab[''.join(most_frequent)] = pairs[most_frequent]
            if len(vocab) == max_len:
                break
            max_len = len(vocab)

        # Create final vocabulary with special tokens
        self.vocab = {**self.special_tokens}
        for idx, (token, _) in enumerate(vocab.most_common(effective_vocab_size), start=len(self.special_tokens)):
            self.vocab[token] = idx
            
        return corpus

    def tokenize(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Tokenize text into token ids"""
        tokenized_text = ' '.join(list(text))
        for token in sorted(self.vocab.keys(), key=len, reverse=True):
            if token not in self.special_tokens:
                tokenized_text = tokenized_text.replace(' '.join(token), token)
        
        tokens = tokenized_text.split()
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.special_tokens['[BOS]'])
            
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.special_tokens['[UNK]'])
                
        if add_special_tokens:
            token_ids.append(self.special_tokens['[EOS]'])
            
        return token_ids

    def pad_or_truncate(self, token_ids: List[int], max_length: int, padding: str = 'post') -> List[int]:
        """
        Pad or truncate a sequence of token ids to max_length
        
        Args:
            token_ids: List of token ids
            max_length: Target length
            padding: 'pre' or 'post' to specify padding position
            
        Returns:
            Padded or truncated sequence
        """
        if len(token_ids) > max_length:
            return token_ids[:max_length]
        
        pad_length = max_length - len(token_ids)
        pad_token = self.special_tokens['[PAD]']
        
        if padding == 'post':
            return token_ids + [pad_token] * pad_length
        else:  # pre padding
            return [pad_token] * pad_length + token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.special_tokens_reverse:
                if not skip_special_tokens:
                    tokens.append(self.special_tokens_reverse[token_id])
            else:
                tokens.append(next((t for t, idx in self.vocab.items() if idx == token_id), '[UNK]'))

        return tokens

class MyByteLevelBPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '[PAD]': 0,
            '[BOS]': 1,
            '[EOS]': 2,
            '[UNK]': 3
        }
        self.merges = {}
        self.vocab = self._build_vocab()

    def save(self, file_prefix: str) -> None:
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        Similar to sentencepiece's model saving:
        - model file is used for loading
        - vocab file is for human inspection
        """
        # Save the model file
        model_file = file_prefix + ".model"
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write("bytebpe v1\n")  # Version identifier
            f.write(f"{self.vocab_size}\n")  # Save vocab size
            
            # Save special tokens
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            
            # Save merges
            f.write(f"{len(self.merges)}\n")
            for (idx1, idx2), idx in self.merges.items():
                f.write(f"{idx1} {idx2} {idx}\n")

        # Save human-readable vocabulary file
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            # Write special tokens
            f.write("Special tokens:\n")
            for token, idx in self.special_tokens.items():
                f.write(f"{token}: {idx}\n")
            
            f.write("\nByte vocabulary:\n")
            # Write basic byte vocabulary
            start_idx = len(self.special_tokens)
            for i in range(256):
                token = self.vocab[start_idx + i]
                s = render_token(token)
                f.write(f"[{s}] {start_idx + i}\n")
            
            f.write("\nMerged tokens:\n")
            # Write merged tokens
            for idx, token in self.vocab.items():
                if idx in inverted_merges:
                    idx1, idx2 = inverted_merges[idx]
                    s = render_token(token)
                    s1 = render_token(self.vocab[idx1])
                    s2 = render_token(self.vocab[idx2])
                    f.write(f"[{s1}][{s2}] -> [{s}] {idx}\n")

    def load(self, model_file: str) -> None:
        """
        Load tokenizer configuration from a saved model file
        
        Args:
            model_file: Path to the saved model file
        """
        if not model_file.endswith(".model"):
            raise ValueError("Model file must have .model extension")
            
        with open(model_file, 'r', encoding='utf-8') as f:
            # Read version
            version = f.readline().strip()
            assert version == "bytebpe v1", f"Incompatible version: {version}"
            
            # Read vocab size
            self.vocab_size = int(f.readline().strip())
            
            # Read special tokens
            num_special = int(f.readline().strip())
            self.special_tokens = {}
            for _ in range(num_special):
                token, idx = f.readline().strip().split()
                self.special_tokens[token] = int(idx)
            
            # Read merges
            num_merges = int(f.readline().strip())
            self.merges = {}
            for _ in range(num_merges):
                idx1, idx2, idx = map(int, f.readline().strip().split())
                self.merges[(idx1, idx2)] = idx
        
        # Rebuild vocabulary
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        # Start with special tokens
        vocab = {idx: token.encode('utf-8') for token, idx in self.special_tokens.items()}
        # Add byte vocabulary
        start_idx = len(self.special_tokens)
        vocab.update({(start_idx + idx): bytes([idx]) for idx in range(256)})
        # Add merged tokens
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def _get_stats(self, ids, counts=None):
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def _merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text, verbose=False):
        # Adjust vocab size to account for special tokens
        assert self.vocab_size >= (256 + len(self.special_tokens))
        num_merges = self.vocab_size - 256 - len(self.special_tokens)

        if isinstance(text, str):
            try:
                text_bytes = text.encode("utf-8")
            except UnicodeEncodeError as e:
                print(f"Encoding error: {e}")
                return []
        elif isinstance(text, list):
            text_bytes = []
            for t in text:
                try:
                    text_bytes.extend(t.encode("utf-8"))
                except UnicodeEncodeError as e:
                    print(f"Encoding error for '{t}': {e}")
                    return []
        else:
            raise ValueError("Input should be a string or a list of strings")
        
        # Start ids after special tokens
        start_idx = len(self.special_tokens)
        ids = [start_idx + b for b in text_bytes]

        merges = {}
        vocab = {idx: token.encode('utf-8') for token, idx in self.special_tokens.items()}
        vocab.update({(start_idx + idx): bytes([idx]) for idx in range(256)})
        
        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = start_idx + 256 + i
            ids = self._merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token ids with optional special tokens"""
        ids = []
        if add_special_tokens:
            ids.append(self.special_tokens['[BOS]'])
            
        text_bytes = text.encode("utf-8")
        start_idx = len(self.special_tokens)
        byte_ids = [start_idx + b for b in text_bytes]
        
        # Merge tokens according to learned merges
        while len(byte_ids) >= 2:
            stats = self._get_stats(byte_ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            byte_ids = self._merge(byte_ids, pair, idx)
        
        ids.extend(byte_ids)
        
        if add_special_tokens:
            ids.append(self.special_tokens['[EOS]'])
            
        return ids

    def decode(self, ids: List[int], concat=True, skip_special_tokens: bool = True) -> str:
        """Decode token ids back to text"""
        # given ids (list of integers), return Python string or list of decoded elements
        if skip_special_tokens:
            ids = [idx for idx in ids if idx not in self.special_tokens.values()]
        text_bytes = [self.vocab[idx] for idx in ids]
        if concat:
            text = b"".join(text_bytes).decode("utf-8", errors="replace")
            return text
        else:
            return [b.decode("utf-8", errors="replace") for b in text_bytes]

    def pad_or_truncate(self, token_ids: List[int], max_length: int, padding: str = 'post') -> List[int]:
        """
        Pad or truncate a sequence of token ids to max_length
        
        Args:
            token_ids: List of token ids
            max_length: Target length
            padding: 'pre' or 'post' to specify padding position
            
        Returns:
            Padded or truncated sequence
        """
        if len(token_ids) > max_length:
            return token_ids[:max_length]
        
        pad_length = max_length - len(token_ids)
        pad_token = self.special_tokens['[PAD]']
        
        if padding == 'post':
            return token_ids + [pad_token] * pad_length
        else:  # pre padding
            return [pad_token] * pad_length + token_ids
            
    # Maintain the tokenize alias
    tokenize = encode

class CustomBPETokenizer:
    def __init__(self, vocab_size=50000, special_tokens=None):
        bpe_model = BPE(
            cache_capacity=5000,
            dropout=0.1,
            unk_token="[UNK]",
            continuing_subword_prefix="Ġ",  # GPT-2 uses "Ġ" to denote spaces
            # continuing_subword_prefix="##",
            # end_of_word_suffix="@@",
        )
        self.tokenizer = Tokenizer(bpe_model)
        # self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens else {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'mask_token': '[MASK]'
        }

    def train(self, data):
        """
        Train the tokenizer on the given data.

        Args:
        - data: List of strings to train the tokenizer on.
        """
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=list(self.special_tokens.values()),
            show_progress=True,
            continuing_subword_prefix="Ġ",  # GPT-2 uses "Ġ" to denote spaces
            # continuing_subword_prefix="##",
            # end_of_word_suffix="@@",
        )
        self.tokenizer.train_from_iterator(data, trainer=trainer)

    def save(self, save_dir="bpe_tokenizer"):
        """
        Save the tokenizer vocab and merges files in the specified directory.

        Args:
        - save_dir: Directory to save vocab.json and merges.txt.
        """
        os.makedirs(save_dir, exist_ok=True)
        # Save vocab and merges in Hugging Face format
        self.tokenizer.model.save(save_dir)
        # Save the serialized tokenizer.json for PreTrainedTokenizerFast
        self.tokenizer.save(f"{save_dir}/tokenizer.json")
        print(f"Tokenizer saved to {save_dir}")

    @staticmethod
    def from_pretrained(save_dir="bpe_tokenizer"):
        """
        Load the tokenizer from a pretrained directory.

        Args:
        - save_dir: Directory containing tokenizer files (vocab.json, merges.txt, tokenizer.json).
        
        Returns:
        - A PreTrainedTokenizerFast or GPT2Tokenizer object.
        """
        try:
            # Attempt to load as GPT2Tokenizer
            Tokenizer = GPT2Tokenizer.from_pretrained(save_dir)
            Tokenizer.add_special_tokens({
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
                'cls_token': '[CLS]',
                'sep_token': '[SEP]',
                'mask_token': '[MASK]'
            })
            return Tokenizer
        except:
            # Fallback to PreTrainedTokenizerFast
            return PreTrainedTokenizerFast.from_pretrained(
                save_dir,
                unk_token="[UNK]",
                bos_token="[BOS]",
                pad_token="[PAD]",
                eos_token="[EOS]",
                cls_token="[CLS]",
                sep_token="[SEP]",
                mask_token="[MASK]"
            )