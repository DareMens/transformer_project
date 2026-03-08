import pytest
import torch
from modelling.tokenizer import pad_sequences

@pytest.fixture
def sample_data():
    # Sample input sequences of different lengths
    seq1 = torch.tensor([1, 2, 3])
    seq2 = torch.tensor([4, 5])
    seq3 = torch.tensor([6, 7, 8, 9])
    
    sequences = [seq1, seq2, seq3]
    
    return sequences

def test_padding(sample_data):
    padded_sequences, mask = pad_sequences(sample_data, pad_token=0)

    # Expected outputs
    expected_padded = torch.tensor([
        [1, 2, 3, 0],    # seq1 padded with one <pad>
        [4, 5, 0, 0],    # seq2 padded with two <pad>
        [6, 7, 8, 9],    # seq3 no padding needed
    ])
    expected_mask = torch.tensor([
        [1, 1, 1, 0],    # Mask for seq1, last position is padding
        [1, 1, 0, 0],    # Mask for seq2, two padding positions
        [1, 1, 1, 1],    # No padding for seq3
    ])

    # Check if padded sequences and mask are correct
    assert torch.equal(padded_sequences, expected_padded), "Padding incorrect"
    assert torch.equal(mask, expected_mask), "Mask generation incorrect"