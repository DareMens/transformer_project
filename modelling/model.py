import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .functional import PositionalEncoding, PositionwiseFeedForward, TrainablePositionalEncoding

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, feature_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, feature_dim)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        self_attn_output = self.self_attention(src, src, src, src_mask)
        src = src + self.dropout(self_attn_output)
        src = self.layer_norm1(src)
        ffn_output = self.feed_forward(src)
        src = src + self.dropout(ffn_output)
        src = self.layer_norm2(src)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, feature_dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, mask_future=True)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feature_transformation = PositionwiseFeedForward(d_model, feature_dim)
        
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.layer_norm_3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, attention_mask=None, cross_attention_mask=None):
        # Self-attention with future masking
        self_attn_output = self.self_attention(x, x, x, attention_mask)
        x = x + self.dropout(self_attn_output)
        x = self.layer_norm_1(x)
        
        # Cross-attention with encoder output
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, cross_attention_mask)
        x = x + self.dropout(cross_attn_output)
        x = self.layer_norm_2(x)
        
        # Feed-forward network
        ffn_output = self.feature_transformation(x)
        x = x + self.dropout(ffn_output)
        x = self.layer_norm_3(x)
        
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, max_len=5000, pe="sinusoidal"):
        super(TransformerModel, self).__init__()
        self.d_model = d_model

        # Embedding layer for input tokens and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = pe
        if self.pe == "sinusoidal":
            self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        elif self.pe == "trainable":
            self.positional_encoding = TrainablePositionalEncoding(d_model, max_len, dropout)
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))

        # Encoder and Decoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Final linear layer for vocabulary projection
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, attention_mask=None, cross_attention_mask=None):
        # Encode source sequence
        src = self.embedding(src) * self.scale
        src = self.positional_encoding(src)

        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        # Decode target sequence
        tgt = self.embedding(tgt) * self.scale
        tgt = self.positional_encoding(tgt)

        for layer in self.decoder_layers:
            tgt = layer(tgt, src, attention_mask, cross_attention_mask)

        # Output logits
        output = self.fc_out(tgt)
        return output
    
    def generate(self, memory, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        tgt = self.embedding(tgt) * self.scale
        tgt = self.positional_encoding(tgt)

        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, tgt_mask, tgt_key_padding_mask)

        output = self.fc_out(tgt)
        return output
    
    def encode(self, src, src_mask=None):
        src = self.embedding(src) * self.scale
        src = self.positional_encoding(src)

        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        return src