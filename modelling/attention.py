import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, d_model=8, mask_future=False):
        super().__init__()
        self.d_model = d_model
        self.mask_future = mask_future
        
    def forward(self, query, key, value, mask=None, return_weights=False):
        """
        Forward pass for the single-head attention mechanism.

        Args:
            q: Query tensor (batch_size, seq_length, d_model)
            k: Key tensor (batch_size, seq_length, d_model)
            v: Value tensor (batch_size, seq_length, d_model)
            mask: Optional binary mask for padded tokens or future tokens

        Returns:
            Output: Resulting tensor after attention mechanism.
        """

        if mask is not None:
            # Ensure the mask has the same dimensions as the attention scores (batch_size, seq_len, seq_len)
            mask = mask.unsqueeze(1)  # Expand mask to (batch_size, 1, seq_len)
            # (batch_size, seq_len, seq_len)
            mask = mask.expand_as(torch.ones(query.size(0), query.size(1), key.size(1)))

        if self.mask_future:
            # Create a causal mask (upper triangular with ones above the diagonal)
            causal_mask = torch.tril(torch.ones(query.size(0), query.size(1), key.size(1))).bool()

            # Expand causal mask to match attention scores shape (batch_size, seq_len, seq_len)
            # causal_mask = causal_mask.unsqueeze(0).expand(query.size(0), -1, -1)  # (batch_size, seq_len, seq_len)

            # Combine with the existing mask if any
            if mask is None:
                mask = causal_mask
            else:
                # Ensure the mask has the same dimensions as the attention scores (batch_size, seq_len, seq_len)
                mask = mask & causal_mask

        # Apply attention mechanism
        d_k = key.size(-1)  # Get the dimension of K or Q (they are the same)

        # Compute scaled dot-product attention scores 
        # (out_shape: (batch_size, num_queries, num_keys))
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # Apply mask (if provided)
        if mask is not None:
            # Apply a very large negative number where the mask is 0 (so softmax will give very low probabilities)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Multiply attention weights by the value matrix to get the output 
        # (out_shape: (batch_size, num_queries, d_v)) here d_v = d_k = d_q
        attention_output = torch.matmul(attention_weights, value)

        if return_weights:
            return attention_output, attention_weights
        else:
            return attention_output
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, mask_future=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.mask_future = mask_future
        
        # Single matrix for each of Q, K, V transforms
        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.output_transform = nn.Linear(d_model, d_model, bias=False)
        
        # Initialize parameters
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Initialize the weights - although in the test they will be overwritten
        nn.init.xavier_uniform_(self.query_transform.weight)
        nn.init.xavier_uniform_(self.key_transform.weight)
        nn.init.xavier_uniform_(self.value_transform.weight)
        nn.init.xavier_uniform_(self.output_transform.weight)
    
    def forward(self, query, key, value, attention_mask=None):
        batch_size = query.size(0)
        
        # Apply the transformations
        Q = self.query_transform(query)  # (batch_size, query_len, d_model)
        K = self.key_transform(key)      # (batch_size, key_len, d_model)
        V = self.value_transform(value)  # (batch_size, value_len, d_model)
        
        # Split heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads)
        K = K.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads)
        V = V.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads)
        
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model // self.num_heads, dtype=torch.float32))
        
        # Apply future masking if required
        if self.mask_future:
            future_mask = torch.triu(torch.ones(query.size(1), key.size(1)), diagonal=1).bool()
            future_mask = future_mask.to(query.device)
            scores = scores.masked_fill(future_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention_mask to match the scores shape
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()  # (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.expand(-1, scores.size(1), scores.size(2), -1) # (batch_size, num_heads, seq_len, key_len)
            scores = scores.masked_fill(~attention_mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        out = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape and apply output transformation
        out = out.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        out = out.view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)
        out = self.output_transform(out)
        
        return out