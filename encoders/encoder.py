import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"

        # Linear projections for Q, K, V (applied after splitting heads)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Final linear layer to mix heads
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask=None):

        # values, keys, query shape: (N, seq_len, embed_size)
        N = query.shape[0]
        value_len = values.shape[1]
        key_len = keys.shape[1]
        query_len = query.shape[1]

        # Split embedding into heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Apply linear projections
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Rearrange to (N, heads, seq_len, head_dim)
        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)

        # Compute attention scores: (N, heads, query_len, key_len)
        energy = torch.matmul(queries, keys.transpose(-2, -1))

        # Scale by sqrt(head_dim) (correct formula)
        energy = energy / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            # mask shape should broadcast to (N, heads, query_len, key_len)
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Softmax over key_len dimension
        attention = torch.softmax(energy, dim=-1)

        # Weighted sum of values: (N, heads, query_len, head_dim)
        out = torch.matmul(attention, values)

        # Concatenate heads: (N, query_len, heads * head_dim)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(N, query_len, self.heads * self.head_dim)

        # Final linear projection
        out = self.fc_out(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Position-wise Feed Forward Network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):

        # Multi-head attention
        attention = self.attention(value, key, query, mask)

        # Residual + LayerNorm
        x = self.dropout(self.norm1(attention + query))

        # Feed Forward Network
        forward = self.feed_forward(x)

        # Residual + LayerNorm
        out = self.dropout(self.norm2(forward + x))

        return out
    

# test
if __name__ == "__main__":

    batch_size = 2
    seq_len = 4
    embed_size = 8
    heads = 2
    dropout = 0.1
    forward_expansion = 4


    block = TransformerBlock(
        embed_size=embed_size,
        heads=heads,
        dropout=dropout,
        forward_expansion=forward_expansion,
    )

    x = torch.randn(batch_size, seq_len, embed_size)

    print("Input shape :", x.shape)

    out = block(x, x, x, mask=None)

    print("Output shape:", out.shape)

    print("\nInput (first batch, first token):")
    print(x[0, 0])

    print("\nOutput (first batch, first token):")
    print(out[0, 0])
