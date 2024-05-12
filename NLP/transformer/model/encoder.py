import torch.nn as nn
from torchtyping import TensorType
from typing import Optional

from model.multi_head_attention import MultiHeadAttention
from model.feed_forward import FeedForward
from model.positional_embedding import PositionalEmbedding


class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 drop_p: float):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff, drop_p=drop_p)

        self.mha_ln = nn.LayerNorm(d_model)
        self.ff_ln = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(drop_p)

    def forward(self,
                x: TensorType["batch", "seq_len", "d_model"],
                encoder_mask: Optional[TensorType["batch", "n_heads", "seq_len", "seq_len"]] = None
                ):
        residual = self.multi_head_attention(q=x, k=x, v=x, mask=encoder_mask)
        residual = self.dropout(residual)
        x = self.mha_ln(x + residual)

        residual = self.feed_forward(x)
        residual = self.dropout(residual)
        x = self.ff_ln(x + residual)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 max_len: int,
                 pad_idx: int,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 drop_p: float,
                 n_layers: int,
                 device: str):
        super().__init__()

        self.embedding = PositionalEmbedding(vocab_size, d_model, max_len, drop_p, pad_idx, device)
        self.blocks = nn.ModuleList(EncoderLayer(d_model, n_heads, d_ff, drop_p) for _ in range(n_layers))

    def forward(self,
                x: TensorType["batch", "seq_len", "d_model"],
                encoder_mask: Optional[TensorType["batch", "n_heads", "seq_len", "seq_len"]] = None
                ):
        x = self.embedding(x)

        for block in self.blocks:
            x = block(x, encoder_mask)

        return x


# if __name__ == '__main__':
#     import torch
#     src = torch.tensor([[123, 123, 230, 1234, 10], [30, 1, 2, 0, 3], [30, 1, 2, 0, 3]]).to('cuda')
#     target = torch.tensor([[2, 15, 2, 42, 1, 1, 1, 1], [302, 1, 2452, 20, 3, 1, 1, 1], [322, 145, 122, 40, 23, 3, 402, 1]]).to('cuda')
#     encoder = Encoder(5000, 100, 1, 512, 8, 2048, 0.1, 6, 'cuda').to('cuda')
#     encoder_out = encoder(src)
#     print(encoder_out.shape)
