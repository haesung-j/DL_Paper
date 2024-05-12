import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import Optional

from model.multi_head_attention import MultiHeadAttention
from model.feed_forward import FeedForward
from model.positional_embedding import PositionalEmbedding


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop_p):
        super().__init__()

        self.masked_multi_head_attention = MultiHeadAttention(d_model, n_heads)
        self.mmha_ln = nn.LayerNorm(d_model)

        self.encoder_decoder_attention = MultiHeadAttention(d_model, n_heads)
        self.eda_ln = nn.LayerNorm(d_model)

        self.feed_forwad = FeedForward(d_model, d_ff, drop_p)
        self.ff_ln = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, encoder_out, decoder_mask, encoder_decoder_mask):
        residual = self.masked_multi_head_attention(k=x, q=x, v=x, mask=decoder_mask)
        residual = self.dropout(residual)
        x = self.mmha_ln(x + residual)

        residual = self.encoder_decoder_attention(q=x, k=encoder_out, v=encoder_out, mask=encoder_decoder_mask)
        residual = self.dropout(residual)
        x = self.eda_ln(x + residual)

        x = self.feed_forwad(x)
        residual = self.dropout(residual)
        x = self.ff_ln(x + residual)

        return x


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 max_len: int,
                 pad_idx: int,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 drop_p: float,
                 n_layers: int,
                 device: str
                 ):
        super().__init__()

        self.embeddings = PositionalEmbedding(vocab_size, d_model, max_len, drop_p, pad_idx, device)
        self.blocks = nn.ModuleList(DecoderLayer(d_model, n_heads, d_ff, drop_p) for _ in range(n_layers))
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self,
                target: TensorType["batch", "seq_len(target)"],
                encoder_out: TensorType["batch", "seq_len(src)", "d_model"],
                decoder_mask: TensorType["batch", "n_heads", "seq_len(target)", "seq_len(target)"],
                encoder_decoder_mask: TensorType["batch", "n_heads", "seq_len(target)", "seq_len(src)"]
                ):
        target = self.embeddings(target)

        for block in self.blocks:
            target = block(target, encoder_out, decoder_mask, encoder_decoder_mask)

        out = self.classifier(target)
        return out


# if __name__ == '__main__':
#     src = torch.tensor([[123, 123, 230, 1234, 10], [30, 1, 2, 0, 3], [30, 1, 2, 0, 3]]).to('cuda')
#     target = torch.tensor([[2, 15, 2, 42, 1, 1, 1, 1], [302, 1, 2452, 20, 3, 1, 1, 1], [322, 145, 122, 40, 23, 3, 402, 1]]).to('cuda')
#     from encoder import Encoder
#     encoder = Encoder(5000, 100, 1, 512, 8, 2048, 0.1, 6, 'cuda').to('cuda')
#     encoder_out = encoder(src)
#
#     dl = Decoder(5000, 100, 1, 512, 8, 2048, 0.1, 6, 'cuda').to('cuda')
#     decoder_output = dl(target, encoder_out, None, None)
#     print(decoder_output.shape)



