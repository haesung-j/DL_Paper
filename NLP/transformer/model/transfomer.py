import torch
import torch.nn as nn
from torchtyping import TensorType
from torchinfo import summary

from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, pad_idx, d_model, n_heads, d_ff, drop_p, n_layers, device):
        super().__init__()
        self.pad_idx = pad_idx
        self.n_heads = n_heads
        self.device = device

        self.encoder = Encoder(vocab_size, max_len, pad_idx, d_model, n_heads, d_ff, drop_p, n_layers, device)
        self.decoder = Decoder(vocab_size, max_len, pad_idx, d_model, n_heads, d_ff, drop_p, n_layers, device)

    def forward(self, x, y):
        encoder_mask = self.make_encoder_mask(x)
        decoder_mask = self.make_decoder_mask(y)
        encoder_decoder_mask = self.make_encoder_decoder_mask(x, y)

        encoder_out = self.encoder(x, encoder_mask)
        out = self.decoder(y, encoder_out, decoder_mask, encoder_decoder_mask)
        return out

    def make_encoder_mask(self,
                          x: TensorType["batch_size", "x_seq_len"]
                          ):
        """
        bs x n_heads x (x_seq_len) x (x_seq_len)
        for each bs x n_heads (let x_seq_len=3)
        F F T
        F F T
        F F T
        """
        encoder_mask = (x == self.pad_idx).unsqueeze(1).unsqueeze(1)
        encoder_mask = encoder_mask.expand(x.shape[0], self.n_heads, x.shape[1], x.shape[1])
        return encoder_mask

    def make_decoder_mask(self,
                          y: TensorType["batch_size", "y_seq_len"]
                          ):
        """
        bs x n_heads x (y_seq_len) x (y_seq_len)
        for each bs x n_heads (let x_seq_len=4)
        F F F T        F T T T      F T T T
        F F F T   or   F F T T  =>  F F T T
        F F F T        F F F T      F F F T
        F F F T        F F F F      F F F T
        """
        pad_mask = (y == self.pad_idx).unsqueeze(1).unsqueeze(1).to(self.device)
        pad_mask = pad_mask.expand(y.shape[0], self.n_heads, y.shape[1], y.shape[1])

        future_mask = torch.tril(torch.ones(y.shape[0], self.n_heads, y.shape[1], y.shape[1])).to(self.device)
        future_mask = (future_mask == 0)

        decoder_mask = pad_mask | future_mask
        return decoder_mask

    def make_encoder_decoder_mask(self,
                                  x: TensorType["batch_size", "x_seq_len"],
                                  y: TensorType["batch_size", "y_seq_len"]
                                  ):
        """
        query from y / key from x -> pad masking은 key를 기준으로 함
        bs x n_heads x (y_seq_len) x (x_seq_len)
        """
        encoder_decoder_mask = (x == self.pad_idx).unsqueeze(1).unsqueeze(1).to(self.device)
        encoder_decoder_mask = encoder_decoder_mask.expand(y.shape[0], self.n_heads, y.shape[1], x.shape[1])
        return encoder_decoder_mask


# if __name__ == '__main__':
#     src = torch.tensor([[23, 2, 1, 103, 0]])
#     trg = torch.tensor([[23, 2, 1, 103, 3]])
#
#     model = Transformer(65001, 100, 0, 512, 8, 2048, 0.1, 6, 'cpu')
#     print(summary(model, input_data=(src, trg)))
