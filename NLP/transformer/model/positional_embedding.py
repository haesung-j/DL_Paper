import torch
import torch.nn as nn
from torchtyping import TensorType


class PositionalEmbedding(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 max_len: int,
                 drop_p: float,
                 pad_idx: int,
                 device: str
                 ):
        super().__init__()
        self.scale = torch.sqrt(torch.tensor(d_model)).to(device)
        self.device = device

        self.word_embedding_layer = nn.Embedding(vocab_size, d_model, pad_idx)
        self.positional_embedding_layer = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(drop_p)

    def forward(self,
                x: TensorType["batch", "seq_len"]):

        word_embedding = self.word_embedding_layer(x)

        position_matrix = torch.arange(x.shape[1]).repeat(x.shape[0], 1).to(self.device)
        positional_embedding = self.positional_embedding_layer(position_matrix)

        embeddings = self.scale * word_embedding + positional_embedding
        embeddings = self.dropout(embeddings)

        return embeddings


# if __name__=='__main__':
#     x = torch.tensor([[123, 123, 230, 1234, 10], [30, 1, 2, 0, 3], [30, 1, 2, 0, 3]]).to('cuda')
#     print(x.shape)
#
#     p = PositionalEmbedding(2048, 512, 100, 0.1, 0, 'cuda').to('cuda')
#     print(p(x).shape)
#     print(x)
