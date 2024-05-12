import torch
import torch.nn as nn
from einops import rearrange
from torchtyping import TensorType
from typing import Optional


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int):
        super().__init__()

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

        self.d_k = torch.sqrt(torch.tensor(d_model/n_heads))
        self.n_heads = n_heads

    def forward(self,
                q: TensorType["batch", "seq_len", "d_model"],
                k: TensorType["batch", "seq_len", "d_model"],
                v: TensorType["batch", "seq_len", "d_model"],
                mask: Optional[TensorType["batch", "n_heads", "seq_len", "seq_len"]] = None
                ) -> TensorType["batch", "seq_len", "d_model"]:
        """
        1. query, key, value 생성
        2. batch_size x num_heads x num_words x d_model/num_heads 로 분할
        3. 각 head마다 Q_i, K_i, V_i, 어텐션
            3.1. 단, mask 여부에 따라 softmax 통과 전 처리
        4. batch_size x num_words x d_model 로 concat
        5. FC Layer 통과
        """
        query = self.fc_q(q)
        key = self.fc_q(k)
        value = self.fc_q(v)

        query = rearrange(query, "batch seq_len (n_heads d_k) -> batch n_heads seq_len d_k", n_heads=self.n_heads)
        key = rearrange(key, "batch seq_len (n_heads d_k) -> batch n_heads seq_len d_k", n_heads=self.n_heads)
        value = rearrange(value, "batch seq_len (n_heads d_k) -> batch n_heads seq_len d_k", n_heads=self.n_heads)

        attention_scores = query @ key.transpose(-2, -1) / self.d_k

        # masking - encoder: pad, decoder: pad|future, encoder-decoder: pad of encoder_output
        if mask is not None:
            attention_scores[mask] = -1e10
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention = attention_weights @ value

        attention = rearrange(attention, "batch n_heads seq_len d_k -> batch seq_len (n_heads d_k)")

        attention = self.linear(attention)
        return attention


# if __name__=='__main__':
#     embeddings = torch.randn(2, 10, 512)
#     mha = MultiHeadAttention(d_model=512, n_heads=8)
#     print(mha(embeddings, embeddings, embeddings).shape)