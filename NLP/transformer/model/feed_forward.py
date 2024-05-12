import torch.nn as nn
from torchtyping import TensorType
from typing import Optional


class FeedForward(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 drop_p: float):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_ff, d_model)
        )

    def forward(self,
                x: Optional[TensorType["batch", "seq_len", "d_model"]] = None):
        return self.ff(x)
