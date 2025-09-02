import torch
import torch.nn as nn
from torch import Tensor
import math

#Serial LoRA method
class SerialLoraLayer(nn.Module):
    def __init__(self, dim_in: int, rank: int):
        super().__init__()
        self.A = nn.Linear(dim_in, rank, bias=False)
        self.B = nn.Linear(rank, dim_in, bias=False)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        # serial LoRA: (I + BA)x
        return x + self.B(self.A(x))