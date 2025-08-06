# Contains implementation of LoRA and its variants
import torch
import torch.nn as nn
from torch import Tensor


### implement the LoRAlayer
class LoraLayer(nn.Module):
    """
    This class implements the LoRA layer
    wt_linear: Weight (which would be left frozen)
    A,B: Lower matrices which constitute delta W
    rank_lora: Rank of A and B matrices
    alpha: some weighing factor
    """
    def __init__(self, wt: nn.Module, A: nn.Module, B: nn.Module, rank_lora: int, alpha: int):
        super().__init__()
        self.wt = wt
        self.A, self.B = A, B
        self.rank = rank_lora
        self.alpha = alpha

    def forward(self,x):

        x=self.wt(x) + (self.alpha / self.rank) * self.B(self.A(x))
        return x