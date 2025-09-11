import torch
import torch.nn as nn

class RepLoraLayer(nn.Module):
    def __init__(self, wt: nn.Module, 
                A_gen, B_gen, 
                rank_lora, alpha, 
                target="q"):
        super().__init__()
        self.wt = wt
        for p in self.wt.parameters():
            p.requires_grad = False
        # other parameters
        self.rank = rank_lora
        self.alpha = alpha
        self.target = target  # "q" or "v"
        # build lora A, B layers
        self.A_gen = A_gen
        self.B_gen = B_gen

    def forward(self, x):
        base_out = self.wt(x)

        A_q, A_v = self.A_gen()
        B_q, B_v = self.B_gen()

        if self.target == "q":
            A_tensor, B_tensor = A_q, B_q
        else:
            A_tensor, B_tensor = A_v, B_v

        delta = (x @ A_tensor) @ B_tensor
        return base_out + (self.alpha / self.rank) * delta
