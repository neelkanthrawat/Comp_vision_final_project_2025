import torch
import torch.nn as nn


class RepLoraLayer(nn.Module):
    def __init__(self, wt: nn.Module, 
                A_gen, B_gen, 
                rank_lora, alpha, 
                target="q"):
        super().__init__()
        self.wt = wt
        self.A_gen = A_gen
        self.B_gen = B_gen
        self.rank = rank_lora
        self.alpha = alpha
        self.target = target  # "q" or "v"

        for p in self.wt.parameters():
            p.requires_grad = False

    def forward(self, x):
        base_out = self.wt(x)

        # Generate A and B
        A_q, A_v = self.A_gen()
        B_q, B_v = self.B_gen()

        if self.target == "q":
            A, B = A_q, B_q
        else:
            A, B = A_v, B_v

        delta = torch.matmul(A.T, x.T)# (r, batch)
        delta = torch.matmul(B, delta)# (out_dim, batch)
        delta = delta.T

        return base_out + (self.alpha / self.rank) * delta
