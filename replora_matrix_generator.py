import torch
import torch.nn as nn

class RepLoRAGenerator(nn.Module):
    """
    Generates low-rank matrices A_q/A_v or B_q/B_v
    from a trainable diagonal seed via a 2-layer MLP
    with two distinct output projections.
    """
    def __init__(self, r:int, dim:int, kind:str="A"):
        super().__init__()
        self.r = r
        self.dim = dim
        self.kind = kind   # "A" => (r, dim), "B" => (dim, r)

        # trainable diagonal seed (vector form)
        self.seed = nn.Parameter(torch.randn(r))

        # shared trunk MLP
        hidden_dim = max(4 * r, 64)
        self.trunk = nn.Sequential(
            nn.Linear(r, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # two distinct outputs: one for Q, one for V
        out_dim = r * dim if kind == "A" else dim * r
        self.output_q = nn.Linear(hidden_dim, out_dim)
        self.output_v = nn.Linear(hidden_dim, out_dim)

    def forward(self):
        # pass seed through trunk
        x = self.seed.unsqueeze(0)    # (1, r)
        h = self.trunk(x)             # (1, hidden_dim)

        # GENERATE Q and V matrices
        ## project through output layers
        raw_q = self.output_q(h)      # shape: (1, r*dim) if kind="A", else (1, dim*r)
        raw_v = self.output_v(h)      # same

        ## reshape into matrices
        if self.kind == "A":
            A_q = raw_q.view(self.dim,self.r)   # shape: (dim, r)
            A_v = raw_v.view(self.dim,self.r)   # shape: (dim,r)            return A_q, A_v
        else: # This is for B
            B_q = raw_q.view(self.r, self.dim)   # shape: (r, dim)
            B_v = raw_v.view(self.r, self.dim)   # shape: (r, dim)
            return B_q, B_v
