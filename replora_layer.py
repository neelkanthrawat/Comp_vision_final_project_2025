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
        self.A, self.B = self._make_lora()

    def _make_lora(self):
        """Create LoRA A and B layers initialized from generators."""
        A_q, A_v = self.A_gen()
        B_q, B_v = self.B_gen()

        if self.target == "q":
            A_tensor, B_tensor = A_q, B_q
        else:
            A_tensor, B_tensor = A_v, B_v

        in_dim, r = A_tensor.shape   # (in_features, rank)
        _, out_dim = B_tensor.shape  # (rank, output_feature)
        assert out_dim == in_dim, "out_dim should be same as in_dim."
        #print(f"shape of A_tensor is: {A_tensor.shape} and B_tensor is: {B_tensor.shape}")

        A_layer = nn.Linear(in_dim, r, bias=True)
        B_layer = nn.Linear(r, out_dim, bias=True)
        #print(f" A_layer is: {A_layer} and B_layer is: {B_layer}")

        with torch.no_grad():
            A_layer.weight.copy_(A_tensor.T)  # nn.Linear expects (out_features, in_features)
            B_layer.weight.copy_(B_tensor.T)

        return A_layer, B_layer

    def forward(self, x):
        base_out = self.wt(x) # (batch_size, 197, 768)
        #print(f"x shape is: {x.shape} and base_out's shape is: {base_out.shape}")
        delta = self.B(self.A(x))
        return base_out + (self.alpha / self.rank) * delta
