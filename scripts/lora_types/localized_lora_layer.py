import torch
import torch.nn as nn
import math

class LocalizedLoraLayer(nn.Module):
    """
    Localized LoRA layer with KxK blockwise low-rank updates.

    This layer applies a blockwise low-rank adaptation to a frozen linear layer, generalizing standard LoRA
    to a localized, spatially-aware variant. Each block of the weight matrix gets its own low-rank matrices.

    Parameters
    ----------
    wt : nn.Module
        The pre-trained linear layer whose weights are frozen and to which the LoRA update is applied.
        Typically nn.Linear.
    d : int
        Dimension of the input and output of the linear layer (assumes square weight matrix d x d).
    r_block : int
        Rank of the low-rank matrices A_ij and B_ij for each block. Typically much smaller than d/K.
    alpha : int or float
        Scaling factor applied to the low-rank update. Analogous to the alpha in standard LoRA.
    K : int
        Number of blocks per row/column. The weight matrix is split into KxK blocks. 
        Each block has its own low-rank matrices.
    
    Attributes
    ----------
    block_size : int
        Size of each block: d // K.
    A_blocks : nn.ModuleList
        Nested ModuleList containing A_ij matrices for each block (KxK blocks).
    B_blocks : nn.ModuleList
        Nested ModuleList containing B_ij matrices for each block (KxK blocks).

    Notes
    -----
    - If K=1, this reduces to standard LoRA.
    - Each block (i,j) has update: delta_W_ij = B_ij @ A_ij.
    - Final weight update is the sum of all blockwise updates: W* = W + sum_ij delta_W_ij.
    """
    def __init__(self, wt: nn.Module, d: int, r_block: int, alpha: int, K: int):
        super().__init__()
        self.wt = wt  # Frozen weight layer
        self.d = d  # Input/output dimension
        self.r_block = r_block  # Rank of low-rank updates per block
        self.alpha = alpha  # Scaling factor
        self.K = K  # Number of blocks per row/column
        self.block_size = d // K  # Dimension of each block

        # Create A_ij and B_ij matrices for each block
        self.A_blocks = nn.ModuleList()
        self.B_blocks = nn.ModuleList()
        for i in range(K):
            A_row, B_row = nn.ModuleList(), nn.ModuleList()
            for j in range(K):
                A_row.append(nn.Linear(self.block_size, r_block, bias=False))
                B_row.append(nn.Linear(r_block, self.block_size, bias=False))
            self.A_blocks.append(A_row)
            self.B_blocks.append(B_row)

        self.init_blocks()

    def init_blocks(self):
        """Initialize the A matrices with Kaiming uniform and B matrices to zero."""
        for A_row in self.A_blocks:
            for A in A_row:
                nn.init.kaiming_uniform_(A.weight, a=math.sqrt(5))
        for B_row in self.B_blocks:
            for B in B_row:
                nn.init.zeros_(B.weight)
    
    def count_parameters(self):
        total = 0
        for A_row, B_row in zip(self.A_blocks, self.B_blocks):
            for A, B in zip(A_row, B_row):
                total += A.weight.numel() + B.weight.numel()
        return total

    def forward(self, x):
        """
        Forward pass for the localized LoRA layer.
        Parameters:
        x : Tensor
            Input tensor of shape (batch_size d).
        
        Returns:
        Tensor
            Output tensor of shape (batch_size, d) with localized low-rank updates applied.
        """
        out = self.wt(x)# Original linear output

        # Compute blockwise low-rank updates
        delta = torch.zeros_like(out)

        # Partition along the hidden dimension
        for i in range(self.K):
            x_block_i = x[:, :, i*self.block_size:(i+1)*self.block_size]  # (batch, tokens (197), block_size(768))
            for j in range(self.K):
                # Apply A_ij and B_ij to the block
                delta_block = self.B_blocks[i][j](self.A_blocks[i][j](x_block_i))
                delta[:, :, j*self.block_size:(j+1)*self.block_size] += delta_block


        return out + (self.alpha / self.r_block) * delta
