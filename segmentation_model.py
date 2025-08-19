import torch.nn as nn
from segmentation_head import CustomSegHead

class SegViT(nn.Module):
    """
    Wraps a ViT model and applies a custom segmentation head to the output.
    Converts patch embeddings into full-resolution segmentation masks.
    """

    def __init__(self,vit_model: nn.Module,## vit_model us vit based backbone
                    image_size: int,
                    patch_size: int,
                    dim: int, # hidden dimension
                    n_classes: int,
                    head=None, # segmentation model
                    ) -> None:
        super().__init__()
        
        self.vit = vit_model

        # Remove classification head if present
        if hasattr(self.vit, "fc"):
            del self.vit.fc
        elif hasattr(self.vit, "lora_vit") and hasattr(self.vit.lora_vit, "fc"):
            del self.vit.lora_vit.fc

        # Use custom segmentation head
        if head==None:
            self.seg_head = CustomSegHead(
                hidden_dim=dim,
                num_classes=n_classes,
                patch_size=patch_size,
                image_size=image_size
            )
        else:
            self.seg_head = head

    @property
    def num_trainable_params(self):
        """Number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # note for me: maybe have an argument stating whether wanna remove the extra class token or not (if it is already removed)
    def forward(self, x): # shape of x: (B,N+1,D)
        """
        Forward pass:
        1. Get ViT patch embeddings (B, N+1, D)
        2. Remove class token (CLS) → (B, N, D)
        3. Feed to custom segmentation head
        """

        x = self.vit(x)  # (B, N+1, D)

        # get the last state
        x= x.last_hidden_state # (B,N+1,D)

        #x = x[:, :-1, :]  # Remove CLS token → (B, N, D)
        # I think it should be this:
        x = x[:,1:,:] # because it is the first token which is
        x = self.seg_head(x)  # (B, num_classes, H, W)
        return x