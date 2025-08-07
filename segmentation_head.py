import torch.nn.functional as F
import torch.nn as nn

class CustomSegHead(nn.Module):
    """ 
    Custom defined segmentation head. This module takes the patch embeddings from a ViT backbone and
    processes them into a full-resolution segmentation map

    Arguments:
    ----------
    hidden_dim : The dimension of the patch embeddings from the ViT (e.g., 768 for ViT-Base).
        
    num_classes : The number of output segmentation classes (e.g., 21 for PASCAL VOC, 10 for pets dataset).
        
    patch_size : The size of each image patch in pixels (e.g., 16 for 16x16 patches). This also determines how much to upsample the output feature map.

    image_size : The height/width of the input image in pixels (assumes square images). Used to calculate how many patches per spatial dimension.
    """
    def __init__(self, hidden_dim:int, num_classes:int, patch_size:int, image_size:int):
        super().__init__()
        
        # Store the patch size (e.g., 16 for 16x16 patches)
        self.patch_size = patch_size
        
        # Calculate the number of patches per spatial dimension (assuming square image and patches)
        self.num_patch_per_dim = image_size // patch_size

        # First conv layer: reduces channels from hidden_dim to half, with 3x3 kernel for local spatial context
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        
        # ReLU activation after conv1 for non-linearity
        self.relu = nn.ReLU()
        
        # Final conv layer: maps features to the number of classes with 1x1 convolution (pixel-wise classification)
        self.conv2 = nn.Conv2d(hidden_dim // 2, num_classes, kernel_size=1)

    @property
    def num_trainable_params(self):
        """Number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):  # x shape: (B, N, D)
        """
        Input x's shape: (B,N,D)
        where B= Extract batch size, N= number of patches, and D= embedding dimension.
        """
        # Extract batch size (B), number of patches (N), and embedding dimension (D)
        B, N, D = x.shape 
        
        # Calculate height and width of patch grid (assume square grid)
        H = W = self.num_patch_per_dim
        # Note: N= HXW
        
        # current shape is : (B,N,D)
        # Rearrange tensor to (B, D, H, W) so it can be processed by Conv2d layers
        # 1- permute swaps the dimensions so channels come before spatial dims
            # -- Swaps dimensions 1 and 2 -> new shape is: (B,D,N) 
        # 2- reshape organizes tokens into 2D spatial layout: (B,D,N)--> (B,D,H,W)
        x = x.permute(0, 2, 1).reshape(B, D, H, W)
        
        # Apply first convolution and ReLU activation to learn local spatial features
        x = self.relu(self.conv1(x))
        
        # Apply final 1x1 convolution to produce per-class scores for each spatial location
        x = self.conv2(x)
        
        # Upsample output to match original image resolution
        # scale_factor = patch size because each patch corresponds to patch_size x patch_size pixels
        x = F.interpolate(x, scale_factor=self.patch_size, mode='bilinear', align_corners=False)
        
        # Return segmentation logits of shape (B, num_classes, H_img, W_img)
        return x
