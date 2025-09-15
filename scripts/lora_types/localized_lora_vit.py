import torch
import torch.nn as nn
import math
from safetensors.torch import save_file, load_file
from localized_lora_layer import LocalizedLoraLayer  # Assume your LocalizedLoRA is in this module

class LocalizedLoraVit(nn.Module):
    """
    ViT wrapper with Localized LoRA layers for query and value projections.
    Each LoRA layer is split into KxK blocks, with rank r_block per block.
    """
    def __init__(self, vit_model, 
                r_block: int, alpha: int, 
                num_blocks_per_row: int = 4, 
                lora_layers=None):
        super().__init__()
        self.vit_model = vit_model
        self.r_block = r_block
        self.alpha = alpha
        self.K = num_blocks_per_row # number of blocks per row/column

        # Determine which layers to apply LoRA to
        if lora_layers:
            self.lora_layers_idx = lora_layers
        else:
            self.lora_layers_idx = list(range(len(vit_model.encoder.layer)))
        
        # Dimension of the input vector to the transformer
        dim = vit_model.encoder.layer[0].attention.attention.query.in_features 

        # Freeze original ViT parameters
        for param in vit_model.parameters():
            param.requires_grad = False

        # Store Localized LoRA layers
        self.lora_layers_objs = nn.ModuleList()

        # Replace Q and V projections with Localized LoRA
        for layer_i, block_i in enumerate(vit_model.encoder.layer):
            if layer_i not in self.lora_layers_idx:
                continue

            #dim = block_i.attention.attention.query.in_features

            lora_q = LocalizedLoraLayer(block_i.attention.attention.query,
                                        d=dim, r_block=self.r_block,
                                        alpha=self.alpha, K=self.K)
            lora_v = LocalizedLoraLayer(block_i.attention.attention.value,
                                        d=dim, r_block=self.r_block,
                                        alpha=self.alpha, K=self.K)

            block_i.attention.attention.query = lora_q
            block_i.attention.attention.value = lora_v

            self.lora_layers_objs.append(lora_q)
            self.lora_layers_objs.append(lora_v)

    @property
    def num_trainable_params(self):
        """Return total number of trainable parameters in all LoRA blocks."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, **kwargs):
        """Forward pass through the ViT with Localized LoRA layers."""
        return self.vit_model(x, **kwargs)

    def save_lora_params(self, filename: str):
        """Save all LoRA block parameters in safetensors format."""
        assert filename.endswith(".safetensors"), "Filename must end with .safetensors"
        state_dict = {}

        for idx, lora_layer in enumerate(self.lora_layers_objs):
            for i in range(self.K):
                for j in range(self.K):
                    state_dict[f'l{idx}_A_{i}_{j}'] = lora_layer.A_blocks[i][j].weight.data
                    state_dict[f'l{idx}_B_{i}_{j}'] = lora_layer.B_blocks[i][j].weight.data

        # Save the layer indices as well
        state_dict['lora_layers_idx'] = torch.tensor(self.lora_layers_idx, dtype=torch.int32)
        save_file(state_dict, filename)
        print(f"Saved Localized LoRA parameters to {filename}")

    def load_lora_params(self, filename: str):
        """Load LoRA block parameters from safetensors format."""
        assert filename.endswith(".safetensors"), "Filename must end with .safetensors"
        loaded = load_file(filename)

        # Load layer indices and update if different
        loaded_layers_idx = loaded['lora_layers_idx'].tolist()
        if loaded_layers_idx != self.lora_layers_idx:
            print("Warning: loaded LoRA layers differ from current model. Updating indices.")
            self.lora_layers_idx = loaded_layers_idx

        for idx, lora_layer in enumerate(self.lora_layers_objs):
            for i in range(self.K):
                for j in range(self.K):
                    lora_layer.A_blocks[i][j].weight.data.copy_(loaded[f'l{idx}_A_{i}_{j}'])
                    lora_layer.B_blocks[i][j].weight.data.copy_(loaded[f'l{idx}_B_{i}_{j}'])

        print(f"Loaded Localized LoRA parameters from {filename}")
