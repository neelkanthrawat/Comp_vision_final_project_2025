import torch.nn as nn
import torch
from torch import Tensor
import math
from safetensors.torch import save_file, load_file
from .serial_lora_layer import SerialLoraLayer  # your separate Serial LoRA implementation

class SerialLoraVit(nn.Module):
    """ 
    This class introduces Serial LoRA layers to a ViT model.
    """
    def __init__(self, vit_model, r: int, lora_layers=None):
        """
        vit_model: pre-trained ViT model
        r: rank of the Serial LoRA matrices
        lora_layers: list of transformer blocks to apply Serial LoRA to
        """
        super().__init__()

        assert r > 0, "r (rank of Serial LoRA matrices) must be >0"

        # choose which layers to apply serial LoRA
        if lora_layers:
            self.lora_layers = lora_layers
        else:
            self.lora_layers = list(range(len(vit_model.encoder.layer)))

        # input dimension of transformer
        dim = vit_model.encoder.layer[0].attention.attention.query.in_features

        # freeze the original ViT parameters
        for param in vit_model.parameters():
            param.requires_grad = False

        # list to store Serial LoRA layers
        self.serial_lora_layers = nn.ModuleList()

        # replace Q/K/V input with Serial LoRA branch
        for layer_i, block_i in enumerate(vit_model.encoder.layer):
            if layer_i not in self.lora_layers:
                self.serial_lora_layers.append(None)
                continue

            # create a single shared Serial LoRA layer for this block
            serial_lora = SerialLoraLayer(dim, r)
            self.serial_lora_layers.append(serial_lora)

            # wrap the original forward of the block to include Serial LoRA
            # we will override the attention forward later in forward()
            # so nothing else needed here

        self.model_vit_serial = vit_model

    def save_lora_params(self, filename: str):
        """Save all Serial LoRA parameters."""
        assert filename.endswith(".safetensors"), "Filename must have .safetensors extension"
        state_dict = {}
        state_dict['lora_layers'] = torch.tensor(self.lora_layers, dtype=torch.int32)

        for i, lora_layer in enumerate(self.serial_lora_layers):
            if lora_layer is None:
                continue
            state_dict[f'Serial_A_{i}'] = lora_layer.A.weight.data
            state_dict[f'Serial_B_{i}'] = lora_layer.B.weight.data

        save_file(state_dict, filename)
        print(f"Saved Serial LoRA params to {filename}")

    def load_lora_params(self, filename: str):
        """Load Serial LoRA parameters."""
        assert filename.endswith(".safetensors"), "Filename must have .safetensors extension"
        loaded = load_file(filename)
        loaded_layers = loaded['lora_layers'].tolist()

        if loaded_layers != self.lora_layers:
            print("Warning: loaded layers differ from current model. Adjusting...")
            self.lora_layers = loaded_layers

        for i, lora_layer in enumerate(self.serial_lora_layers):
            if lora_layer is None:
                continue
            lora_layer.A.weight.data.copy_(loaded[f'Serial_A_{i}'])
            lora_layer.B.weight.data.copy_(loaded[f'Serial_B_{i}'])

        print(f"Loaded Serial LoRA params from {filename}")

    @property
    def num_trainable_params(self):
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Run the ViT with Serial LoRA applied to selected blocks."""
        # iterate through transformer blocks
        for i, block_i in enumerate(self.model_vit_serial.encoder.layer):
            if self.serial_lora_layers[i] is not None:
                # apply Serial LoRA to the input before Q/K/V
                lora_layer = self.serial_lora_layers[i]
                block_i_input = x
                x_adapted = lora_layer(block_i_input)

                # override the Q/K/V projections inside this block
                block_i.attention.attention.query_out = block_i.attention.attention.query(x_adapted)
                block_i.attention.attention.key_out = block_i.attention.attention.key(x_adapted)
                block_i.attention.attention.value_out = block_i.attention.attention.value(x_adapted)

                # forward through the rest of the block normally
                x = block_i(x, **kwargs)
            else:
                x = block_i(x, **kwargs)

        return x


# 2nd implementation

class SerialLoraVitPrintable(nn.Module):
    """
    ViT model with Serial LoRA integrated directly inside each transformer block.
    """
    def __init__(self, vit_model, r: int, lora_layers=None):
        super().__init__()

        assert r > 0, "rank r must be > 0"

        # choose which transformer blocks to modify
        if lora_layers:
            self.lora_layers = lora_layers
        else:
            self.lora_layers = list(range(len(vit_model.encoder.layer)))

        # input dimension for the transformer
        dim = vit_model.encoder.layer[0].attention.attention.query.in_features

        # freeze all original ViT parameters
        for param in vit_model.parameters():
            param.requires_grad = False

        # attach Serial LoRA layers directly inside blocks
        for idx, block in enumerate(vit_model.encoder.layer):
            if idx in self.lora_layers:
                block.serial_lora = SerialLoraLayer(dim, r)
            else:
                block.serial_lora = None  # for consistency

        self.model_vit_serial = vit_model

    def save_lora_params(self, filename: str):
        assert filename.endswith(".safetensors"), "Filename must have .safetensors extension"
        state_dict = {}
        state_dict['lora_layers'] = torch.tensor(self.lora_layers, dtype=torch.int32)

        for i, block in enumerate(self.model_vit_serial.encoder.layer):
            if block.serial_lora is None:
                continue
            state_dict[f'Serial_A_{i}'] = block.serial_lora.A.weight.data
            state_dict[f'Serial_B_{i}'] = block.serial_lora.B.weight.data

        save_file(state_dict, filename)
        print(f"Saved Serial LoRA params to {filename}")

    def load_lora_params(self, filename: str):
        assert filename.endswith(".safetensors"), "Filename must have .safetensors extension"
        loaded = load_file(filename)
        loaded_layers = loaded['lora_layers'].tolist()

        if loaded_layers != self.lora_layers:
            print("Warning: loaded layers differ from current model. Adjusting...")
            self.lora_layers = loaded_layers

        for i, block in enumerate(self.model_vit_serial.encoder.layer):
            if block.serial_lora is None:
                continue
            block.serial_lora.A.weight.data.copy_(loaded[f'Serial_A_{i}'])
            block.serial_lora.B.weight.data.copy_(loaded[f'Serial_B_{i}'])

        print(f"Loaded Serial LoRA params from {filename}")

    @property
    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Forward pass through the ViT with Serial LoRA applied to selected blocks.
        """
        for idx, block in enumerate(self.model_vit_serial.encoder.layer):
            if block.serial_lora is not None:
                x_adapted = block.serial_lora(x)
                # replace Q/K/V input with adapted features
                # assuming the attention forward uses x directly
                block_input = x_adapted
                x = block(block_input, **kwargs)
            else:
                x = block(x, **kwargs)
        return x
    
### third implementation idea:
class SerialLoraInjectedLinear(nn.Module):
    """
    Wraps an nn.Linear layer to apply Serial LoRA: (I + BA)x before linear projection
    """
    def __init__(self, linear: nn.Linear, rank: int):
        super().__init__()
        self.linear = linear
        dim_in = linear.in_features
        self.serial_lora = SerialLoraLayer(dim_in, rank)

    def forward(self, x):
        x_adapted = self.serial_lora(x)
        return self.linear(x_adapted)


class SerialLoraVit(nn.Module):
    """
    ViT with Serial LoRA injected into attention projections (Q, K, V)
    """
    def __init__(self, vit_model, r: int, lora_layers=None):
        super().__init__()

        assert r > 0, "rank r must be > 0"

        # decide which blocks to modify
        if lora_layers:
            self.lora_layers = lora_layers
        else:
            self.lora_layers = list(range(len(vit_model.encoder.layer)))

        # freeze all original parameters
        for param in vit_model.parameters():
            param.requires_grad = False

        # inject Serial LoRA into each block’s Q/K/V projections
        for idx, block in enumerate(vit_model.encoder.layer):
            if idx in self.lora_layers:
                # wrap attention linear layers
                block.attention.attention.query = SerialLoraInjectedLinear(
                    block.attention.attention.query, r
                )
                block.attention.attention.key = SerialLoraInjectedLinear(
                    block.attention.attention.key, r
                )
                block.attention.attention.value = SerialLoraInjectedLinear(
                    block.attention.attention.value, r
                )

        self.model_vit_serial = vit_model

    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: Tensor, **kwargs):
        return self.model_vit_serial(x, **kwargs)

    def save_lora_params(self, filename: str):
        """
        Save all Serial LoRA parameters
        """
        assert filename.endswith(".safetensors"), "Filename must end with .safetensors"
        state_dict = {}
        state_dict['lora_layers'] = torch.tensor(self.lora_layers, dtype=torch.int32)

        for i, block in enumerate(self.model_vit_serial.encoder.layer):
            if i not in self.lora_layers:
                continue
            for name in ['query', 'key', 'value']:
                linear_layer = getattr(block.attention.attention, name)
                state_dict[f'{name}_A_{i}'] = linear_layer.serial_lora.A.weight.data
                state_dict[f'{name}_B_{i}'] = linear_layer.serial_lora.B.weight.data

        save_file(state_dict, filename)
        print(f"Saved Serial LoRA params to {filename}")

    def load_lora_params(self, filename: str):
        """
        Load Serial LoRA parameters
        """
        assert filename.endswith(".safetensors"), "Filename must end with .safetensors"
        loaded = load_file(filename)
        loaded_layers = loaded['lora_layers'].tolist()

        if loaded_layers != self.lora_layers:
            print("Warning: loaded layers differ from current model. Adjusting...")
            self.lora_layers = loaded_layers

        for i, block in enumerate(self.model_vit_serial.encoder.layer):
            if i not in self.lora_layers:
                continue
            for name in ['query', 'key', 'value']:
                linear_layer = getattr(block.attention.attention, name)
                linear_layer.serial_lora.A.weight.data.copy_(loaded[f'{name}_A_{i}'])
                linear_layer.serial_lora.B.weight.data.copy_(loaded[f'{name}_B_{i}'])

        print(f"Loaded Serial LoRA params from {filename}")

