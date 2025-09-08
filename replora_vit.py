import torch
import torch.nn as nn
from torch import Tensor
from replora_generator import RepLoRAGenerator
from replora_layer import RepLoraLayer  # the one with target="q"/"v"
from safetensors.torch import save_file, load_file

class RepLoraVit(nn.Module):
    """
    Introduce RepLoRA layers into a ViT model.
    """
    def __init__(self, vit_model, r: int, alpha: int, lora_layers=None):
        super().__init__()

        assert r > 0, "r must be > 0"
        assert alpha > 0, "alpha must be > 0"

        if lora_layers:
            self.lora_layers = lora_layers
        else:
            self.lora_layers = list(range(len(vit_model.encoder.layer)))

        # embedding dimension
        dim = vit_model.encoder.layer[0].attention.attention.query.in_features

        # freeze backbone
        for param in vit_model.parameters():
            param.requires_grad = False

        # store generators
        self.list_A_gens, self.list_B_gens = [], []

        # replace with RepLoRA layers
        for layer_i, block_i in enumerate(vit_model.encoder.layer):
            if layer_i not in self.lora_layers:
                continue

            w_q_linear = block_i.attention.attention.query
            w_v_linear = block_i.attention.attention.value

            # create shared generators for this block
            A_gen = RepLoRAGenerator(r, dim, kind="A")
            B_gen = RepLoRAGenerator(r, dim, kind="B")
            self.list_A_gens.append(A_gen)
            self.list_B_gens.append(B_gen)

            # wrap query and value
            block_i.attention.attention.query = RepLoraLayer(w_q_linear, A_gen, B_gen, r, alpha, target="q")
            block_i.attention.attention.value = RepLoraLayer(w_v_linear, A_gen, B_gen, r, alpha, target="v")

        self.model_vit_replora = vit_model

    @property
    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.model_vit_replora(x, **kwargs)
    
    def save_replora_params(self, filename: str):
            assert filename.endswith(".safetensors"), "Filename must end with .safetensors"
            state_dict = {}

            # save lora layers
            state_dict["lora_layers"] = torch.tensor(self.lora_layers, dtype=torch.int32)

            # save generator weights
            for i, (A_gen, B_gen) in enumerate(zip(self.list_A_gens, self.list_B_gens)):
                for name, param in A_gen.state_dict().items():
                    state_dict[f"A_gen_{i}_{name}"] = param
                for name, param in B_gen.state_dict().items():
                    state_dict[f"B_gen_{i}_{name}"] = param

            save_file(state_dict, filename)
            print(f"Saved RepLoRA params to {filename}")

    def load_replora_params(self, filename: str):
        assert filename.endswith(".safetensors"), "Filename must end with .safetensors"
        loaded = load_file(filename)

        # load layer indices
        loaded_layers = loaded["lora_layers"].tolist()
        if loaded_layers != self.lora_layers:
            print("Warning: loaded lora_layers differ, adjusting...")
            self.lora_layers = loaded_layers

        # load generator weights
        for i, (A_gen, B_gen) in enumerate(zip(self.list_A_gens, self.list_B_gens)):
            A_state, B_state = {}, {}
            for key, tensor in loaded.items():
                if key.startswith(f"A_gen_{i}_"):
                    A_state[key[len(f"A_gen_{i}_"):]] = tensor
                elif key.startswith(f"B_gen_{i}_"):
                    B_state[key[len(f"B_gen_{i}_"):]] = tensor
            A_gen.load_state_dict(A_state)
            B_gen.load_state_dict(B_state)

            print(f"Loaded RepLoRA params from {filename}")