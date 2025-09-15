import torch.nn as nn
import torch
from torch import Tensor
import math
from safetensors.torch import save_file, load_file
from .lora import LoraLayer

class LoraVit(nn.Module):
    """ 
    This class is to introduce LoRA layer to the model.
    """
    def __init__(self, vit_model, r:int, alpha:int, lora_layers=None):

        """ 
        vit_model: pre-trained vit model
        r: rank
        alpha: scaling strength for lora
        lora_layers: Layers we want to apply lora to
        """
        super().__init__()

        assert r>0, "r (rank of lora matrices) must be >0"
        assert alpha>0 , "alpha >0"

        if lora_layers:
            self.lora_layers = lora_layers
        else: ## apply lora to all
            ## ? here I need to see how will I check the number of transformer blocks
            self.lora_layers = list(range(len(vit_model.encoder.layer)))
        
        # Dimension of the input vector to the transformer
        dim = vit_model.encoder.layer[0].attention.attention.query.in_features 
        #print(f"dim is: {dim}")
        
        # freeze the parameters
        ## ? How can we invoke paramters in the vit_model
        for param in vit_model.parameters():
            param.requires_grad = False
        
        ## for storing the lora parameters
        self.list_q_As, self.list_q_Bs = [], []
        self.list_v_As, self.list_v_Bs = [], []

        # replace the normal q and V with LoRA layers
        for layer_i, block_i in enumerate(vit_model.encoder.layer):
            if layer_i not in self.lora_layers:
                continue # (next iteration)
            w_q_linear = block_i.attention.attention.query
            w_v_linear = block_i.attention.attention.value
            # Q and V layers' weights

            ## do I need to initialise weights here? or should I do it after this loop?
            a_linear_q = nn.Linear(dim, r, bias=False)
            b_linear_q = nn.Linear(r, dim, bias=False)
            a_linear_v = nn.Linear(dim, r, bias=False)
            b_linear_v = nn.Linear(r, dim, bias=False)

            # Append lora params to the list
            self.list_q_As.append(a_linear_q); self.list_q_Bs.append(b_linear_q)
            self.list_v_As.append(a_linear_v); self.list_v_Bs.append(b_linear_v) 

            # replace with LoRA layer
            # block_i.attn.proj_q = LoraLayer(w_q_linear, a_linear_q, b_linear_q, r, alpha)
            block_i.attention.attention.query = LoraLayer(w_q_linear, a_linear_q, b_linear_q, r, alpha)
            # block_i.attn.proj_v = LoraLayer(w_v_linear, a_linear_v, b_linear_v, r, alpha)
            block_i.attention.attention.value =  LoraLayer(w_v_linear, a_linear_v, b_linear_v, r, alpha)

        self.init_lora_layers()# initialise the lora parameters
        self.model_vit_lora = vit_model

    def init_lora_layers(self) -> None:
        """
        Method to initialise the LoRA layers. A would be initalised using normal distribution and B as 0 i believe
        A initialized with small normal values, B to zeros
        """
        for A in self.list_q_As + self.list_v_As:
            nn.init.kaiming_uniform_(A.weight, a=math.sqrt(5))
            # if you want to use normal distn for initialisation: nn.init.normal_(A.weight, std=1e-3)
        for B in self.list_q_Bs + self.list_v_Bs:
            nn.init.zeros_(B.weight)

        
    def save_lora_params(self, filename:str): 
        """ 

        """
        assert filename.endswith(".safetensors"), "File name is required to have .safetensors extensions"

        # Create dict for safetensors, keys = str, values = tensors
        state_dict = {}

        # Save lora_layers as a tensor
        state_dict['lora_layers'] = torch.tensor(self.lora_layers, dtype=torch.int32)

        # Save all LoRA params with keys indicating their index and type
        for i, (a_q, b_q, a_v, b_v) in enumerate(zip(self.list_q_As, self.list_q_Bs,self.list_v_As, self.list_v_Bs)):
            state_dict[f'q_A_{i}'] = a_q.weight.data
            state_dict[f'q_B_{i}'] = b_q.weight.data
            state_dict[f'v_A_{i}'] = a_v.weight.data
            state_dict[f'v_B_{i}'] = b_v.weight.data

        save_file(state_dict, filename)
        print(f"Saved LoRA params and layers to {filename}")

    def load_lora_params(self, filename:str):
        """ 
        
        """

        assert filename.endswith(".safetensors"), "File name is required to have .safetensors extensions"
        loaded = load_file(filename)
        # Load lora_layers first (convert to list)
        loaded_layers = loaded['lora_layers'].tolist()

        # If current self.lora_layers differs, you might want to reset or warn
        if loaded_layers != self.lora_layers:
            print("Warning: loaded lora_layers differ from current model's layers. Adjusting...")
            ## maybe here I need to add assertion error so that there is not any major mistake later on
            self.lora_layers = loaded_layers
            # Optionally: re-initialize LoRA modules for these layers here

        # Now load weights into LoRA modules
        for i, (a_q, b_q, a_v, b_v) in enumerate(zip(self.list_q_As, self.list_q_Bs, self.list_v_As, self.list_v_Bs)):
            a_q.weight.data.copy_(loaded[f'q_A_{i}'])
            b_q.weight.data.copy_(loaded[f'q_B_{i}'])
            a_v.weight.data.copy_(loaded[f'v_A_{i}'])
            b_v.weight.data.copy_(loaded[f'v_B_{i}'])

        print(f"Loaded LoRA params and layers from {filename}")

    @property
    def num_trainable_params(self):
        """Number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



    def forward(self, x:Tensor, **kwargs) -> Tensor:
        """ 
        run the LoRA vit
        """
        return self.model_vit_lora(x, **kwargs)