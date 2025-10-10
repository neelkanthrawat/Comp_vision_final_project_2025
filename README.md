# Investigating Novel LoRA Variants For Vision Transformer-Based Segmentation

## About the project
This project focuses on applying Low-Rank Adaptation (LoRA) based techniques to Vision Transformer (ViT) models for image segmentation tasks. The main objective is to overcome the computational challenges associated with full-fledged fine-tuning large-scale ViT models by exploring efficient, parameter-efficient fine-tuning methods.


## Installation
### Prerequisites
Python 3.10,
CUDA 12.3

### Main packages
Pytorch+cuda,
Transformers model,
optuna,
wandb




### Environment setup
1. Clone the repository:

```python
git clone https://github.com/neelkanthrawat/Comp_vision_final_project_2025.git

```

2. Create and activate conda environment:

```python
conda env create -f env.yml
conda activate lora_env
```

## Documentation
### ViT-backbone Architecture

We used the Vision Transformer base-sized ViT-b model, available at HuggingFace (https://huggingface.co/google/vit-base-patch16-224) as the backbone. It's a transformer encoder model pretrained on a massive image dataset at a resolution of 224x224 pixels. It consists of 12 transformer encoder blocks with an embedding dimension of 768. Each transformer block processes the image patch embeddings through multi-head self-attention and a feed-forward layer. It has approximately $86 M$ trainable parameters.

### Segmentation Head

**Segmentation head:**  
Our custom-defined segmentation head architecture is shown below, where  
- `N = H × W` is the number of image tokens/patches,  
- `D` is the embedding dimension (768), and  
- `p` is the patch size (16 in our case).

| Layer | Output Shape | Description |
|--------|----------------|-------------|
| Input | B × N × D | embedding O/P from the ViT encoder |
| Reshape / Permute | B × D × H × W | Appropriate shape for CNN input |
| Conv2d 3×3 | B × D/2 × H × W | — |
| BatchNorm* | B × D/2 × H × W | Optional |
| ReLU | B × D/2 × H × W | — |
| Dropout* | B × D/2 × H × W | Optional |
| Conv2d 1×1 | B × C × H × W | — |
| Upsample (bilinear, scale=p) | B × C × (H·p) × (W·p) | — |
| **Output** | B × C × H_img × W_img | — |

(* indicates optional layers)



### LoRA techniques

## Fine-tuning Strategies Used

For our project, we focused on the widely-used **LoRA** method and its recently proposed variants, which enable efficient adaptation of transformer models by injecting trainable low-rank matrices into the attention layers.

---

### 🧩 LoRA

LoRA assumes that the weight update ΔW can be approximated by the product of two low-rank matrices:

![LoRA Equation](https://latex.codecogs.com/png.latex?\mathbf{W}^{*}=\mathbf{W}+\Delta\mathbf{W}=\mathbf{W}+\mathbf{B}\mathbf{A})


where  
- **W** ∈ ℝ<sup>d × d</sup> is the pre-trained weight matrix (kept frozen during fine-tuning), and  
- **B** ∈ ℝ<sup>d × r</sup>, **A** ∈ ℝ<sup>r × d</sup> are trainable LoRA matrices,  
- **r** is the rank.

In our transformer model, we introduced these LoRA matrices to the **Query (Q)** and **Value (V)** sections of each transformer block.

---

### 🔁 Serial LoRA

This novel variant introduces a shared low-rank matrix that is **serially composed** with the attention mechanism. It learns a pair of low-rank matrices (**A<sub>s</sub>**, **B<sub>s</sub>**) to directly transform the input features.

![serial_lora_architecture](https://github.com/user-attachments/assets/4601d9cd-f45c-466f-b4cf-bb46b0a24bd9)

> *Figure: Serial-LoRA architecture (adapted from Zhong et al., 2025)*

**Key differences from vanilla LoRA:**

1. **Shared Adaptation Matrix:**  
   Unlike LoRA, which applies separate learnable matrices in parallel for each attention component, Serial LoRA learns a **single shared matrix** that acts as a common adjustment — significantly reducing trainable parameters.

2. **Pre-Projection Transformation:**  
   The transformation is applied **before** projection through pre-trained weights, allowing more uniform adaptation across attention components:

![Serial LoRA Equation](https://latex.codecogs.com/png.latex?\tilde{q}=W_{q/k/v}(I+BA)x=W_{q/k/v}x+W_{q/k/v}BAx)


---

### 🌐 Localised LoRA

Vanilla LoRA approximates the weight update as a **globally low-rank** matrix, which, while parameter-efficient, may limit expressiveness.  
To address this, *Barazandeh et al. (2025)* proposed **Localized LoRA**, where weight updates are modeled as **locally low-rank** without drastically increasing the number of trainable parameters.

They partition the weight matrix **W** ∈ ℝ<sup>d × d</sup> into **K × K** equally sized blocks, dividing both rows and columns into K segments.  
Each block *(i, j)* is assigned independent low-rank adapters **A<sub>ij</sub>** ∈ ℝ<sup>r<sub>ij</sub> × d/K</sup> and **B<sub>ij</sub>** ∈ ℝ<sup>d/K × r<sub>ij</sub></sup>.  
For convenience, *r<sub>ij</sub> = r<sub>block</sub>* for all blocks, typically smaller than the global rank in standard LoRA.

The block-wise operator is defined as:

The block-wise operator is defined as:  
![Localized LoRA Block Equation](https://latex.codecogs.com/png.latex?%5Cmathcal%7BB%7D%5Cleft%5B%5C%7BB_%7Bij%7D%2CA_%7Bij%7D%5C%7D_%7Bi%2Cj%3D1%7D%5E%7BK%7D%5Cright%5D%3D%5Cbegin%7Bbmatrix%7D%20B_%7B11%7DA_%7B11%7D%20%26%20%5Cdots%20%26%20B_%7B1K%7DA_%7B1K%7D%20%5C%5C%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%20B_%7BK1%7DA_%7BK1%7D%20%26%20%5Cdots%20%26%20B_%7BKK%7DA_%7BKK%7D%20%5Cend%7Bbmatrix%7D)


and the final weight update becomes:  
![Localized LoRA Update Equation](https://latex.codecogs.com/png.latex?%5Cmathbf%7BW%7D%5E%7B*%7D%20%3D%20%5Cmathbf%7BW%7D%20%2B%20%5Cmathcal%7BB%7D%5Cleft%5B%5C%7BB_%7Bij%7D%2CA_%7Bij%7D%5C%7D_%7Bi%2Cj%3D1%7D%5E%7BK%7D%5Cright%5D)

---

In all three cases above, we followed *Hu et al. (2021)* and initialized:
- **A** with *Kaiming-uniform initialization*, and  
- **B** with zeros.


### In-domain and Out-of-Domain dataset

### Hyperparameter optimization
Automated hyperparameter search using **Optuna** framework:
* Optimization Objective: maximize dice score
* Search Space:
LoRA rank: [4, 8, 16, 32],
LoRA alpha: [4, 8, 16, 32, 64],
Learnig rate: [1e-5, 1e-4,1e-3, 1e-2],
Weight decay: [4, 8, 16, 32, 64]

Additionally, for Localized LoRA, block rank and the number of block are explored in search space [2, 4, 8, 16]

### Ablation study
1. The effect of the LoRA rank on performance
2. Fine-tuning with frozen vit-backbone vs unfrozen vit-backbone

## Examples and Usage
### 1. Hyperparameter Optimzation

```python
python scripts/hyperparam_optim.py \
    --lora_type serial_lora \
    --backbone_frozen  \
    --separate_lr
```

Example code structure:

```python
import optuna
from transformers import ViTModel
from lora_types.lora_vit import LoraVit
from lora_types.serial_lora_vit import SerialLoraVit
from lora_types.localized_lora_vit import LocalizedLoraVit

def objective(
    trial, 
    device,
    img_dir,
    mask_dir,
    lora_type: Literal["lora", "serial_lora", "localized_lora"] = "lora", 
    optimize_for: Literal["loss", "dice", "iou"] = "dice",
    separate_lr = False,
    backbone_frozen=False,
):
    # Lora parameters
    lora_rank = trial.suggest_categorical("lora_rank", [4, 8, 16, 32])
    lora_alpha = trial.suggest_categorical("lora_alpha", [4, 8, 16, 32, 64])
    
    # Optimizer
    optimizer = torch.optim.AdamW
    weight_decay = trial.suggest_categorical("weight_decay", [1e-4, 1e-3, 1e-2]) 

    # Model initialization
    Vit_pretrained = ViTModel.from_pretrained(MODEL_NAME).to(device)
    if lora_type == "lora":
        lora_vit_base = LoraVit(vit_model=Vit_pretrained, r=lora_rank, alpha=lora_alpha)
    elif lora_type == "serial_lora":
        lora_vit_base = SerialLoraVit(vit_model=Vit_pretrained, r=lora_rank)
    elif lora_type == "localized_lora":
        r_block = trial.suggest_categorical("r_block", [2, 4, 8, 16])
        num_blocks = trial.suggest_categorical("num_blocks", [2, 4, 8, 16])
        lora_vit_base = LocalizedLoraVit(vit_model=Vit_pretrained,
        r_block=r_block,
        alpha=lora_alpha,
        num_blocks_per_row=num_blocks) 
    
    result = trainer_seg_model.val_dice_epoch_list[-1]
    return result

study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: objective(trial, 
                                        device=device,
                                        lora_type=lora_type,
                                        separate_lr=True,
                                        backbone_frozen=False,
                                        optimize_for="dice"), n_trials=20)
```

### 2. Model Training

```python
# LoRA
python scripts/train.py \
    --random_seed 42 \
    --data_percentage 0.5 \
    --lora_type lora \
    --rank 16 \
    --alpha 4 \
    --lr 0.01 \
    --weight_decay 0.0001

# Localized LoRA
python scripts/train.py \
    --random_seed 30 \
    --data_percentage 1.0 \
    --lora_type localized_lora \
    --r_block 16 \
    --num_blocks_per_row 8 \
    --alpha 64 \
    --separate_lr \
    --lr_vit_backbone 0.001 \
    --lr_seg_head 0.01 \
    --weight_decay 0.0001

```

Demo notebooks for training the model can be found inside the folder notebooks. It contains the training notebooks for all 3 LoRA techniques (Vanilla LoRA, localised LoRA and serial LoRA).

### 3. Model Inference

```python
python scripts/inference.py \
    --lora_type serial_lora \
    --test_domains in_domain \
    --rank 32 \
    --alpha 32 \
    --output_dir "results_serial_lora_indomain"
```

## Results

Results are logged to WandB and saved locally in `logs/` and `plots/` directory:

* Training loss and validation dice
* OOD performance and visualization

