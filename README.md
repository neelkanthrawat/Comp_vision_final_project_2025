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
conda activate env_name
```

## Documentation
### ViT-backbone Architecture

We used the Vision Transformer base-sized ViT-b model, available at huggingface (https://huggingface.co/google/vit-base-patch16-224) as the backbone. It's a transformer encoder model pretrained on a massive image dataset at a resolution of 224x224 pixels. It consists of 12 transformer encoder blocks with an embedding dimension of 768. Each transformer block processes the image patch embeddings through multi-head self-attention and a feed-forward layer. It has approximately $86 M$ trainable parameters.

### Segmentation head architecture


### LoRA techniques

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

