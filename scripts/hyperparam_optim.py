import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

# hyperparameter optimization
import optuna
import wandb

# segmentation model
from transformers import ViTModel
from model.segmentation_model import SegViT
from model.segmentation_head import CustomSegHead

# lora types
from lora_types.lora_vit import LoraVit
from lora_types.serial_lora_vit import SerialLoraVit
from lora_types.localized_lora_vit import LocalizedLoraVit

from data.pet_dataset_class import PreprocessedPetDataset
from data.create_dataloaders import get_pet_dataloaders
from trainer.trainer import trainer
from trainer.loss_and_metrics_seg import * 

from typing import Literal
import matplotlib.pyplot as plt


# global parameters
MODEL_NAME = 'google/vit-base-patch16-224'
NUM_BLOCKS = 12
WEIGHT_DEACY=0.0005
BATCH_SIZE = 32
USE_BN=True
DROPOUT_RATE=0.1
NUM_EPOCHS=10
N_TRIALS = 5

def objective(
    trial, 
    device,
    img_dir,
    mask_dir,
    lora_type: Literal["lora", "serial_lora", "localized_lora"] = "lora", 
    optimize_for: Literal["loss", "dice", "iou"] = "loss",
    separate_lr = False,
    want_backbone_frozen_initially=False,
):
    
    # Lora parameters
    lora_rank = trial.suggest_int("lora_rank", 4, 8, 16, 32)
    lora_alpha = trial.suggest_int("lora_alpha", 4, 8, 16, 32, 64)
    
    # Optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "adam"]) 
    optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    
    # W&B config
    wandb_config = {
        "lora_type": lora_type,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "optimizer": optimizer_name,
        "want_backbone_frozen_initially": want_backbone_frozen_initially,
    }   

    # Model initialization
    Vit_pretrained = ViTModel.from_pretrained(MODEL_NAME).to(device)
    if lora_type == "lora":
        lora_vit_base = LoraVit(vit_model=Vit_pretrained, r=lora_rank, alpha=lora_alpha)
    elif lora_type == "serial_lora":
        lora_vit_base = SerialLoraVit(vit_model=Vit_pretrained, r=lora_rank)
    elif lora_type == "replora":
        lora_vit_base = RepLoraVit(vit_model=Vit_pretrained, r=lora_rank, alpha=lora_alpha)
    elif lora_type == "localized_lora":
        r_block = trial.suggest_int("r_block", 2,4,8,16 )
        num_blocks = trial.suggest_int("num_blocks", 2,4,8,16)
        lora_vit_base = LocalizedLoraVit(vit_model=Vit_pretrained,
                                        r_block=r_block,
                                        alpha=lora_alpha,
                                        num_blocks_per_row=num_blocks)
        wandb_config.update({
            "r_block": r_block,
            "num_blocks_per_row": num_blocks
        })
    
    seg_head = CustomSegHead(hidden_dim=768, num_classes=3,
                        patch_size=16,image_size=224,
                        dropout_rate=DROPOUT_RATE, 
                        use_bn=USE_BN)
    
    vit_seg_model = SegViT(vit_model=lora_vit_base,
                        image_size=224, patch_size=16,
                        dim=768, n_classes=3,
                        head=seg_head, device=device)

    # Create dataloaders with sampled batch size 
    train_dl, val_dl, _ = get_pet_dataloaders(
        image_folder=img_dir,
        mask_folder=mask_dir,
        DatasetClass=PreprocessedPetDataset,
        all_data=True,
        val_ratio=0.2,
        test_ratio=0.1,
        BATCH_SIZE=BATCH_SIZE
    )

    # Learning rate and optimizer
    if separate_lr:
        lr_backbone = trial.suggest_float("lr_vit_backbone", 1e-5, 1e-4, 1e-3, log=True)
        lr_head = trial.suggest_float("lr_seg_head", 1e-4, 1e-3, 1e-2, log=True)
        optimizer = optimizer_cls([
            {"params": vit_seg_model.backbone_parameters, "lr": lr_backbone},
            {"params": vit_seg_model.head_parameters, "lr": lr_head},
        ], weight_decay=WEIGHT_DEACY)

        wandb_config.update({
            "lr_vit_backbone": lr_backbone,
            "lr_seg_head": lr_head
        })
    else:
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        optimizer = optimizer_cls(vit_seg_model.parameters(), lr=lr, weight_decay=WEIGHT_DEACY)
        wandb_config.update({"lr": lr})


    if want_backbone_frozen_initially:
        freeze_epochs=trial.suggest_int("freeze_epochs", 2,5,10)
        wandb_config.update({
            "freeze_epochs": freeze_epochs
        })

    ## Initialize W&B
    wandb.init(
        project="Lora_vit_segmentation",
        config=wandb_config,
        reinit='finish_previous'
    )

    # Trainer
    trainer_input_params = {
        "model": vit_seg_model,
        "optimizer": optimizer,
        "criterion": log_cosh_dice_loss,
        "num_epoch": NUM_EPOCHS,
        "dataloaders": {"train": train_dl, "val": val_dl},
        "use_trap_scheduler":True,
        "device": device,
        "criterion_kwargs": {"num_classes": 3, "epsilon": 1e-6},
        "want_backbone_frozen_initially":want_backbone_frozen_initially,
        "freeze_epochs":freeze_epochs # I have fixed it to 3
    }

    trainer_seg_model = trainer(**trainer_input_params)

    # Training loop with W&B logging
    for epoch in range(trainer_seg_model.num_epoch):
        #  train and validation step for each epoch
        avg_train_loss = trainer_seg_model.train_epoch(epoch)
        avg_val_loss, avg_val_dice, avg_val_iou = trainer_seg_model.val_epoch()# unpack all three values returned by val_epoch

        # accumulate losses
        trainer_seg_model.train_error_epoch_list.append(avg_train_loss)
        trainer_seg_model.val_error_epoch_list.append(avg_val_loss)
        # accumulate metrics
        trainer_seg_model.val_dice_epoch_list.append(avg_val_dice)
        trainer_seg_model.val_iou_epoch_list.append(avg_val_iou)

        # log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_dice": avg_val_dice,
            "val_iou": avg_val_iou,
            "lr": optimizer.param_groups[0]["lr"]
        })

    # after we train(), in the class trainer_seg_model, validation loss and validation metrics are stored for each epoch as class attributes
    if optimize_for == "loss":
        result = trainer_seg_model.val_error_epoch_list[-1]
    elif optimize_for == "dice":
        result = trainer_seg_model.val_dice_epoch_list[-1]
    elif optimize_for == "iou":
        result = trainer_seg_model.val_iou_epoch_list[-1]
    else:
        raise ValueError(f"Unknown optimize_for='{optimize_for}'")
    
    # finish W&B run
    wandb.finish() 

    return result

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, optimize_for="dice"), n_trials=N_TRIALS)


if __name__ == "__main__":
    main()