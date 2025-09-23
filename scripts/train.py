import os
import sys
import argparse
import wandb

import triton
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from transformers import ViTModel

from model.segmentation_model import SegViT
from data.create_dataloaders import *
from data.pet_dataset_class import *
from lora_types.lora_vit import LoraVit
from lora_types.serial_lora_vit import SerialLoraVit
from lora_types.localized_lora_vit import LocalizedLoraVit
from trainer.trainer import trainer
from trainer.loss_and_metrics_seg import *

from typing import Optional

def main():
    parser = argparse.ArgumentParser(description='ViT with pet dataset')
    parser.add_argument('--model_name', default='google/vit-base-patch16-224')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--image_dir', default='pet_dataset/resized_images')
    parser.add_argument('--mask_dir', default='pet_dataset/resized_masks')   
    parser.add_argument('--out_dir', default='models/') 
    # Random seed
    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--data_percentage", default=0.5, type=float)
    # Lora
    parser.add_argument('--lora_type', default="lora", type=str)
    parser.add_argument('--rank', default=4, type=int)
    parser.add_argument('--alpha', default=16, type=int)
    # Localized lora
    parser.add_argument('--r_block', type=int, default=2)
    parser.add_argument('--num_blocks_per_row', type=int, default=2)
    ## Learning rate
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--separate_lr', action="store_true")
    parser.add_argument("--lr_vit_backbone", default=0.001, type=float)
    parser.add_argument("--lr_seg_head", default=0.01, type=float)

    parser.add_argument("--weight_decay", default=0.0001, type=float)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")
    run_name = f"{args.lora_type}_{args.random_seed}_{int(args.data_percentage * 100)}"
    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)
    model_path = f"{output_dir}/{run_name}.pth"

    wandb_config = {
        "lora_type": args.lora_type,
        "lora_alpha": args.alpha,
        "weight_decay": args.weight_decay,
    }  

    # Load ViT based backbone model
    vit_pretrained = ViTModel.from_pretrained(args.model_name).to(device)
    if args.lora_type == "lora":
        lora_vit_base = LoraVit(vit_model=vit_pretrained, 
                            r=args.rank, 
                            alpha=args.alpha)
        wandb_config.update({
            "lora_rank": args.rank,
        })
    elif args.lora_type == "serial_lora":
        lora_vit_base = SerialLoraVit(vit_model=vit_pretrained, 
                            r=args.rank)
        wandb_config.update({
            "lora_rank": args.rank,
        })
    elif args.lora_type == "localized_lora":
        lora_vit_base = LocalizedLoraVit(vit_model=vit_pretrained,
                                r_block=args.r_block,
                                alpha=args.alpha,
                                num_blocks_per_row=args.num_blocks_per_row)
        wandb_config.update({
            "r_block": args.r_block,
            "num_blocks_per_row": args.num_blocks_per_row,
        })
    else:
        raise("Lora type is not implemented")

    # Segmentation Model
    vit_seg_model = SegViT(vit_model=lora_vit_base, 
                            image_size=224, 
                            patch_size=16, 
                            dim=768,
                            n_classes=3,
                            device=device)
    # Dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_pet_dataloaders(
                                        image_folder=args.image_dir,
                                        mask_folder=args.mask_dir,
                                        DatasetClass=PreprocessedPetDataset,
                                        seed=args.random_seed,
                                        data_ratio=args.data_percentage,
                                        val_ratio=0.2,
                                        test_ratio=0.1,
                                        batch_size=args.batch_size)
    
    # DEFINING INPUT PARAMETERS FOR THE TRAINER
    ## set up the appropriate optimizer
    if args.separate_lr:
        optimizer = optim.Adam([
                        {"params": vit_seg_model.backbone_parameters, "lr": args.lr_vit_backbone},
                        {"params": vit_seg_model.head_parameters, "lr": args.lr_seg_head},
                        ], weight_decay=args.weight_decay)
        wandb_config.update({
            "lr_vit_backbone_init": args.lr_vit_backbone,
            "lr_seg_head_init": args.lr_seg_head,
        })
    else:
        optimizer = optim.Adam(vit_seg_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        wandb_config.update({"lr_init": args.lr})

    wandb.init(
        project="Lora_vit_segmentation_training",
        name=run_name, 
        config=wandb_config,
        reinit="finish_previous"
    )
    
    criterion = log_cosh_dice_loss
    criterion_kwarg = {"num_classes": 3, "epsilon": 1e-6,}
    ## dataloader
    dataloaders={"train": train_dataloader, "val": val_dataloader, "test" : test_dataloader}

    # INSTANTIATING THE TRAINER CLASS
    trainer_seg = trainer(model=vit_seg_model, 
                            optimizer=optimizer,
                            criterion=criterion,
                            num_epoch=args.epochs,
                            dataloaders=dataloaders,
                            model_path= model_path,
                            use_trap_scheduler=True,
                            device=device,
                            criterion_kwargs=criterion_kwarg,
                            )
    
    # Train and evaluation
    for epoch in range(trainer_seg.num_epoch):
        avg_train_loss = trainer_seg.train_epoch(epoch)
        avg_val_loss, avg_val_dice, avg_val_iou = trainer_seg.val_epoch(split="val")

        # accumulate loss
        trainer_seg.train_error_epoch_list.append(avg_train_loss)
        trainer_seg.val_error_epoch_list.append(avg_val_loss)
        # accumulate metrics
        trainer_seg.val_dice_epoch_list.append(avg_val_dice)
        trainer_seg.val_iou_epoch_list.append(avg_val_iou)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
                      f"| Val Dice: {avg_val_dice} | Val IoU : {avg_val_iou}")
        # log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_dice": avg_val_dice,
            "val_iou": avg_val_iou,
        })

    trainer_seg.save_model()
    trainer_seg.load_model()
    _, test_dice, test_iou = trainer_seg.val_epoch(split="test")
    print(f"Test Dice {test_dice}, Test IoU {test_iou}")

if __name__ == '__main__':
    main()