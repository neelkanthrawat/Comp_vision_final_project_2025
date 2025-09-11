import os
import sys
import argparse
import wandb

import triton
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

#
from transformers import ViTModel
from lora_vit import LoraVit
from serial_lora_vit import SerialLoraVit
from replora_vit import RepLoraVit
from localised_lora_vit import LocalizedLoraVit
from segmentation import SegViT

from dataset import PreprocessedPetDataset, get_pet_dataloaders
from trainer import trainer
from loss_and_metrics_seg import *

from typing import Optional

def main():
    parser = argparse.ArgumentParser(description='ViT with pet dataset')
    parser.add_argument('--model_name', default='google/vit-base-patch16-224')
    parser.add_argument('--model_path', default='models/')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--rank', default=4, type=int)
    parser.add_argument('--alpha', default=16, type=int)
    parser.add_argument('--lora_type', default="lora", type=str)
    # Extra args (only used if localized lora)
    parser.add_argument('--r_block', type=int, default=2)
    parser.add_argument('--num_blocks_per_row', type=int, default=2)
    ##
    parser.add_argument('--image_dir', default='pet_dataset/resized_images')
    parser.add_argument('--mask_dir', default='pet_dataset/resized_masks')    
    ## single_training or seperate training
    parser.add_argument('--seperate_learning_rate', default=False, type=bool)
    parser.add_argument("-- lr_vit_backbone", default=1e-4, type=float)
    parser.add_argument("lr_seg_head", default=5e-4, type=float)
    ## whether wanna freeze the lora layers for a few epochs
    parser.add_argument("--want_backbone_frozen_initially", default=False, type=bool)
    parser.add_argument("--freeze_epochs", default=None,type=Optional[int])

    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")

    # Load ViT based backbone model
    vit_pretrained = ViTModel.from_pretrained(args.model_name).to(device)
    if args.lora_type == "lora":
        lora_vit_base = LoraVit(vit_model=vit_pretrained, 
                            r=args.rank, 
                            alpha=args.alpha)
    elif args.lora_type == "serial_lora":
        lora_vit_base = SerialLoraVit(vit_model=vit_pretrained, 
                            r=args.rank)
    elif args.lora_type == "replora":
        lora_vit_base = RepLoraVit(vit_model=vit_pretrained, 
                            r=args.rank, 
                            alpha=args.alpha)
    elif args.lora_type == "localised_lora":
        lora_vit_base = LocalizedLoraVit(vit_model=vit_pretrained,
                                r_block=args.r_block,
                                alpha=args.alpha,
                                num_blocks_per_row=args.num_blocks)
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
                                        all_data=False,
                                        num_datapoints=3000,
                                        batch_size=args.batch_size)
    
    # DEFINING INPUT PARAMETERS FOR THE TRAINER
    ## set up the appropriate optimizer
    if args.seperate_learning_rate:
        optimizer = optim.Adam([
                        {"params": vit_seg_model.backbone_parameters, "lr": args.lr_vit_backbone},
                        {"params": vit_seg_model.head_parameters, "lr": args.lr_seg_head},
                        ])
    else:
        optimizer = optim.Adam(vit_seg_model.parameters(), lr=args.lr)

    criterion = log_cosh_dice_loss
    criterion_kwargs = {"num_classes": 3, "epsilon": 1e-6,}

    ## Scheduler: this needs to be fixed a little bit
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ## dataloader
    dataloaders={"train": train_dataloader, "val": val_dataloader, "test" : test_dataloader}

    # INSTANTIATING THE TRAINER CLASS
    trainer_seg = trainer(model=vit_seg_model, 
                            optimizer=optimizer,
                            criterion=criterion,
                            num_epoch=args.epochs,
                            dataloaders=dataloaders,
                            device=device,
                            criterion_kwargs=criterion_kwargs,
                            want_backbone_frozen_initially=args.want_backbone_frozen_initially,
                            freeze_epochs=args.freeze_epochs
                            )
    
    # Train and evaluation
    trainer_seg.train()

    #test_loss, test_dice, test_iou = trainer_seg.test()
    #print(f"Test Loss {test_loss}, Test Dice {test_dice}, Test IoU {test_iou}")

if __name__ == '__main__':
    main()