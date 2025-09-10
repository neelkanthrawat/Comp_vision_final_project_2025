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

from lora_vit import LoraVit
from serial_lora_vit import SerialLoraVit
from replora_vit import RepLoraVit
from segmentation import SegViT
from dataset import PreprocessedPetDataset, get_pet_dataloaders
from trainer import trainer
from loss_and_metrics_seg import *

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
    parser.add_argument('--image_dir', default='pet_dataset/resized_images')
    parser.add_argument('--mask_dir', default='pet_dataset/resized_masks')    
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
        lora_vit_base = 0
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
    # Model initialization
    optimizer = optim.Adam(vit_seg_model.parameters(), lr=args.lr) 
    criterion = log_cosh_dice_loss
    criterion_kwargs = {"num_classes": 3, "epsilon": 1e-6,}

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    dataloaders={"train": train_dataloader, "val": val_dataloader, "test" : test_dataloader}
    trainer_seg = trainer(model=vit_seg_model, 
                          optimizer=optimizer,
                          criterion=criterion,
                          num_epoch=args.epochs,
                          dataloaders=dataloaders,
                          device=device,
                          criterion_kwargs=criterion_kwargs
                          )
    # Train and evaluation
    trainer_seg.train()
    #test_loss, test_dice, test_iou = trainer_seg.test()
    #print(f"Test Loss {test_loss}, Test Dice {test_dice}, Test IoU {test_iou}")

if __name__ == '__main__':
    main()







