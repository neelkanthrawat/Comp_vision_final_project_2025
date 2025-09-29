import os
import torch
from transformers import ViTConfig, ViTModel
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

from model.segmentation_model import SegViT
from data.create_dataloaders import get_pet_dataloaders
from data.pet_dataset_class import PreprocessedPetDataset
from lora_types.lora_vit import LoraVit
from lora_types.serial_lora_vit import SerialLoraVit
from lora_types.localized_lora_vit import LocalizedLoraVit

import matplotlib.pyplot as plt

def load_model(args, device, model_path=None):
    """Load ViT fine-tuned models"""
    vit_pretrained = ViTModel(ViTConfig.from_pretrained("google/vit-base-patch16-224")).to(device)

    if args.lora_type == "lora":
        lora_vit_base = LoraVit(vit_model=vit_pretrained, r=args.rank, alpha=args.alpha)
    elif args.lora_type == "serial_lora":
        lora_vit_base = SerialLoraVit(vit_model=vit_pretrained, r=args.rank)
    elif args.lora_type == "localized_lora":
        lora_vit_base = LocalizedLoraVit(vit_model=vit_pretrained,
                                         r_block=args.r_block,
                                         alpha=args.alpha,
                                         num_blocks_per_row=args.num_blocks_per_row)
    else:
        raise ValueError(f"Unknown args.lora_type: {args.lora_type}")

    vit_seg_model = SegViT(vit_model=lora_vit_base,
                           image_size=224,
                           patch_size=16,
                           dim=768,
                           n_classes=3,
                           device=device)

    if model_path and os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        vit_seg_model.load_state_dict(state, strict=False)
        print(f"Loaded fine-tuned checkpoint from {model_path}")

    return vit_seg_model

def get_dataloaders(args):
    """Get dataloaders for in-domain or out-of-domain data"""
    if args.test_domains == "in_domain":
        _, _, test_dataloader = get_pet_dataloaders(
            image_folder=args.image_dir,
            mask_folder=args.mask_dir,
            DatasetClass=PreprocessedPetDataset,
            seed=args.random_seed,
            num_train=args.num_train,
            batch_size=args.batch_size)
    else:
        # Out-of-domain: use full data ratio for test split
        _, _, test_dataloader = get_pet_dataloaders(
            image_folder=f"{args.style_dir}/{args.style}/images",
            mask_folder=f"{args.style_dir}/{args.style}/masks",
            DatasetClass=PreprocessedPetDataset,
            seed=args.random_seed,
            data_ratio=1.0,
            test_ratio=1.0,
            batch_size=args.batch_size)

    return test_dataloader

def main():
    parser = argparse.ArgumentParser(description='ViT inference and result aggregation')
    parser.add_argument('--model_name', default='google/vit-base-patch16-224')
    parser.add_argument('--model_dir', default='models/')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--image_dir', default='pet_dataset/resized_images')
    parser.add_argument('--mask_dir', default='pet_dataset/resized_masks')
    parser.add_argument('--style_dir', default='stylized_dataset')
    parser.add_argument('--output_dir', default='results/', help='Directory to save results')

    # Model configurations to test
    parser.add_argument('--lora_type', choices=['lora', 'serial_lora', 'localized_lora'], default='localized_lora')
    parser.add_argument("--style", default="candy", choices=["candy", "mosaic", "rain_princess", "udnie"])

    # LoRA parameters
    parser.add_argument('--rank', default=4, type=int)
    parser.add_argument('--alpha', default=16, type=int)
    parser.add_argument('--r_block', default=2, type=int)
    parser.add_argument('--num_blocks_per_row', type=int, default=2)

    # Domain/test/dataset parameters
    parser.add_argument('--test_domains', choices=['in_domain', 'out_domain'], default='in_domain')
    parser.add_argument('--data_percentage', type=float, default=1.0)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--num_train', default=1000, type=int)

    parser.add_argument("--num_samples", default=10, type=int)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    args.data_percentage = 1.0
    args.num_train = 5173
    args.random_seeds = 42
    if args.test_domains == "in_domain":
        style = "origin"
    else:
        style = args.style
    
    args.style = style

    run_name = f"{args.lora_type}_{args.random_seeds}_{int(args.data_percentage * 100)}"
    model_path = os.path.join(args.model_dir, f"{run_name}.pth")
    print(f"Testing seed {args.random_seeds}, model: {model_path}")

    vit_seg_model = load_model(args, device, model_path=model_path)
    test_dataloader = get_dataloaders(args)

    vit_seg_model.eval()
    num_save = 0
    with torch.no_grad():
        for inputs, targets, paths in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = vit_seg_model(inputs)
            preds = torch.argmax(outputs, dim=1)

            for i in range(len(inputs)):
                if num_save >= args.num_samples:
                    break

                prediction = preds[i].detach().cpu().numpy()
                fname = os.path.splitext(os.path.basename(paths[i]))[0]
                save_base = Path(output_dir) / fname

                orig = inputs[i].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
                target = targets[i].detach().cpu().numpy().astype(np.uint8)
                pred = prediction.astype(np.uint8)
                
                plt.imsave(f"{save_base}_input.jpg", orig)    
                plt.imsave(f"{save_base}_target.png", target, cmap="gray")                
                plt.imsave(f"{save_base}_pred.png", pred, cmap="gray")

                num_save += 1

            if num_save >= args.num_samples:
                break

if __name__ == "__main__":
    main()