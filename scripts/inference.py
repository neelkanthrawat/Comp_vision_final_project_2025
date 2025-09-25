#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from collections import defaultdict

from transformers import ViTConfig, ViTModel

from model.segmentation_model import SegViT
from data.create_dataloaders import get_pet_dataloaders
from data.pet_dataset_class import PreprocessedPetDataset
from lora_types.lora_vit import LoraVit
from lora_types.serial_lora_vit import SerialLoraVit
from lora_types.localized_lora_vit import LocalizedLoraVit
from trainer.trainer import trainer
from trainer.loss_and_metrics_seg import log_cosh_dice_loss

logger = logging.getLogger()


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
        logger.info(f"Loaded fine-tuned checkpoint from {model_path}")

    return vit_seg_model


def get_dataloaders(args):
    """Get dataloaders for in-domain or out-of-domain data"""
    if args.test_domains == "in_domain":
        _, _, test_dataloader = get_pet_dataloaders(
            image_folder=args.image_dir,
            mask_folder=args.mask_dir,
            DatasetClass=PreprocessedPetDataset,
            seed=args.random_seed,
            data_ratio=args.data_percentage,
            val_ratio=0.2,
            test_ratio=0.1,
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


def run_inference(model_path, args, device):
    """Run inference for a single model"""
    vit_seg_model = load_model(args, device, model_path=model_path)
    test_dataloader = get_dataloaders(args)

    # Set up trainer
    criterion = log_cosh_dice_loss
    criterion_kwarg = {"num_classes": 3, "epsilon": 1e-6}
    dataloaders = {"test": test_dataloader}
    optimizer = torch.optim.Adam(vit_seg_model.parameters(), lr=1e-3)

    trainer_seg = trainer(model=vit_seg_model,
                         optimizer=optimizer,
                         criterion=criterion,
                         num_epoch=1,
                         dataloaders=dataloaders,
                         model_path=model_path,
                         device=device,
                         criterion_kwargs=criterion_kwarg)

    # Run inference
    _, test_dice, test_iou = trainer_seg.val_epoch(split="test")

    return test_dice, test_iou


def aggregate_results_across_seeds(results_dict):
    """Aggregate results across different random seeds"""
    aggregated_results = {}

    for config_key, seed_results in results_dict.items():
        dice_scores = [result['dice'] for result in seed_results.values() if result and result['dice'] is not None]
        iou_scores = [result['iou'] for result in seed_results.values() if result and result['iou'] is not None]

        if dice_scores and iou_scores:
            aggregated_results[config_key] = {
                'dice_mean': float(np.mean(dice_scores)),
                'dice_std': float(np.std(dice_scores)),
                'iou_mean': float(np.mean(iou_scores)),
                'iou_std': float(np.std(iou_scores)),
                'num_seeds': len(dice_scores)
            }
        else:
            aggregated_results[config_key] = {
                'dice_mean': None,
                'dice_std': None,
                'iou_mean': None,
                'iou_std': None,
                'num_seeds': 0
            }

    return aggregated_results


def save_results(results, output_file):
    """Save results to JSON and CSV files"""
    json_file = Path(str(output_file)).with_suffix('.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Convert aggregated results to dataframe when possible
    rows = []
    for config, metrics in results.items():
        if isinstance(metrics, dict) and 'dice_mean' in metrics:
            parts = config.split('_')
            lora_type = parts[0] if len(parts) > 0 else 'unknown'
            data_percentage = parts[1] if len(parts) > 1 else 'unknown'
            test_domains = parts[2] if len(parts) > 2 else 'unknown'

            rows.append({
                'config_key': config,
                'lora_type': lora_type,
                'data_percentage': data_percentage,
                'test_domains': test_domains,
                'dice_mean': metrics['dice_mean'],
                'dice_std': metrics['dice_std'],
                'iou_mean': metrics['iou_mean'],
                'iou_std': metrics['iou_std'],
                'num_seeds': metrics['num_seeds']
            })

    if rows:
        df = pd.DataFrame(rows)
        csv_file = Path(str(output_file)).with_suffix('.csv')
        df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to {json_file} and {csv_file}")
    else:
        logger.info(f"Results saved to {json_file}")


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
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)

    args = parser.parse_args()

    # create output directory and configure logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging properly
    logging.basicConfig(
        filename= output_dir / 'inference.log',
        filemode='a',
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG
    )
    logger = logging.getLogger()

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    all_results = defaultdict(lambda: defaultdict(dict))
    data_percentages = [0.25, 0.5, 0.75, 1.0]
    random_seeds = [10, 20, 30, 42, 123]

    for data_percentage in data_percentages:
        config_key = f"{args.lora_type}_{int(data_percentage*100)}_{args.test_domains}"
        logger.info(f"\nProcessing configuration: {config_key}")

        if args.test_domains == "in_domain":
            style = "origin"
        else:
            style = args.style

        args.data_percentage = data_percentage
        args.style = style
        args.lora_type = args.lora_type

        for seed in random_seeds:
            args.random_seed = seed
            run_name = f"{args.lora_type}_{seed}_{int(data_percentage * 100)}"
            model_path = os.path.join(args.model_dir, f"{run_name}.pth")

            logger.info(f"Testing seed {seed}, model: {model_path}")

            try:
                dice, iou = run_inference(model_path, args, device)

                all_results[config_key][seed] = {
                    'dice': float(dice) if dice is not None else None,
                    'iou': float(iou) if iou is not None else None
                }

                if dice is not None and iou is not None:
                    logger.info(f"Seed {seed} - Dice: {dice:.4f}, IoU: {iou:.4f}")
                else:
                    logger.info(f"Seed {seed} - No valid metrics (None returned)")

            except Exception as e:
                logger.exception(f"Error with seed {seed}: {e}")
                all_results[config_key][seed] = {
                    'dice': None,
                    'iou': None
                }

    # Save individual results
    individual_results_file = output_dir / f'individual_results_{args.style}'
    save_results(dict(all_results), individual_results_file)

    # Aggregate results across seeds
    logger.info("\nAggregating results across seeds...")
    aggregated_results = aggregate_results_across_seeds(all_results)

    # Save aggregated results
    aggregated_results_file = output_dir / f'aggregated_results_{args.style}'
    save_results(aggregated_results, aggregated_results_file)

    # Log aggregated summary
    for config_key, metrics in aggregated_results.items():
        if metrics['dice_mean'] is not None:
            logger.info(f"\n{config_key}:")
            logger.info(f"Dice: {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}")
            logger.info(f"IoU:  {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}")
            logger.info(f"Seeds: {metrics['num_seeds']}")
        else:
            logger.info(f"\n{config_key}: No valid results")


if __name__ == "__main__":
    main()
