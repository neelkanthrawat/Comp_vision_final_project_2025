import os
import shutil
import random
import subprocess
from pathlib import Path
from typing import List, Tuple
import argparse
import logging

logger = logging.getLogger()

class DatasetStylizer:
    def __init__(self, 
                 img_dir: str,
                 mask_dir: str,
                 stylized_output_path: str,
                 fast_neural_style_path: str):
        """
        Initialize the dataset stylizer.
        
        Args:
            img_dir, mask_dir: Path to original dataset with images and masks
            stylized_output_path: Path where stylized dataset will be saved
            fast_neural_style_path: Path to fast_neural_style repository
        """
        self.imgs_dir = Path(img_dir) 
        self.masks_dir = Path(mask_dir) 
        self.stylized_output_path = Path(stylized_output_path)
        self.fast_neural_style_path = Path(fast_neural_style_path) 
        
        # Style models 
        self.style_models = {
            'mosaic': 'saved_models/mosaic.pth',
            'candy': 'saved_models/candy.pth',
            'rain_princess': 'saved_models/rain_princess.pth',
            'udnie': 'saved_models/udnie.pth'
        }
        
        # Create output directories
        self.stylized_output_path.mkdir(parents=True, exist_ok=True)
    
    def get_shuffled_selected_pairs(self, images_per_style: int = 100) -> List[Tuple[Path, Path]]:
        """
        Shuffle images first, select required number, then find corresponding masks.
        """
        images_dir = self.imgs_dir
        masks_dir = self.masks_dir
        
        # Get all image paths and shuffle them
        all_images = list(images_dir.glob("*.jpg"))
        logger.info(f"Found {len(all_images)} images total")
        
        total_needed = len(self.style_models) * images_per_style

        # Shuffle and select required number of images
        random.shuffle(all_images)
        selected_images = all_images[:total_needed]
        
        # Find corresponding masks for selected images only
        selected_pairs = []
        
        for img_path in selected_images:
            mask_path = masks_dir / f"{img_path.stem}.png"
            selected_pairs.append((img_path, mask_path))
        
        logger.info(f"Selected {len(selected_pairs)} valid image-mask pairs for stylization")
        return selected_pairs
    
    def stylize_images(self, selected_pairs: List[Tuple[Path, Path]], 
                      images_per_style: int = 100):
        """
        Apply different styles to the selected images.
        """
        # Create style-specific directories
        for style_name in self.style_models.keys():
            style_dir = self.stylized_output_path / style_name
            (style_dir / "images").mkdir(parents=True, exist_ok=True)
            (style_dir / "masks").mkdir(parents=True, exist_ok=True)
        
        # Process images for each style
        pair_idx = 0
        for style_name, model_path in self.style_models.items():
            logger.info(f"\nProcessing style: {style_name}")
            
            style_dir = self.stylized_output_path / style_name
            model_full_path = self.fast_neural_style_path / model_path
            
            # Process images for this style
            for i in range(images_per_style):
                if pair_idx >= len(selected_pairs):
                    break
                
                img_path, mask_path = selected_pairs[pair_idx]
                
                # Create output paths
                output_img_name = f"{style_name}_{i:04d}_{img_path.name}"
                output_mask_name = f"{style_name}_{i:04d}_{mask_path.name}"
                
                output_img_path = style_dir / "images" / output_img_name
                output_mask_path = style_dir / "masks" / output_mask_name
                
                # Stylize the image
                success = self.stylize_single_image(img_path, output_img_path, model_full_path)
                
                if success:
                    # Copy the corresponding mask
                    shutil.copy2(mask_path, output_mask_path)
                    logger.info(f" Processed {i+1}/{images_per_style}: {img_path.name}")
                else:
                    logger.info(f"Failed to process {img_path.name}")
                
                pair_idx += 1
    
    def stylize_single_image(self, input_path: Path, output_path: Path, model_path: Path) -> bool:
        """
        Stylize a single image using fast neural style.
        """
        cmd = [
            "python", 
            str(self.fast_neural_style_path / "neural_style.py"),
            "eval",
            "--content-image", str(input_path),
            "--model", str(model_path),
            "--output-image", str(output_path),
            "--accel"
        ]
    
    def run_pipeline(self, images_per_style: int = 100, seed: int = 42):
        """
        Run the complete stylization pipeline.
        """  
        logger.info("Starting dataset stylization pipeline...")
        random.seed(seed)
        
        # Shuffle images and select pairs with corresponding masks
        logger.info("Shuffling images and finding corresponding masks...")
        selected_pairs = self.get_shuffled_selected_pairs(images_per_style)
    
        # Stylize images
        logger.info("Stylizing images...")
        self.stylize_images(selected_pairs, images_per_style)
        
        logger.info(f"\nPipeline completed!")
        logger.info(f"Stylized dataset saved to: {self.stylized_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stylize dataset with neural style transfer")
    parser.add_argument("--img_dir", default="pet_dataset/resized_images")
    parser.add_argument("--mask_dir", default="pet_dataset/resized_masks")
    parser.add_argument("--stylized_output", default="stylized_dataset")
    parser.add_argument("--fast_neural_style_path", default="fast_neural_style")
    parser.add_argument("--images_per_style", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    logging.basicConfig(filename='log',
                filemode='a',
                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                datefmt='%H:%M:%S',
                level=logging.DEBUG)
    
    # Initialize and run the stylizer
    stylizer = DatasetStylizer(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        stylized_output_path=args.stylized_output,
        fast_neural_style_path=args.fast_neural_style_path
    )
    
    stylizer.run_pipeline(
        images_per_style=args.images_per_style,
        seed=args.seed
    )


if __name__ == "__main__":
    main()