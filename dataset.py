import os
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import random

class PreprocessedPetDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, mask_folder, image_list):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        file_name = self.image_list[idx]
        img_path = os.path.join(self.image_folder, file_name)
        mask_path = os.path.join(self.mask_folder, os.path.splitext(file_name)[0] + ".png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        img_tensor = T.ToTensor()(img)
        img_tensor = T.Normalize(mean=[0.5]*3, std=[0.5]*3)(img_tensor)

        mask_tensor = T.PILToTensor()(mask).long().squeeze()  # squeeze to remove extra channel # shape: (H, W)

        # Remap: 1 → 0, 2 → 1, 3 → 2
        remapped = torch.zeros_like(mask_tensor)
        remapped[mask_tensor == 1] = 0 # foreground
        remapped[mask_tensor == 2] = 1 # background
        remapped[mask_tensor == 3] = 2 # Not classified

        return img_tensor, remapped



def get_pet_dataloaders(
    image_folder,
    mask_folder,
    DatasetClass,  # eg. send PreprocessedPetDataset class here
    val_ratio=0.1,    # Portion for validation
    test_ratio=0.1,   # Portion for testing
    batch_size=16,
    num_workers=4,
    seed=42,
    all_data = True,
    num_datapoints = 1000, # when we set all_data to False
    ):
    """
    Creates PyTorch DataLoaders for the Oxford-IIIT Pet dataset with train, validation, and test splits.

    Parameters:
    -----------
    image_folder : str
        Path to the folder containing preprocessed/resized images.
    mask_folder : str
        Path to the folder containing corresponding segmentation masks.
    DatasetClass : type
        Dataset class to use
    val_ratio : float, optional (default=0.1)
        Fraction of the dataset to use for validation.
    test_ratio : float, optional (default=0.1)
        Fraction of the dataset to use for testing.
    batch_size : int, optional (default=16)
        Number of samples per batch to load.
    num_workers : int, optional (default=4)
        Number of subprocesses to use for data loading.
    seed : int, optional (default=42)
        Random seed for reproducible shuffling and splitting.

    Returns:
    --------
    train_loader : DataLoader
        DataLoader for the training set with shuffling enabled.
    val_loader : DataLoader
        DataLoader for the validation set without shuffling.
    test_loader : DataLoader
        DataLoader for the test set without shuffling.
    """

    # List all files in the image folder without filtering extensions
    all_files = os.listdir(image_folder)
    all_files.sort()  # Sort to ensure consistent order before shuffling

    # Shuffle the list of filenames using a fixed seed for reproducibility
    random.seed(seed)
    all_files_shuffled = all_files.copy()
    random.shuffle(all_files_shuffled)

    if all_data:
        total_size = len(all_files_shuffled)
    else:
        total_size = min(num_datapoints, len(all_files_shuffled))
        print(f"[INFO] Using only {total_size} datapoints out of {len(all_files_shuffled)} total files.")

    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size - test_size
    print(f"train_size: {train_size}, val_size: {val_size} and test_size: {test_size}")

    # Split into train, val, and test lists
    train_files = all_files_shuffled[:train_size] # Training files
    val_files = all_files_shuffled[train_size:train_size + val_size]#val files
    test_files = all_files_shuffled[train_size + val_size:total_size]# Test files

    # Create datasets for each split
    if DatasetClass is None:
        raise ValueError("You must provide a DatasetClass to use.")

    train_dataset = DatasetClass(image_folder, mask_folder, train_files)
    val_dataset = DatasetClass(image_folder, mask_folder, val_files)
    test_dataset = DatasetClass(image_folder, mask_folder, test_files)

    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
