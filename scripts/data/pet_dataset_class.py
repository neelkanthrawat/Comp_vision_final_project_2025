import os
from PIL import Image
import torch
import torchvision.transforms as T

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
        # note: this remap is segmentation mask with remapped label values
        remapped = torch.zeros_like(mask_tensor)
        remapped[mask_tensor == 1] = 0 # foreground
        remapped[mask_tensor == 2] = 1 # background
        remapped[mask_tensor == 3] = 2 # Not classified

        return img_tensor, remapped
