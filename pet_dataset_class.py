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

        mask_tensor = T.PILToTensor()(mask).long().squeeze()  # squeeze to remove extra channel

        return img_tensor, mask_tensor
