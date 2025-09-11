# basic torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim

# hyperparameter optimization rtd
import optuna
import wandb

# os related
import os

# file handling

# segmentation model
from transformers import ViTModel, ViTImageProcessor # modules for loading the vit model
from lora_vit import LoraVit
from segmentation_model import SegViT
from serial_lora_vit import SerialLoraVit
from replora_vit import RepLoraVit
from localised_lora_vit import LocalizedLoraVit
from segmentation_head import CustomSegHead

# dataset class
from pet_dataset_class import PreprocessedPetDataset

# dataloaders
from create_dataloaders import get_pet_dataloaders

# trainer
from trainer import trainer

# loss and metrics
from scripts.trainer.loss_and_metrics_seg import * # idk what to import here tbh. Need to look into it

# data plotting
from data_plotting import plot_random_images_and_trimaps_2

#
from typing import Literal



# for parsing


# global parameters
## load the pre-trained ViT-model (86 Mil)
model_name = 'google/vit-base-patch16-224'
NUM_BLOCKS = 12


def objective():
    pass

def main():
