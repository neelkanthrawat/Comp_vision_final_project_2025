import os
import random
import matplotlib.pyplot as plt
from PIL import Image

def plot_random_images_and_trimaps(image_folder, trimap_folder, num_samples=10):
    """
    Plots random images alongside their corresponding trimaps in a 5x2 grid.

    Args:
        image_folder (str): Path to the folder containing pet images.
        trimap_folder (str): Path to the folder containing trimap masks.
        num_samples (int, optional): Number of random samples to plot. Default is 10.
    """
    image_files = os.listdir(image_folder)
    random_files = random.sample(image_files, num_samples)

    rows = num_samples // 2 + num_samples % 2  # Calculate needed rows for 2 images per row
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))  # 2 samples per row, each sample uses 2 subplots

    for i, file_name in enumerate(random_files):
        img_path = os.path.join(image_folder, file_name)
        # Replace extension with .png for trimap
        trimap_file = os.path.splitext(file_name)[0] + ".png"
        trimap_path = os.path.join(trimap_folder, trimap_file)

        img = Image.open(img_path).convert("RGB")
        trimap = Image.open(trimap_path)

        row = i // 2
        col_img = (i % 2) * 2  # 0 or 2 for image
        col_trimap = col_img + 1  # 1 or 3 for trimap

        axes[row, col_img].imshow(img)
        axes[row, col_img].set_title("Image")
        axes[row, col_img].axis('off')

        axes[row, col_trimap].imshow(trimap, cmap='jet')
        axes[row, col_trimap].set_title("Trimap")
        axes[row, col_trimap].axis('off')

    plt.tight_layout()
    plt.show()

## I am not quite sure at the moment which one is better, the one below or the one above.

def plot_random_images_and_trimaps_2(dataset_root='data_oxford_iiit', num_samples=10):
    """
    Plots random images alongside their corresponding trimaps in a 5x2 grid.

    Args:
        dataset_root (str): Root folder containing 'resized_images' and 'resized_masks' subfolders.
        num_samples (int, optional): Number of random samples to plot. Default is 10.
    """
    # Resolve base directory 
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    data_dir = os.path.join(base_dir, dataset_root)
    image_folder = os.path.join(data_dir, 'resized_images')
    trimap_folder = os.path.join(data_dir, 'resized_masks')

    image_files = os.listdir(image_folder)
    random_files = random.sample(image_files, num_samples)

    rows = num_samples // 2 + num_samples % 2
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))

    for i, file_name in enumerate(random_files):
        img_path = os.path.join(image_folder, file_name)
        trimap_file = os.path.splitext(file_name)[0] + ".png"
        trimap_path = os.path.join(trimap_folder, trimap_file)

        img = Image.open(img_path).convert("RGB")
        trimap = Image.open(trimap_path)

        row = i // 2
        col_img = (i % 2) * 2
        col_trimap = col_img + 1

        axes[row, col_img].imshow(img)
        axes[row, col_img].set_title("Image")
        axes[row, col_img].axis('off')

        axes[row, col_trimap].imshow(trimap, cmap='jet')
        axes[row, col_trimap].set_title("Trimap")
        axes[row, col_trimap].axis('off')

    plt.tight_layout()
    plt.show()

