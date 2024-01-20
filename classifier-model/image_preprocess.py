import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import os

data_dir = "../garbage-large"

classes = os.listdir(data_dir)

from torchvision import transforms

transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.8, 1.2)), 
    transforms.GaussianBlur(kernel_size=3),  
    transforms.ToTensor()
])


dataset = ImageFolder(data_dir, transform=transformations)

def count_images_in_subfolders(root_folder):
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    counts = []

    for subfolder in subfolders:
        image_count = count_images_in_folder(subfolder)
        counts.append((os.path.basename(subfolder), image_count))
        print(f"Subfolder: {os.path.basename(subfolder)}, Number of Images: {image_count}")

    total_images = sum(count for _, count in counts)
    print(f"Total number of images: {total_images}")

    # Plotting
    plot_counts(counts)

def count_images_in_folder(folder):
    image_count = sum(1 for entry in os.scandir(folder) if entry.is_file())
    return image_count

def plot_counts(counts):
    subfolders, image_counts = zip(*counts)
    plt.bar(subfolders, image_counts)
    plt.xlabel('Subfolders')
    plt.ylabel('Number of Images')
    plt.title('Image Count in Subfolders')
    plt.xticks(rotation=45, ha='right')
    plt.savefig("dataset.png")

if __name__ == "__main__":
    root_folder = "../garbage-large"  # Replace with the path to your folder
    count_images_in_subfolders(root_folder)