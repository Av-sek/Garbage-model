import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

data_dir = "../garbage-large"

classes = os.listdir(data_dir)

"""
    compare image size
    Resizing the images to 256x256 and converting them to tensors

    data augumatation(to be done)
    grayscale v/s rgb ()

    number of data in each class and augumentation on least number of data
    dataset ko study garne
    
"""

from torchvision import transforms

transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  
    transforms.RandomRotation(20),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), 
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.8, 1.2)), 
    transforms.GaussianBlur(kernel_size=3),  
    transforms.ToTensor()
])


dataset = ImageFolder(data_dir, transform=transformations)

print(dataset.classes)