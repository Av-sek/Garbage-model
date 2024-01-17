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

transformations = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor()]
)

dataset = ImageFolder(data_dir, transform=transformations)

print(dataset.classes)