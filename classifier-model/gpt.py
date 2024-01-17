import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
from image_preprocess import dataset
from garbage import accuracy


# Assuming you have a dataset class and an instance of it
# Replace YourDatasetClass with your actual dataset class
# Replace dataset with your actual dataset instance
# Here, we also use some basic transformations (e.g., ToTensor)
transform = Compose([ToTensor()])
dataset.transform = transform

# Split the dataset into training and validation sets
# Adjust the split ratio based on your preference
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
batch_size = 64  # Adjust based on your system's memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Instantiate the ResNet model
num_classes = len(dataset.classes)
resnet_model = ResNet(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet_model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10  # Adjust based on your requirements

for epoch in range(num_epochs):
    resnet_model.train()  # Set the model to training mode

    # Training phase
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()  # Zero the gradients
        outputs = resnet_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

    # Validation phase
    resnet_model.eval()  # Set the model to evaluation mode
    val_losses = []
    val_accuracies = []

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            outputs = resnet_model(inputs)
            val_loss = criterion(outputs, labels).item()
            val_acc = accuracy(outputs, labels).item()
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

    # Calculate and print average validation loss and accuracy
    avg_val_loss = sum(val_losses) / len(val_losses)
    avg_val_acc = sum(val_accuracies) / len(val_accuracies)
    
    # Create a dictionary with results for epoch_end method
    result = {"train_loss": loss.item(), "val_loss": avg_val_loss, "val_acc": avg_val_acc}
    
    # Print or log the results for the epoch
    resnet_model.epoch_end(epoch, result)
