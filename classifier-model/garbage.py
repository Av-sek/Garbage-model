import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from architecture import ResNet
from device import get_default_device, DeviceDataLoader, to_device


from image_preprocess import dataset
from plots import save_batch_images, save_sample_image, save_accuracies_plot, save_losses_plot, save_losses_batch_plot

"""# Loading and Splitting Data:"""
img, label = dataset[12]
save_sample_image(img, label)


random_seed = 42
torch.manual_seed(random_seed)

total_size = len(dataset)
print(total_size)
train_split = int(0.7 * total_size)
val_split = int(0.2 * total_size)
test_split = total_size - train_split - val_split

train_ds, val_ds, test_ds = random_split(dataset, [train_split, val_split, test_split])

batch_size = 16

train_dl = DataLoader(
    train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True
)
val_dl = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)


save_batch_images(train_dl)

model = ResNet()

device = get_default_device()

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device)

"""# Training the Model:

This is the function for fitting the model.
"""


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    losses = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history, train_losses


model = to_device(ResNet(), device)

evaluate(model, val_dl)

"""Let's start training the model:"""

num_epochs = 10
opt_func = torch.optim.Adam
lr = 5e-5

history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)



save_accuracies_plot(history)
save_losses_plot(history)
save_losses_batch_plot(history)
"""# Visualizing Predictions:"""


def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]


"""Let us see the model's predictions on the test dataset:"""

img, label = test_ds[17]
print("Label:", dataset.classes[label], ", Predicted:", predict_image(img, model))

img, label = test_ds[23]
print("Label:", dataset.classes[label], ", Predicted:", predict_image(img, model))

img, label = test_ds[51]
print("Label:", dataset.classes[label], ", Predicted:", predict_image(img, model))

"""# Predicting External Images:

Let's now test with external images.

I'll use `urllib` for downloading external images.
"""
torch.save(model.state_dict(), "garbage.pth")
# backend/prediction/predict.py
import urllib.request

urllib.request.urlretrieve(
    "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.2F0uH6BguQMctAYEJ-s-1gHaHb%26pid%3DApi&f=1",
    "cans.jpg",
)
urllib.request.urlretrieve(
    "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftinytrashcan.com%2Fwp-content%2Fuploads%2F2018%2F08%2Ftiny-trash-can-bulk-wine-bottle.jpg&f=1&nofb=1",
    "wine-trash.jpg",
)
urllib.request.urlretrieve(
    "http://ourauckland.aucklandcouncil.govt.nz/media/7418/38-94320.jpg",
    "paper-trash.jpg",
)
