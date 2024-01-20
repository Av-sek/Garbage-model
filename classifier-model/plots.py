import os
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from image_preprocess import dataset    
save_directory='plots_images/res18'
# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_batch_images(dl):
    for i, (images, labels) in enumerate(dl):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        
        # Save the image
        image_filename = os.path.join(save_directory, f'batch_image_{i}.png')
        plt.savefig(image_filename)
        plt.close()
        break

def save_sample_image(img, label):
    print("Label:", dataset.classes[label], "(Class No: " + str(label) + ")")
    plt.imshow(img.permute(1, 2, 0))

    # Save the image
    image_filename = os.path.join(save_directory, f'sample_image_label_{label}.png')
    plt.savefig(image_filename)
    plt.close()

def save_losses_plot(history):
    train_losses = [x["train_loss"] for x in history]
    val_losses = [x["val_loss"] for x in history]

    epochs = [i for i in range(1, len(history) + 1)]

    plt.plot(epochs, train_losses, "-bx", label="Training")
    plt.plot(epochs, val_losses, "-rx", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs. Epochs")

    # Save the plot
    plot_filename = os.path.join(save_directory, 'losses_plot.png')
    plt.savefig(plot_filename)
    plt.close()

def save_accuracies_plot(history):
    accuracies = [x.get("val_acc", 0) for x in history]

    epochs = [i for i in range(1, len(history) + 1)]

    plt.plot(epochs, accuracies, "-x")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epochs")

    # Save the plot
    plot_filename = os.path.join(save_directory, 'accuracies_plot.png')
    plt.savefig(plot_filename)
    plt.close()
    
def save_losses_batch_plot(history, save_directory="."):
    train_losses = [x.get("train_loss") for x in history]
    val_losses = [x["val_loss"] for x in history]
    batches = range(1, len(history) + 1)

    plt.plot(batches, train_losses, "-bx", label="Training")
    plt.plot(batches, val_losses, "-rx", label="Validation")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs. Batches")

    # Save the plot
    plot_filename = os.path.join(save_directory, 'losses_plot.png')
    plt.savefig(plot_filename)
    plt.close()


