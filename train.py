# Imports here
# Check torch version and CUDA status if GPU is enabled.
import torch
import argparse
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16, VGG16_Weights, vgg19, VGG19_Weights
from torch.utils.data import DataLoader
from torch import nn, optim
import os


def load_data_image(data_dir):
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")
    # TODO: Define your transforms for the training, validation, and testing sets
    training_transform = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    testing_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    data_transforms = {
        "training": training_transform,
        "validation": validation_transform,
        "testing": testing_transform,
    }

    training_image_folder = ImageFolder(
        train_dir, transform=data_transforms["training"]
    )
    validation_image_folder = ImageFolder(
        valid_dir, transform=data_transforms["validation"]
    )
    testing_image_folder = ImageFolder(test_dir, transform=data_transforms["testing"])

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        "training": training_image_folder,
        "validation": validation_image_folder,
        "testing": testing_image_folder,
    }

    training_data_loader = DataLoader(
        image_datasets["training"], batch_size=64, shuffle=True
    )
    validation_data_loader = DataLoader(image_datasets["validation"], batch_size=64)
    testing_data_loader = DataLoader(image_datasets["testing"], batch_size=64)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        "training": training_data_loader,
        "validation": validation_data_loader,
        "testing": testing_data_loader,
    }
    return {"data": dataloaders, "training_folder": training_image_folder}


def train(data_dir, epochs, hidden_units, model_arch, learning_rate):
    load_data = load_data_image(data_dir=data_dir)
    dataloaders = load_data["data"]
    # TODO: Build and train your network
    # TODO: Build and train your network
    if model_arch == "vgg16":
        model = vgg16(weights=VGG16_Weights.DEFAULT)
    elif model_arch == "vgg19":
        model = vgg19(vgg16(weights=VGG19_Weights.DEFAULT))
    else:
        model = vgg16(weights=VGG16_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(25088, hidden_units, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_units, 256, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, 102, bias=True),
        nn.LogSoftmax(dim=1),
    )

    # Use CUDA
    device = torch.device("cuda")
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    current = 0
    accuracy = 0

    # Loop through each batch
    for epoch in range(epochs):
        current += 1
        t_loss = 0
        print(f"----- Epochs: {current}/{epochs} -----")
        i = 0
        # Training
        for images, labels in dataloaders["training"]:

            i += 1

            print(f"Images: {i}/{len(dataloaders['training'])}\n")

            # Reset validation loss and accuracy
            v_loss = 0

            accuracy = 0

            c_images = images.to(device)

            c_labels = labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward image
            outputs = model.forward(c_images)

            # Compute loss
            loss = criterion(outputs, c_labels)

            # Compute the gradients
            loss.backward()

            # Update weights
            optimizer.step()

            # Calculate total loss
            t_loss += loss.item()

            # Validate
            model.eval()
            with torch.no_grad():

                for v_images, v_labels in dataloaders["validation"]:

                    v_images = v_images.to(device)

                    v_labels = v_labels.to(device)

                    # Forward image
                    v_outputs = model.forward(v_images)

                    # Compute loss
                    loss = criterion(v_outputs, v_labels)

                    # Calculate total loss
                    v_loss += loss.item()

                    ps = torch.exp(v_outputs)

                    top_p, top_class = ps.topk(1, dim=1)

                    # Check for correct prediction
                    equals = top_class == v_labels.view(*top_class.shape)

                    # Calculate accuracy
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # Calculate average training loss
            avg_t_loss = t_loss / len(dataloaders["training"])

            # Calculate average validation loss
            avg_v_loss = v_loss / len(dataloaders["validation"])

            # Calculate average accuracy
            avg_accuracy = round(accuracy / len(dataloaders["validation"]) * 100, 1)

            print(f"Training Loss: {avg_t_loss}\n")
            print(f"Validation Loss: {avg_v_loss}\n")
            print(f"Accuracy: {avg_accuracy}%\n")

    # TODO: Do validation on the test set
    # Validation
    model.eval()
    with torch.no_grad():

        accuracy = 0

        loss = 0

        for images, labels in dataloaders["testing"]:

            images = images.to(device)

            labels = labels.to(device)

            # Forward image
            outputs = model.forward(images)
            # Compute loss
            loss = criterion(outputs, labels)
            # Calculate total loss
            loss += loss.item()

            ps = torch.exp(outputs)

            top_p, top_class = ps.topk(1, dim=1)

            equals = top_class == labels.view(*top_class.shape)

            # Calculate accuracy
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        # Calculate average loss
        avg_loss = loss / len(dataloaders["validation"])

        # Calculate average accuracy
        avg_accuracy = round(accuracy / len(dataloaders["validation"]) * 100, 1)

        print(f"Test Loss: {avg_loss}\n")
        print(f"Test Accuracy: {avg_accuracy}%\n")

    # TODO: Save the checkpoint
    model.class_to_idx = load_data["training_folder"].class_to_idx
    checkpoint = {
        "epochs": epochs,
        "hidden_units": hidden_units,
        "learning_rate": learning_rate,
        "classifier": model.classifier,
        "optim_stat_dict": optimizer.state_dict(),
        "class_to_idx": load_data["training_folder"].class_to_idx,
        "state_dict": model.state_dict(),
    }

    torch.save(checkpoint, "checkpoint.pth")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument(dest="data_dir", type=str)
    parser.add_argument(
        "--model_arch",
        dest="model_arch",
        default="vgg16",
        type=str,
        choices=["vgg16", "vgg19"],
    )
    parser.add_argument("--epochs", dest="epochs", default=3, type=int)
    parser.add_argument("--hidden_units", dest="hidden_units", type=int, default=64)
    parser.add_argument(
        "--learning_rate", dest="learning_rate", default=0.001, type=float
    )

    # Parse
    args = parser.parse_args()
    data_dir = args.data_dir
    model_arch = args.model_arch
    epochs = args.epochs
    hidden_units = args.hidden_units
    learning_rate = args.learning_rate

    train(
        data_dir=data_dir,
        epochs=epochs,
        hidden_units=hidden_units,
        model_arch=model_arch,
        learning_rate=learning_rate,
    )
