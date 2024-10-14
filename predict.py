# Imports here
# Check torch version and CUDA status if GPU is enabled.
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os, random
import signal
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import DataLoader
from torch import nn, optim
import argparse

import json

data_dir = "flowers"
train_dir = data_dir + "/train"
valid_dir = data_dir + "/valid"
test_dir = data_dir + "/test"


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    return model


def process_image(image):
    """Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """
    # Load image
    input_img = Image.open(image)

    # TODO: Process a PIL image for use in a PyTorch model

    # Make a thumbnail
    input_img.thumbnail(size=(256, 256))

    # Center crop the image to 224x224
    w, h = input_img.size
    new_width = 224
    new_height = 224
    left = (w - new_width) / 2
    top = (h - new_height) / 2
    right = (w + new_width) / 2
    bottom = (h + new_height) / 2
    crop_img = input_img.crop((left, top, right, bottom))

    # Convert color channel
    np_img = np.array(crop_img) / 255

    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_img = (np_img - mean) / std

    # Transpose
    tp_img = normalized_img.transpose((2, 0, 1))

    return tp_img


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, cat_to_name, topk=5):
    """Predict the class (or classes) of an image using a trained deep learning model."""

    # Process image
    process_img = process_image(image=image_path)

    # Convert the processed image
    img_tensor = torch.from_numpy(process_img).float()

    # Unsqueeze
    img_tensor = img_tensor.unsqueeze(0)

    # TODO: Implement the code to predict the class from an image file

    # Use CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)
    model = model.to(device)

    model.eval()

    with torch.no_grad():
        # Forward pass
        output = model(img_tensor)

        # Get probabilities
        ps = torch.softmax(output, dim=1)

        # Get the probabilities
        top_ps, top_classes = ps.topk(topk, dim=1)

    top_ps = top_ps.cpu().numpy().tolist()[0]
    top_classes = top_classes.cpu().numpy().tolist()[0]

    # Map the class labels
    class_to_idx = {v: k for k, v in model.class_to_idx.items()}
    predicted_classes = [class_to_idx[c] for c in top_classes]

    # Map the flower names
    predicted_flowers = [cat_to_name[str(cls)] for cls in predicted_classes]

    return top_ps, predicted_flowers


def exec_predict(image_path, model_path, topk, gpu, category):

    # Load category
    with open(category, "r") as f:
        cat_to_name = json.load(f)

    # Load model
    model = load_checkpoint(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() and gpu == 1 else "cpu")
    model = model.to(device)

    # Take random label
    rand_lbl = random.choice(os.listdir(test_dir))

    # Take random image
    rand_img = random.choice(os.listdir(f"{test_dir}/{rand_lbl}"))

    # Get image path
    img_path = os.path.join(test_dir, rand_lbl, rand_img)

    # Get flower name
    flower_name = cat_to_name[rand_lbl]

    # Process image
    image = process_image(image=img_path)

    # Predict image
    top_ps, top_classes = predict(image_path, model, cat_to_name=cat_to_name, topk=topk)

    print("Predictions:")
    for i in range(topk):
        print(f"#{i} --- {top_classes[i]}: {round(top_ps[i]* 100,1)}%\n")

    # # Display result
    # plt.barh(top_classes, top_ps)

    # plt.show()

    # # Display Image
    # imshow(image).set_title(f"{flower_name}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        dest="image_path",
    )
    parser.add_argument(
        dest="model_path",
    )
    parser.add_argument(
        "--category", dest="category", default="cat_to_name.json", type=str
    )
    parser.add_argument("--gpu", dest="gpu", default=0, type=int, choices=[0, 1])
    parser.add_argument("--topk", dest="topk", default=5, type=int)

    args = parser.parse_args()
    image_path = args.image_path
    model_path = args.model_path
    category = args.category
    gpu = args.gpu
    topk = args.topk

    exec_predict(
        image_path=image_path,
        model_path=model_path,
        gpu=gpu,
        category=category,
        topk=topk,
    )
