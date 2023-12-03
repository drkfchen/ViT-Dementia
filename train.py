import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import argparse

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from functions import data_setup, engine, helper_functions, model_builder, utils

from pathlib import Path

# Create device agnostic code
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

data_path = Path("data/")

# Setup parser
parser = argparse.ArgumentParser(description="Get some hyperparameter for training")

parser.add_argument(
    "--epochs", type=int, default=90, help="Number of epochs to train the model"
)

parser.add_argument(
    "--batch_size", type=int, default=32, help="Number of sample per batch"
)

parser.add_argument(
    "--lr", type=float, default=1e-3, help="Learning rate for the optimizer"
)

parser.add_argument(
    "--train_dir",
    type=str,
    default=data_path / "train",
    help="Path to the training directory",
)

parser.add_argument(
    "--test_dir",
    type=str,
    default=data_path / "test",
    help="Path to the training directory",
)

parser.add_argument(
    "--patch_size",
    type=int,
    default=16,
    help="Size of the patch for the transformer encoder",
)

parser.add_argument(
    "--num_transformer_layer",
    type=int,
    default=12,
    help="Number of transformer layer for the transformer encoder",
)

parser.add_argument(
    "--embedding_dim",
    type=int,
    default=768,
    help="Size of the embedding dimension for the transformer encoder",
)

parser.add_argument(
    "--mlp_size",
    type=int,
    default=3072,
    help="Size of the MLP for the transformer encoder",
)

parser.add_argument(
    "--num_heads",
    type=int,
    default=12,
    help="Number of Head for the transformer encoder",
)

parser.add_argument(
    "--num_classes",
    type=int,
    default=2,
    help="Number of classes for the transformer encoder",
)

# Get arguments from the parser
args = parser.parse_args()

# Setup hyperparameter
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.lr
PATCH_SIZE = args.patch_size
NUM_TRANSFORMER_LAYER = args.num_transformer_layer
EMBEDDING_DIM = args.embedding_dim
MLP_SIZE = args.mlp_size
NUM_HEADS = args.num_heads
NUM_CLASSES = args.num_classes
print(f"[INFO] Training the model for {NUM_EPOCHS} with batch size of {BATCH_SIZE}")

# Setup directories
train_dir = args.train_dir
test_dir = args.test_dir
print(f"[INFO] Training data file: {train_dir}")
print(f"[INFO] Testing data file: {test_dir}")


# Transform images into tensor
simple_transform = transforms.Compose(
    [transforms.Resize(size=(224, 224)), transforms.ToTensor()]
)

# Create test and train DataLoader
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    batch_size=BATCH_SIZE,
    train_transform=simple_transform,
    transform=simple_transform,
)


# Instantiate the ViT Model
ViTModel = model_builder.ViT(
    patch_size=PATCH_SIZE,
    num_transformer_layer=NUM_TRANSFORMER_LAYER,
    embedding_dim=EMBEDDING_DIM,
    mlp_size=MLP_SIZE,
    num_heads=NUM_HEADS,
    num_classes=NUM_CLASSES,
).to(device)

# Define the optimizer
optimizer = torch.optim.Adam(
    params=ViTModel.parameters(), lr=LEARNING_RATE, weight_decay=0.03
)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer)

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()

helper_functions.set_seeds()

# Create device agnostic code
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

results = engine.train(
    model=ViTModel,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device,
)

utils.save_model(model=ViTModel, target_dir="models", model_name="ViTModelV1.pth")
