import torch, torchvision
import argparse
from torch import nn
from torchvision import models
from pathlib import Path
from functions import data_setup, engine, helper_functions, model_builder, utils

# Create device agnostic code
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Set the weights of the pretrained model
weights = (
    torchvision.models.ViT_B_16_Weights.DEFAULT
)  # "DEFAULT" means the best available weights

# Set the transform pipeline
auto_transform = weights.transforms()

simple_transform = transforms.Compose(
    [transforms.Resize(size=(224, 224)), transforms.ToTensor()]
)

# Instantiate the model
ViTModelTransfer = torchvision.models.vit_b_16(weights=weights).to(device)

# Freeze the baseline model so the params won't change
for param in ViTModelTransfer.parameters():
    param.requires_grad = False

data_path = Path("data")

# Set the parser
parser = argparse.ArgumentParser(description="Get some hyperparameter")

parser.add_argument(
    "--epochs", type=int, default=10, help="Number of epochs to train the model"
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

# Get the parser attributes
args = parser.parse_args()

# Setup hyperparameter
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.lr
print(f"[INFO] Training the model for {NUM_EPOCHS} with batch size of {BATCH_SIZE}")

# Setup directories
train_dir = args.train_dir
test_dir = args.test_dir
print(f"[INFO] Training data file: {train_dir}")
print(f"[INFO] Testing data file: {test_dir}")

(
    train_dataloader_pretrained,
    test_dataloader_pretrained,
    class_names,
) = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    train_transform=auto_transform,
    transform=simple_transform,
    batch_size=BATCH_SIZE,
)

# Adjust the model to the classification problem
helper_functions.set_seeds()
ViTModelTransfer.heads = nn.Linear(in_features=768, out_features=len(class_names))

# Send model to target device before training
ViTModelTransfer.to(device)

# Set optimizer and loss function
optimizer = torch.optim.Adam(
    params=ViTModelTransfer.parameters(), lr=LEARNING_RATE, weight_decay=0.03
)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer)

loss_fn = nn.CrossEntropyLoss()

results_transfer = engine.train(
    model=ViTModelTransfer,
    train_dataloader=train_dataloader_pretrained,
    test_dataloader=test_dataloader_pretrained,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device,
)

# Save the model
utils.save_model(
    model=ViTModelTransfer,
    target_dir="models",
    model_name="ViTModelTransferLearning90E.pth",
)
