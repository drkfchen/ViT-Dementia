"""
Contains functions for training and testing PyTorch model.
"""

from typing import Dict, List, Tuple

import torch
import pickle

from tqdm.auto import tqdm

from typing import Tuple


# Create train step
def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler = None,
    device=torch.device,
) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch

    Turns a target PyTorch Model to training mode and
    then runs through all of the required training steps
    (forward pass, loss calculation, backprop and gradient descent)

    Args:
      model: A PyTorch model to be trained.
      dataloader: A DataLoader instance for the model to be trained on.
      loss_fn: A PyTorch loss function to minimize.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      device: A target device to compute on (e.g. 'cuda' or 'cpu' or 'metal').

    Returns:
      A tuple of training loss and training accuracy metrics.
      In the form (train_loss, train_accuracy). For example:

      (0.1112, 0.8743)
    """
    # Put the model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through the dataloader
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # Forward Pass
        train_pred = model(X)  # output logits

        # Calculate loss
        loss = loss_fn(train_pred, y)
        train_loss += loss.item()

        # Optimizer Zero Grad
        optimizer.zero_grad()

        # Backprop
        loss.backward()

        # Gradient Descent
        optimizer.step()

        # Calculate Accuracy metric
        train_pred_class = torch.argmax(torch.softmax(train_pred, dim=1), dim=1)
        train_acc += (train_pred_class == y).sum().item() / len(train_pred)

    if scheduler:
        scheduler.step()
    else:
        pass

    # Adjust metrics to get average loss and accuracy per batch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


# Create test step
def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device=torch.device,
) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
      model: A PyTorch model to be tested.
      dataloader: A DataLoader instance for the model to be tested on.
      loss_fn: A PyTorch loss function to calculate loss on the test data.
      device: A target device to compute on (e.g. "cuda" or "cpu" or "metal").

    Returns:
      A tuple of testing loss and testing accuracy metrics.
      In the form (test_loss, test_accuracy). For example:

      (0.0223, 0.8985)
    """

    # Put model in eval mode
    model.eval()

    # Setup test loss and acc values
    test_loss, test_acc = 0, 0

    # Turn on inference mode
    with torch.inference_mode():
        # Loop through dataloader
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # Forward Pass
            test_logits = model(X)

            # Calculate Loss
            loss = loss_fn(test_logits, y)
            test_loss += loss.item()

            # Calculate Acc
            test_labels = test_logits.argmax(dim=1)
            test_acc += ((test_labels == y)).sum().item() / len(test_labels)

    # Adjust metrics to get average loss and acc per batch
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc

def save_loss_dictionaries(
    results: Dict[str, List[float]], target_dir: str, model_name: str
) -> None:
    """Saves training and testing loss dictionaries to a target directory.

    Saves the training and testing loss dictionaries as a pickle file
    in the target directory.

    Args:
      results: A dictionary of training and testing loss as well as training and
               testing accuracy metrics. Each metric has a value in a list for
               each epoch.
               In the form: {train_loss: [...],
                             train_acc: [...],
                             test_loss: [...],
                             test_acc: [...]}
      target_dir: A string indicating the target directory to save the JSON file.
      model_name: A string indicating the name of the model to be saved.
    """
    with open(f"{target_dir}/{model_name}_losses.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Loss dictionaries saved in {target_dir}")


from tqdm.auto import tqdm


# 1. Create a train function that takes in various model parameters + optimizer + dataloaders + loss function
def train(
    model: torch.nn.Module,
    model_name: str,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device=torch.device,
    scheduler: torch.optim.lr_scheduler = None,
) -> Dict[str, List[float]]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for
      each epoch.
      In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]}
      For example if training for epochs=2:
                   {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]}
    """
    # Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # 2. Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | Train Loss:{train_loss:.4f} | Train Accuracy: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["train_acc"].append(train_acc)
        results["test_acc"].append(test_acc)
    
    # 6. Save loss dictionaries to a target directory
    save_loss_dictionaries(results=results, target_dir="models", model_name=model_name)

    # 7. Return the filled results at the end of the epoch
    return results
