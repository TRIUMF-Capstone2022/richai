from asyncio.log import logger
import os
from tokenize import Pointfloat
from models.pointnet import PointNetFeat
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm


from dataset.rich_dataset import RICHDataset
from dataset.data_loader import data_loader
from utils.helpers import get_config

# device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cpu"


def trainer(
    model,
    criterion,
    optimizer,
    scheduler,
    trainloader,
    validloader,
    epochs=5,
    verbose=True,
):
    """Simple training wrapper for PyTorch network."""

    train_loss, valid_loss, valid_accuracy = [], [], []
    for epoch in range(epochs):
        scheduler.step()

        # for each epoch
        train_batch_loss = 0
        valid_batch_loss = 0
        valid_batch_acc = 0

        # Training
        for X, y in trainloader:
            # GPU
            X = X.transpose(2, 1).to(device)
            y = y.long().to(device)

            optimizer.zero_grad()  # Zero all the gradients w.r.t. parameters
            y_hat = model(X)  # Forward pass to get output
            loss = criterion(y_hat, y)  # Calculate loss based on output
            loss.backward()  # Calculate gradients w.r.t. parameters
            optimizer.step()  # Update parameters
            train_batch_loss += loss.item()  # Add loss for this batch to running total
        train_loss.append(train_batch_loss / len(trainloader))

        # Validation
        model.eval()
        with torch.no_grad():  # this stops pytorch doing computational graph stuff under-the-hood and saves memory and time
            for X, y in validloader:
                # GPU
                X = X.transpose(2, 1).to(device)
                y = y.long().to(device)

                y_hat = model(X)
                _, y_hat_labels = torch.softmax(y_hat, dim=1).topk(1, dim=1)
                loss = criterion(y_hat, y)
                valid_batch_loss += loss.item()
                valid_batch_acc += (
                    (y_hat_labels.squeeze() == y).type(torch.float32).mean().item()
                )
        valid_loss.append(valid_batch_loss / len(validloader))
        valid_accuracy.append(valid_batch_acc / len(validloader))  # accuracy

        model.train()

        # Print progress
        if verbose:
            logger.info(
                f"Epoch {epoch + 1}:",
                f"Train Loss: {train_loss[-1]:.3f}.",
                f"Valid Loss: {valid_loss[-1]:.3f}.",
                f"Valid Accuracy: {valid_accuracy[-1]:.2f}.",
            )

    results = {
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "valid_accuracy": valid_accuracy,
    }
    return results


if __name__ == "__main__":
    path = os.path.join(
        get_config("dataset.base_dir"),
        "A",
        "Run008548.EOSlist.CTRL.p.v2.0.4-01_f.v2.0.4-01.h5",
    )
    dataset = RICHDataset(dset_path=path, val_split=0.1, test_split=0.1)
    num_classes = get_config("model.pointnet.num_classes")
    batch_size = get_config("data_loader.batch_size")
    model = PointNetFeat(k=num_classes).to(device)
    criterion = F.nll_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    trainloader, validloader, testloader = data_loader(dataset)

    # start training
    trainer(
        model,
        criterion,
        optimizer,
        scheduler,
        trainloader,
        validloader,
        epochs=5,
        verbose=True,
    )

    # Save model
    PATH = "pointnet.pt"
    torch.save(model.state_dict(), PATH)     # save model at PATH

    # Load model
    model = PointNetFeat(k=num_classes)             # create an instance of the model
    model.load_state_dict(torch.load(PATH))  # load model from PATH
