import os
from models.pointnet import PointNetFeat, PointNetFeedForward
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm


from dataset.rich_dataset import RICHDataset, combine_dataset
from dataset.data_loader import data_loader
from utils.helpers import get_config, get_logger

logger = get_logger()

def trainer(
    model,
    criterion,
    optimizer,
    scheduler,
    trainloader,
    validloader,
    epochs=5,
    device="cpu"
):
    """Simple training wrapper for PyTorch network."""

    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []
    
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch+1}/{epochs}...")

        running_loss = 0.0
        running_total, running_correct = 0, 0
        total_pions, true_pions = 0, 0

        # Training
        model.train()
        for i, (X, y, p) in enumerate(trainloader, 0):    
            # GPU
            X = X.transpose(2, 1).to(device)
            y = y.long().to(device)
            p = p.to(device)

            optimizer.zero_grad()  # Zero all the gradients w.r.t. parameters
            y_hat = model(X, p).flatten()
            y_hat_labels = torch.sigmoid(y_hat) > 0.5
            loss = criterion(y_hat, y.type(torch.float32))
            loss.backward()  # Calculate gradients w.r.t. parameters
            optimizer.step()  # Update parameters
            scheduler.step()
            running_loss += loss.item()  # Add loss for this batch to running total
            
            # calculate predictions and accuracy
            running_total += X.size(0)
            running_correct += y_hat_labels.eq(y).sum().item()
            acc = running_correct / running_total

            # calculate pion efficiency
            pions = torch.where(y == 1, 1, 0)
            total_pions += len(pions)
            true_pions += (y_hat_labels == pions).sum().item()
            pion_efficiency = true_pions / total_pions
            
            # log results every 100 batches or upon final batch
            if (i + 1) % 250 == 0 or i == len(trainloader) - 1:
                outstr = "Epoch %d (batch %4d/%4d), loss: %.4f, accuracy: %.4f, pion eff: %.4f" % (
                    epoch + 1,
                    i + 1,
                    len(trainloader),
                    running_loss / running_total,
                    acc,
                    pion_efficiency,
                )

                logger.info(outstr)
        
        train_losses.append(running_loss / len(trainloader))
        train_accs.append(acc)

        logger.info(f"Completed training phase of epoch {epoch+1}!")        

        # Validation
        logger.info(f"Starting validation phase of epoch {epoch+1}...")
        model.eval()

        running_loss = 0.0
        running_total, running_correct = 0, 0
        total_pions, true_pions = 0, 0
        
        with torch.no_grad():  # this stops pytorch doing computational graph stuff under-the-hood and saves memory and time
            for i, (X, y, p) in enumerate(validloader, 0):
                # GPU
                X = X.transpose(2, 1).to(device)
                y = y.long().to(device)
                p = p.to(device)

                y_hat = model(X, p).flatten()
                y_hat_labels = torch.sigmoid(y_hat) > 0.5
                loss = criterion(y_hat, y.type(torch.float32))
                valid_batch_loss += loss.item()
                running_loss += loss.item()

                running_total += X.size(0)
                running_correct += y_hat_labels.eq(y).sum().item()
                acc = running_correct / running_total

                pions = torch.where(y == 1, 1, 0)
                total_pions += len(pions)
                true_pions += (y_hat_labels == pions).sum().item()
                pion_efficiency = true_pions / total_pions
                
        valid_losses.append(running_loss / len(validloader))
        valid_accs.append(acc)

        # log validation results
        outstr = (
            "Epoch %d, validation accuracy: %.4f, validation pion eff: %.4f"
            % (
                epoch + 1,
                acc,
                pion_efficiency,
            )
        )

        logger.info(outstr) # accuracy
        
    logger.info(f"Completed validation phase of epoch {epoch+1}!")    


    return


def train_combined(reload_model=True):
    """Train the model on combined dataset"""

    # define model
    model = PointNetFeedForward(k=get_config("model.pointnet.num_classes"))

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # enable multi GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=get_config("gpu"))
        device = f"cuda:{model.device_ids[0]}"

    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # model path
    model_path = get_config("model.pointnet.saved_model")

    logger.info(
        f"""
    Device: {device},
    model_path: {model_path}
    """
    )

    for file_, dataset in combine_dataset("train", val_split=0.2).items():

        logger.info(f"Training for {file_}")

        trainloader, validloader, _ = data_loader(dataset)

        if reload_model and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            logger.info(
                f"model reloaded from existing path {model_path}, continue training"
            )

        # start training
        trainer(
            model,
            criterion,
            optimizer,
            scheduler,
            trainloader,
            validloader,
            epochs=get_config("model.pointnet.epochs"),
            device=device
        )

        logger.info(f"Training completed with file: {file_}")
        logger.info(f"Saving trained model to {model_path}")

        # save model
        torch.save(model.state_dict(), model_path)
        logger.info(f"model successfully saved to {model_path}")


if __name__ == "__main__":
    train_combined()