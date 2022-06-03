"""
Loss function `cross_entropy_ls ` adapted from:
https://github.com/AnTao97/dgcnn.pytorch/blob/master/util.py

Trainer function `trainer` inspired by:
https://github.com/AnTao97/dgcnn.pytorch/blob/master/main_cls.py
"""


import time
import torch
import pandas as pd
import torch.nn.functional as F
from utils.helpers import get_config, get_logger
from models.dgcnn import DGCNN
from dataset.rich_dataset import RICHDataset
from dataset.data_loader import data_loader

logger = get_logger()


def cross_entropy_ls(y_pred, y_true, smoothing=True):
    """Calculate cross entropy loss, apply label smoothing if needed."""

    y_true = y_true.contiguous().view(-1)

    # TODO PyTorch now has built in support for this.  Consider removing
    # if/else logic in favour of this.  Left as is in the meantime.
    if smoothing:
        eps = 0.2
        n_class = y_pred.size(1)

        one_hot = torch.zeros_like(y_pred).scatter(1, y_true.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(y_pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(y_pred, y_true, reduction="mean")

    return loss


def trainer(
    model,
    optimizer,
    train_loader,
    val_loader=None,
    criterion=cross_entropy_ls,
    epochs=5,
    scheduler=None,
    device="cuda",
    results=False,
    operating_point=0.5,
    show_results=2500,
):
    """Trainer for DGCNN"""

    logger.info(f"Starting training...")
    training_start = time.time()

    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []

    model.to(device)

    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch+1}/{epochs}...")
        epoch_start = time.time()

        running_loss = 0.0
        running_total, running_correct = 0, 0
        total_pions, total_muons = 0, 0
        true_pion_preds, false_neg_muons = 0, 0

        # --------------------------------------- TRAINING ---------------------------------------
        logger.info(f"Starting training phase of epoch {epoch+1}...")
        model.train()

        for i, (X, y, p) in enumerate(train_loader, 0):
            # X needs to be reshaped for knn calculation to work
            # (batch_size, PMTs, points) -> (batch_size, points, PMTs)
            X = X.float().to(device)
            X = X.permute(0, 2, 1)

            # label, momentum, radius are OK with shape (1, batch_size)
            y = y.long().to(device)
            p = p.float().to(device)

            # zero parameter gradients
            optimizer.zero_grad()

            # forward pass and calculate loss
            logits = model(X, p).flatten()
            loss = criterion(logits, y.type(torch.float32))
            running_loss += loss.item()

            # calculate predictions and accuracy
            preds = torch.sigmoid(logits) > operating_point
            running_total += X.size(0)
            running_correct += preds.eq(y).sum().item()
            acc = running_correct / running_total

            # total pions and muons this batch
            total_pions += torch.sum(y == 1).item()
            total_muons += torch.sum(y == 0).item()

            # true positive pions and false negative muons
            true_pion_preds += torch.sum((preds == 1) & (y == 1)).item()
            false_neg_muons += torch.sum((preds == 0) & (y == 1)).item()

            # pion efficiency and muon misclassification as pion
            pion_efficiency = true_pion_preds / total_pions
            muon_misclass = false_neg_muons / total_muons

            # back prop and update weights
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            # log results every 100 batches or upon final batch
            if (i + 1) % show_results == 0 or i == len(train_loader) - 1:
                outstr = (
                    "(T)E: %d (B: %4d/%4d), Loss: %.4f, Acc: %.4f, Pion eff: %.4f, Muon misclass: %.4f"
                    % (
                        epoch + 1,
                        i + 1,
                        len(train_loader),
                        running_loss / running_total,
                        acc,
                        pion_efficiency,
                        muon_misclass,
                    )
                )

                logger.info(outstr)

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(acc)

        logger.info(f"Completed training phase of epoch {epoch+1}!")

        # --------------------------------------- VALIDATION -------------------------------------
        if val_loader:
            logger.info(f"Starting validation phase of epoch {epoch+1}...")
            model.eval()

            running_loss = 0.0
            running_total, running_correct = 0, 0
            total_pions, total_muons = 0, 0
            true_pion_preds, false_neg_muons = 0, 0

            with torch.no_grad():
                for i, (X, y, p) in enumerate(val_loader, 0):
                    X = X.float().to(device)
                    X = X.permute(0, 2, 1)
                    y = y.long().to(device)
                    p = p.float().to(device)

                    logits = model(X, p).flatten()
                    loss = criterion(logits, y.type(torch.float32))
                    running_loss += loss.item()

                    # preds = logits.max(dim=1)[1]
                    preds = torch.sigmoid(logits) > operating_point
                    running_total += X.size(0)
                    running_correct += preds.eq(y).sum().item()
                    acc = running_correct / running_total

                    # total pions and muons this batch
                    total_pions += torch.sum(y == 1).item()
                    total_muons += torch.sum(y == 0).item()

                    # true positive pions and false negative muons
                    true_pion_preds += torch.sum((preds == 1) & (y == 1)).item()
                    false_neg_muons += torch.sum((preds == 0) & (y == 1)).item()

                    # pion efficiency and muon misclassification as pion
                    pion_efficiency = true_pion_preds / total_pions
                    muon_misclass = false_neg_muons / total_muons

                    # log results every 100 batches or upon final batch
                    if (i + 1) % show_results == 0 or i == len(val_loader) - 1:
                        outstr = (
                            "(V)E: %d (B: %4d/%4d), Loss: %.4f, Acc: %.4f, Pion eff: %.4f, Muon misclass: %.4f"
                            % (
                                epoch + 1,
                                i + 1,
                                len(val_loader),
                                running_loss / running_total,
                                acc,
                                pion_efficiency,
                                muon_misclass,
                            )
                        )

            valid_losses.append(running_loss / len(val_loader))
            valid_accs.append(acc)

            logger.info(outstr)

        logger.info(f"Completed validation phase of epoch {epoch+1}!")

        # log epoch elapsed time
        epoch_time_elapsed = time.time() - epoch_start

        outstr = ("Epoch %d/%d completed in %.0fm %.0fs") % (
            epoch + 1,
            epochs,
            epoch_time_elapsed // 60,
            epoch_time_elapsed % 60,
        )

        logger.info(outstr)

    # log total elapsed training time
    total_time_elapsed = time.time() - training_start
    outstr = "Training completed in {:.0f}m {:.0f}s".format(
        total_time_elapsed // 60, total_time_elapsed % 60
    )

    logger.info(outstr)

    if results:
        data = {
            "train_loss": train_losses,
            "train_acc": train_accs,
            "val_loss": valid_losses,
            "val_acc": valid_accs,
        }
        results = pd.DataFrame(data)

        results.to_csv("dgcnn_training_results.csv")


def train_combined():

    k = 8
    gpus = [7]
    delta = 0.3
    model_path = f"saved_models/dgcnn_k{k}_delta{delta}.pt"

    # model parameters
    # k = get_config("model.dgcnn.k")
    output_channels = get_config("model.dgcnn.output_channels")
    # model_path = get_config("model.dgcnn.saved_model")
    epochs = get_config("model.dgcnn.epochs")
    # gpus = get_config("gpu")

    # dataset parameters
    dset_path = get_config("dataset.dgcnn.dataset")
    sample_file = get_config("dataset.dgcnn.sample_file")
    val_split = get_config("dataset.dgcnn.val")
    test_split = get_config("dataset.dgcnn.test")
    seed = get_config("seed")
    # delta = get_config("model.dgcnn.k")

    # log training information
    logger.info(
        f"""
    TRAINING SESSION INFORMATION
    Main dataset: {dset_path}
    Sample file: {sample_file}
    Model save path: {model_path}
    Total epochs: {epochs}
    Train/val/test: {1-val_split-test_split:.2f}/{val_split}/{test_split}
    Seed: {seed}
    Time delta: {delta}
    KNN k: {k}
    Final output layer nodes: {output_channels}
    GPU: {gpus}
    """
    )

    model = DGCNN(k=k, output_channels=output_channels)

    # enable multi GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=gpus)
        device = f"cuda:{model.device_ids[0]}"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # get the dataset, for now this is C since it's been updated
    dataset = RICHDataset(
        dset_path=dset_path,
        sample_file=sample_file,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        delta=delta,
    )

    # get the data loaders
    train_loader, val_loader, _ = data_loader(dataset)

    # train the model
    trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        epochs=epochs,
        device=device,
        show_results=2500,
    )

    torch.save(model.state_dict(), model_path)
    logger.info(f"Model successfully saved to {model_path}")


if __name__ == "__main__":
    train_combined()
