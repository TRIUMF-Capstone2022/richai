"""
Loss function `cross_entropy_ls ` adapted from:
https://github.com/AnTao97/dgcnn.pytorch/blob/master/util.py

Trainer function `trainer` inspired by:
https://github.com/AnTao97/dgcnn.pytorch/blob/master/main_cls.py
"""


import time

from sklearn.feature_selection import VarianceThreshold
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
from utils.helpers import get_config, get_logger

logger = get_logger()
device = torch.device("cuda" if torch.cuda else "cpu")


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
    epochs=5,
    scheduler=None,
    device="cuda",
    results=False,
):
    """Trainer for DGCNN"""

    # TODO add support for early stopping

    logger.info(f"Starting training...")
    training_start = time.time()

    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []

    model.to(device)
    criterion = cross_entropy_ls

    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch+1}/{epochs}...")
        epoch_start = time.time()

        running_loss = 0.0
        running_total, running_correct = 0, 0
        total_pions, true_pions = 0, 0

        # --------------------------------------- TRAINING ---------------------------------------
        logger.info(f"Starting training phase of epoch {epoch+1}...")
        model.train()

        for i, (X, y, p) in enumerate(train_loader, 0):
            # X needs to be reshaped for knn calculation to work
            # (batch_size, PMTs, points) -> (batch_size, points, PMTs)
            X = X.float().to(device)
            X = X.permute(0, 2, 1)

            # label and momentum are OK with shape (1, batch_size)
            y = y.long().to(device)
            p = p.float().to(device)

            # zero parameter gradients
            optimizer.zero_grad()

            # forward pass and calculate loss
            logits = model(X)
            loss = criterion(logits, y)
            running_loss += loss.item()

            # calculate predictions and accuracy
            preds = logits.max(dim=1)[1]
            running_total += X.size(0)
            running_correct += preds.eq(y).sum().item()
            acc = running_correct / running_total

            # calculate pion efficiency
            pions = torch.where(y == 1, 1, 0)
            total_pions += len(pions)
            true_pions += (preds == pions).sum().item()
            pion_efficiency = true_pions / total_pions

            # back prop and update weights
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            # log results every 100 batches or upon final batch
            if (i + 1) % 250 == 0 or i == len(train_loader) - 1:
                outstr = "Epoch %d (batch %4d/%4d), loss: %.4f, accuracy: %.4f, pion eff: %.4f" % (
                    epoch + 1,
                    i + 1,
                    len(train_loader),
                    running_loss / running_total,
                    acc,
                    pion_efficiency,
                )

                logger.info(outstr)

                # TODO remove break
                # break

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(acc)

        logger.info(f"Completed training phase of epoch {epoch+1}!")

        # --------------------------------------- VALIDATION -------------------------------------
        if val_loader:
            logger.info(f"Starting validation phase of epoch {epoch+1}...")
            model.eval()

            running_loss = 0.0
            running_total, running_correct = 0, 0
            total_pions, true_pions = 0, 0

            with torch.no_grad():
                for i, (X, y, p) in enumerate(val_loader, 0):
                    X = X.float().to(device)
                    X = X.permute(0, 2, 1)
                    y = y.long().to(device)
                    p = p.float().to(device)

                    logits = model(X)
                    loss = criterion(logits, y)
                    running_loss += loss.item()

                    preds = logits.max(dim=1)[1]
                    running_total += X.size(0)
                    running_correct += preds.eq(y).sum().item()
                    acc = running_correct / running_total

                    pions = torch.where(y == 1, 1, 0)
                    total_pions += len(pions)
                    true_pions += (preds == pions).sum().item()
                    pion_efficiency = true_pions / total_pions

                    # TODO remove break
                    # break

            valid_losses.append(running_loss / len(val_loader))
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

        # TODO remove break
        # break

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
