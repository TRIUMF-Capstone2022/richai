"""
Loss function `cross_entropy_ls ` adapted from:
https://github.com/AnTao97/dgcnn.pytorch/blob/master/util.py

Trainer function `trainer` inspired by:
https://github.com/AnTao97/dgcnn.pytorch/blob/master/main_cls.py
"""


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sklearn.metrics as metrics
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
    device="cuda",
):
    """Simple training wrapper for PyTorch network."""

    # add logging time
    # add support for multiple GPU training
    # add support for early stopping
    # add support for learning rate scheduler
    # add support for printing relevant metrics

    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []

    model.to(device)
    criterion = cross_entropy_ls

    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch}")

        train_loss, train_acc, count = 0.0, 0.0, 0.0
        train_pred, train_true = [], []

        # --------------------------------------- TRAINING ---------------------------------------
        logger.info(f"Starting training phase of epoch {epoch}")

        model.train()

        for i, (X, y, p) in enumerate(train_loader, 0):
            # X needs to be reshaped for knn calculation to work
            # (batch_size, PMTs, points) -> (batch_size, points, PTs)
            X = X.float().to(device)
            X = X.permute(0, 2, 1)

            # label and momentum are OK with size (1, batch_size)
            y = y.long().to(device)
            p = p.float().to(device)

            # number of examples processed this epoch
            batch_size = X.size()[0]
            count += batch_size

            # zero gradients
            optimizer.zero_grad()

            # forward pass and calc batch training loss
            logits = model(X)
            loss = criterion(logits, y)
            # train_loss += loss.item() * batch_size
            train_loss += loss.item()

            # track batch predictions
            preds = logits.max(dim=1)[1]
            train_acc += (preds == y).type(torch.float32).mean().item()
            train_pred.append(preds.detach().cpu().numpy())
            train_true.append(y.cpu().numpy())

            # back prop and update weights
            loss.backward()
            optimizer.step()

            # print results every 10 batches or upon final batch
            if (i + 1) % 10 == 0 or i == len(train_loader) - 1:
                outstr = "Epoch %d, batch %4d / %4d, train loss: %.6f" % (
                    epoch,
                    i,
                    len(train_loader),
                    train_loss * 1.0 / count,
                )

                logger.info(outstr)

                # TODO remove break
                break

        # epoch training results
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        logger.info(f"Completed training phase of epoch {epoch}")

        # --------------------------------------- VALIDATION -------------------------------------
        logger.info(f"Starting validation phase of epoch {epoch}")

        valid_loss, valid_acc, count = 0.0, 0.0, 0.0
        valid_pred, valid_true = [], []
        model.eval()

        for i, (X, y, p) in enumerate(val_loader, 0):
            # see train loader comments for shapes
            X = X.float().to(device)
            X = X.permute(0, 2, 1)
            y = y.long().to(device)
            p = p.float().to(device)

            # number of examples processed this epoch
            batch_size = X.size()[0]
            count += batch_size

            # calc batch validation loss
            logits = model(X)
            loss = criterion(logits, y)
            valid_loss += loss.item() * batch_size

            # track batch predictions
            preds = logits.max(dim=1)[1]
            valid_pred.append(preds.detach().cpu().numpy())
            valid_true.append(y.cpu().numpy())

            # TODO remove break
            break

        # epoch validation results
        valid_true = np.concatenate(valid_true)
        valid_pred = np.concatenate(valid_pred)

        outstr = "Epoch %d, validation loss: %.6f" % (
            epoch,
            valid_loss * 1.0 / count,
        )

        logger.info(outstr)
        # TODO remove break
        break
