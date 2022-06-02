import pickle
import os
import time
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from utils.helpers import get_config, get_logger
from models.pointnet import PointNetFc
from dataset.rich_dataset import RICHDataset
from dataset.data_loader import data_loader

logger = get_logger()
device = torch.device('cuda' if torch.cuda else 'cpu')


def trainer(
    model,
    optimizer,
    train_loader,
    val_loader=None,
    criterion=None,
    epochs=5,
    scheduler=None,
    device='cuda',
    results=False,
):
    """Trainer for PyTorch"""

    logger.info(f'Starting training...')
    training_start = time.time()

    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []

    model.to(device)

    for epoch in range(epochs):
        logger.info(f'Starting epoch {epoch+1}/{epochs}...')
        epoch_start = time.time()

        running_loss = 0.0
        running_total, running_correct = 0, 0
        total_pions, true_pions = 0, 0

        # --------------------------------------- TRAINING ---------------------------------------
        logger.info(f'Starting training phase of epoch {epoch+1}...')
        model.train()

        # for i, (X, y, p, radius) in enumerate(train_loader, 0):
        for i, (X, y, p) in enumerate(train_loader, 0):
            # X needs to be reshaped for knn calculation to work
            # (batch_size, PMTs, points) -> (batch_size, points, PMTs)
            X = X.float().to(device)

            # label, momentum, radius are OK with shape (1, batch_size)
            y = y.long().to(device)
            p = p.float().to(device)
            # radius = radius.float().to(device)

            # zero parameter gradients
            optimizer.zero_grad()

            # forward pass and calculate loss
            # logits = model(X, p, radius)
            # logits = model(X, p)
            logits = model(X, p).flatten()  # binary classification

            # loss = criterion(logits, y)
            loss = criterion(
                logits, y.type(torch.float32)
            )  # binary classification
            running_loss += loss.item()

            # calculate predictions and accuracy
            # preds = logits.max(dim=1)[1]
            preds = torch.sigmoid(logits) > 0.5  # binary classification
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
            if (i + 1) % 2500 == 0 or i == len(train_loader) - 1:
                outstr = (
                    'Epoch %d (batch %4d/%4d), loss: %.4f, accuracy: %.4f, pion eff: %.4f'
                    % (
                        epoch + 1,
                        i + 1,
                        len(train_loader),
                        running_loss / running_total,
                        acc,
                        pion_efficiency,
                    )
                )

                logger.info(outstr)

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(acc)

        logger.info(f'Completed training phase of epoch {epoch+1}!')

        # --------------------------------------- VALIDATION -------------------------------------
        if val_loader:
            logger.info(f'Starting validation phase of epoch {epoch+1}...')
            model.eval()

            running_loss = 0.0
            running_total, running_correct = 0, 0
            total_pions, true_pions = 0, 0

            with torch.no_grad():
                # for i, (X, y, p, radius) in enumerate(val_loader, 0):
                for i, (X, y, p) in enumerate(val_loader, 0):
                    X = X.float().to(device)
                    y = y.long().to(device)
                    p = p.float().to(device)
                    # radius = radius.float().to(device)

                    # logits = model(X, p)
                    # logits = model(X, p, radius)
                    logits = model(X, p).flatten()
                    # loss = criterion(logits, y)
                    loss = criterion(logits, y.type(torch.float32))
                    running_loss += loss.item()

                    # preds = logits.max(dim=1)[1]
                    preds = torch.sigmoid(logits) > 0.5
                    running_total += X.size(0)
                    running_correct += preds.eq(y).sum().item()
                    acc = running_correct / running_total

                    pions = torch.where(y == 1, 1, 0)
                    total_pions += len(pions)
                    true_pions += (preds == pions).sum().item()
                    pion_efficiency = true_pions / total_pions

            valid_losses.append(running_loss / len(val_loader))
            valid_accs.append(acc)

            # log validation results
            outstr = (
                'Epoch %d, validation accuracy: %.4f, validation pion eff: %.4f'
                % (
                    epoch + 1,
                    acc,
                    pion_efficiency,
                )
            )

            logger.info(outstr)

        logger.info(f'Completed validation phase of epoch {epoch+1}!')

        # log epoch elapsed time
        epoch_time_elapsed = time.time() - epoch_start

        outstr = ('Epoch %d/%d completed in %.0fm %.0fs') % (
            epoch + 1,
            epochs,
            epoch_time_elapsed // 60,
            epoch_time_elapsed % 60,
        )

        logger.info(outstr)

    # log total elapsed training time
    total_time_elapsed = time.time() - training_start
    outstr = 'Training completed in {:.0f}m {:.0f}s'.format(
        total_time_elapsed // 60, total_time_elapsed % 60
    )

    logger.info(outstr)

    if results:
        data = {
            'train_loss': train_losses,
            'train_acc': train_accs,
            'val_loss': valid_losses,
            'val_acc': valid_accs,
        }
        results = pd.DataFrame(data)

        results.to_csv('pointnet_training_results.csv')


def train_combined(reload_model=True):

    model = PointNetFc(
        k=get_config('model.pointnet.num_classes'),
    )

    # enable multi GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=get_config('gpu'))
        device = f'cuda:{model.device_ids[0]}'
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    # model path
    model_path = get_config('model.pointnet.saved_model')

    logger.info(
        f"""Device: {device}
        model_path: {model_path}
        """
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # get the dataset, for now this is C since it's been updated
    dataset = RICHDataset(
        get_config('dataset.pointnet.dataset'),
        val_split=get_config('dataset.pointnet.val'),
        test_split=get_config('dataset.pointnet.test'),
        seed=get_config('seed'),
        sample_file='/fast_scratch_1/capstone_2022/datasetC_combined.h5',
    )

    # get the data loaders
    train_loader, val_loader, test_loader = data_loader(dataset)
    
    if reload_model and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    logger.info(
        f"model reloaded from existing path {model_path}, continue training"
    )

    trainer(
        model,
        optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        epochs=get_config('model.pointnet.epochs'),
        device=device,
    )

    logger.info(f'Saving trained model to {model_path}')
    torch.save(model.state_dict(), model_path)
    logger.info(f'Model successfully saved to {model_path}')


if __name__ == '__main__':
    train_combined()
