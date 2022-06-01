import os
import sched
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
import pandas as pd

from models.pointnet import PointNetFc

from dataset.rich_dataset import combine_dataset
from dataset.data_loader import data_loader
from utils.helpers import get_config, get_logger

logger = get_logger()


def pointnetloss(
    outputs, Y, m3x3, m64x64, alpha=0.0001, weight=None, device='cpu'
):
    """Loss function for pointnet
    Ref: https://github.com/nikitakaraevv/pointnet/
    """
    criterion = torch.nn.NLLLoss()
    bs = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
    id3x3 = id3x3.to(device)
    id64x64 = id64x64.to(device)
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
    if weight is not None:
        criterion = nn.NLLLoss(weight=weight)
    return criterion(outputs, Y) + alpha * (
        torch.norm(diff3x3) + torch.norm(diff64x64)
    ) / float(bs)


def trainer(
    model,
    optimizer,
    train_loader,
    val_loader=None,
    scheduler=None,
    epochs=5,
    device='cpu',
):
    """Simple training wrapper for PyTorch network."""
    logger.info(f'Starting training...')
    training_start = time.time()

    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []

    for epoch in range(epochs):
        logger.info(f'Starting epoch {epoch+1}/{epochs}...')
        epoch_start = time.time()

        running_loss = 0.0
        running_total, running_correct = 0, 0
        total_pions, true_pions = 0, 0

        logger.info(f'Starting training phase of epoch {epoch+1}...')
        model.train()
        for i, (X, y, p) in enumerate(train_loader, 0):
            X, y = X.to(device).float(), y.long().to(device)
            p = p.to(device)

            optimizer.zero_grad()
            outputs, m3x3, m64x64 = model(X.transpose(1, 2), p)

            loss = pointnetloss(outputs, y, m3x3, m64x64, device=device)

            # print statistics
            running_loss += loss.item()

            # calculate predictions and accuracy
            _, preds = torch.max(outputs.data, 1)
            running_total += X.size(0)
            running_correct += preds.eq(y).sum().item()
            acc = running_correct / running_total

            # calculate pion efficiency
            pions = torch.where(y == 1, 1, 0)
            total_pions += len(pions)
            true_pions += (preds == pions).sum().item()
            pion_efficiency = true_pions / total_pions

            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            # log results every 100 batches or upon final batch
            if (i + 1) % 250 == 0 or i == len(train_loader) - 1:
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

        # validation
        if val_loader:
            logger.info(f'Starting validation phase of epoch {epoch+1}...')
            model.eval()

            running_loss = 0.0
            running_total, running_correct = 0, 0
            total_pions, true_pions = 0, 0

            with torch.no_grad():
                for i, (X, y, p) in enumerate(val_loader, 0):
                    X, y = X.to(device).float(), y.long().to(device)
                    p = p.to(device)
                    outputs, m3x3, m64x64 = model(X.transpose(1, 2), p)
                    loss = pointnetloss(
                        outputs, y, m3x3, m64x64, device=device
                    )

                    # print statistics
                    running_loss += loss.item()

                    _, preds = torch.max(outputs.data, 1)
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


def train_combined(reload_model=True, class_weights=True):
    """Train the model on combined dataset"""

    # device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = PointNetFc(k=get_config('model.pointnet.num_classes'))

    # enable multi GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=get_config('gpu'))
        device = f'cuda:{model.device_ids[0]}'

    model.to(device)

    # model path
    model_path = get_config('model.pointnet.saved_model')

    logger.info(
        f"""
    Device: {device},
    model_path: {model_path}
    """
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5
    )

    for file_, dataset in combine_dataset('train', val_split=0.2).items():

        logger.info(f'Training for {file_}')

        # get class weights for training
        if class_weights is not None:
            class_weights = torch.Tensor(dataset.get_class_weights()).to(
                device
            )
            logger.info(f'Class weights: {class_weights}')

        trainloader, validloader, _ = data_loader(dataset)

        if reload_model and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            logger.info(
                f'model reloaded from existing path {model_path}, continue training'
            )

        trainer(
            model,
            optimizer,
            train_loader=trainloader,
            val_loader=validloader,
            scheduler=scheduler,
            epochs=get_config('model.pointnet.epochs'),
            device=device,
        )

        logger.info(f'Training completed with file: {file_}')
        logger.info(f'Saving trained model to {model_path}')

        # save model
        torch.save(model.state_dict(), model_path)
        logger.info(f'model successfully saved to {model_path}')


if __name__ == '__main__':
    train_combined(class_weights=get_config('model.pointnet.class_weights'))
