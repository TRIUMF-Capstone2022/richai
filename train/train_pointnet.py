import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models.pointnet import PointNetFc

from dataset.rich_dataset import combine_datset
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
    class_weights=None,
    scheduler=None,
    epochs=5,
    device='cpu',
):
    """Simple training wrapper for PyTorch network."""
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (X, y, p) in enumerate(train_loader, 0):
            X, y = X.to(device).float(), y.long().to(device)
            optimizer.zero_grad()
            # outputs, m3x3, m64x64 = model(X.transpose(1, 2), p) # with momentum
            outputs, m3x3, m64x64 = model(
                X.transpose(1, 2)
            )  # Without momentum
            if class_weights is not None: 
                loss = pointnetloss(
                    outputs,
                    y,
                    m3x3,
                    m64x64,
                    weight=class_weights,
                    device=device,
                )
            else:
                loss = pointnetloss(outputs, y, m3x3, m64x64, device=device)

            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:
                logger.info(
                    '[Epoch: %d, Batch: %4d / %4d], loss: %.3f'
                    % (epoch + 1, i + 1, len(train_loader), running_loss / 10)
                )
                running_loss = 0.0

        model.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for X, y, p in val_loader:
                    X, y = X.to(device).float(), y.long().to(device)
                    # outputs, __, __ = model(X.transpose(1, 2), p) # With momentum
                    outputs, __, __ = model(
                        X.transpose(1, 2)
                    )  # Without momentum
                    _, predicted = torch.max(outputs.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            val_acc = 100.0 * correct / total
            logger.info('Validation accuracy: %d %%' % val_acc)

    logger.info('Training completed')


def train_combined(reload_model=True, class_weights=True):
    """Train the model on combined dataset"""

    # device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = PointNetFc()

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
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for file_, dataset in combine_datset('train', val_split=0.2).items():

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
            class_weights=class_weights,
            scheduler=None,
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
