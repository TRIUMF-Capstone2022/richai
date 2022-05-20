import numpy as np
import torch
from utils.helpers import get_logger, get_config
import pandas as pd

from models.pointnet import PointNetc
from dataset.rich_dataset import combine_datset

logger = get_logger()

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, data_loader):
    """Evaluate the trained model on the test set."""
    labels, predictions = [], []
    accuracy = 0

    # enable multi GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=get_config("gpu"))
        device = f"cuda:{model.device_ids[0]}"
    model.to(device)

    model.load_state_dict(torch.load(get_config("model.pointnet.saved_model")))

    model.eval()
    with torch.no_grad():  # this stops pytorch doing computational graph stuff under-the-hood and saves memory and time
        for X, y, p in data_loader:
            X, y = X.to(device).float(), y.long().to(device)
            # outputs, __, __ = model(X.transpose(1, 2), p) # With momentum
            outputs, __, __ = model(X.transpose(1, 2))  # Without momentum
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            y, predicted = (
                y.cpu().detach().numpy(),
                predicted.squeeze().cpu().detach().numpy(),
            )
            predictions.extend(predicted)
            labels.extend(y)

    val_acc = 100.0 * correct / total
    logger.info("Validation accuracy: %d %%" % val_acc)

    # convert labels and prediction in dataframe
    df = pd.DataFrame({"labels": labels, "predictions": predictions})

    # pion efficiency
    pion = df[df.labels == 1]
    pion_efficiency = sum(pion.predictions == 1) / len(pion)

    # muon efficiency
    muon = df[df.labels == 0]
    muon_efficiency = sum(muon.predictions == 0) / len(muon)

    # positron efficiency
    positron = df[df.labels == 2]
    positron_efficiency = sum(positron.predictions == 2) / len(positron)

    logger.info(
        f"""Pion efficiency : {pion_efficiency}
    Muon efficiency : {muon_efficiency}
    Positron efficiency : {positron_efficiency}
    """
    )

    return df


def evaluate_pointnet():
    """Evaluate pointnet"""

    # result
    result = pd.DataFrame()

    # define model
    model = PointNetc(k=get_config("model.pointnet.num_classes"))

    # data_loader
    for file_, dataset in combine_datset("test").items():

        logger.info(f"Evaluating for {file_}")

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=get_config("data_loader.batch_size"),
            shuffle=False,
            num_workers=get_config("data_loader.num_workers"),
        )

        result = pd.concat([result, evaluate(model, data_loader)])

    result.to_csv(get_config("model.pointnet.predictions"), index=False)

    logger.info("model evaluation completed")


if __name__ == "__main__":

    evaluate_pointnet()
