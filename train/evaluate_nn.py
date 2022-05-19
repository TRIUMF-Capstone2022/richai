from asyncio.log import logger
from dataset.data_loader import data_loader
import numpy as np
import torch
import torch.nn.functional as F
from utils.helpers import get_logger, get_config
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from models.pointnet import PointNetFeat, PointNetFeedForward
from dataset.rich_dataset import combine_datset

logger = get_logger()

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, data_loader, criterion):
    """Evaluate the trained model on the test set."""
    labels, predictions = [], []
    accuracy = f1score = precision = recall = 0

    # enable multi GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=get_config("gpu"))
        device = f"cuda:{model.device_ids[0]}"
    model.to(device)

    model.load_state_dict(torch.load(get_config("model.pointnet.saved_model")))

    model.eval()
    with torch.no_grad():  # this stops pytorch doing computational graph stuff under-the-hood and saves memory and time
        for X, y, p in data_loader:
            # GPU
            X = X.transpose(2, 1).to(device)
            y = y.long().to(device)
            p = p.to(device)

            y_hat = model(X, p).to(device)
            _, y_hat_labels = torch.softmax(y_hat, dim=1).topk(1, dim=1)
            loss = criterion(y_hat, y)
            loss += loss.item()
            accuracy += (y_hat_labels.squeeze() == y).type(torch.float32).mean().item()
            y, y_hat_labels = (
                y.cpu().detach().numpy(),
                y_hat_labels.squeeze().cpu().detach().numpy(),
            )
            predictions.extend(y_hat_labels)
            labels.extend(y)

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
        f"""Average loss:  {loss/len(data_loader)},
    Average accuracy : {accuracy/len(data_loader)}
    Pion efficiency : {pion_efficiency}
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
    model = PointNetFeedForward(k=get_config("model.pointnet.num_classes"))

    # data_loader
    for file_, dataset in combine_datset("test").items():

        logger.info(f"Evaluating for {file_}")

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=get_config("data_loader.batch_size"),
            shuffle=False,
            num_workers=get_config("data_loader.num_workers"),
        )

        result = pd.concat([result, evaluate(model, data_loader, criterion=F.nll_loss)])

    result.to_csv(get_config("model.pointnet.predictions"), index=False)

    logger.info("model evaluation completed")


if __name__ == "__main__":

    evaluate_pointnet()