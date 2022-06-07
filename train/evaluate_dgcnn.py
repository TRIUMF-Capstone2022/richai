import torch
import pandas as pd

from models.dgcnn import DGCNN
from dataset.rich_dataset import RICHDataset
from dataset.data_loader import data_loader
from utils.helpers import get_logger, get_config
from tqdm import tqdm
import numpy as np
from sklearn.metrics import recall_score


logger = get_logger()


def evaluate(model, data_loader):
    """Evaluate the trained DGCNN model on the test set."""

    labels, predictions = [], []

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=get_config("gpu"))
        device = f"cuda:{model.device_ids[0]}"

    model.to(device)

    model.load_state_dict(torch.load(get_config("model.dgcnn.saved_model")))

    model.eval()

    running_total, running_correct = 0, 0

    with torch.no_grad():
        # for i, (X, y, p, radius) in tqdm(
        #     enumerate(data_loader, 0), total=len(data_loader)
        # ):
        for i, (X, y, p) in tqdm(enumerate(data_loader, 0), total=len(data_loader)):
            # for i, (X, y, p) in enumerate(data_loader, 0):
            X = X.float().to(device)
            X = X.permute(0, 2, 1)
            y = y.long().to(device)
            p = p.float().to(device)
            # radius = radius.float().to(device)

            # logits = model(X, p, radius)
            # logits = model(X)
            logits = model(X, p).flatten()
            # preds = logits.max(dim=1)[1]
            preds = torch.where(torch.sigmoid(logits) > 0.5, 1, 0)
            running_total += X.size(0)
            running_correct += preds.eq(y).sum().item()

            y, preds = (
                y.cpu().detach().numpy(),
                preds.squeeze().cpu().detach().numpy(),
            )

            labels.extend(y)
            predictions.extend(preds)

    acc = running_correct / running_total

    logger.info("Test accuracy: %d " % acc)

    # convert labels and prediction in dataframe
    df = pd.DataFrame({"labels": labels, "predictions": predictions})

    # pion efficiency
    pion = df[df.labels == 1]
    pion_efficiency = sum(pion.predictions == 1) / len(pion)

    # muon efficiency
    muon = df[df.labels == 0]
    muon_efficiency = sum(muon.predictions == 0) / len(muon)

    # positron efficiency
    # positron = df[df.labels == 2]
    # positron_efficiency = sum(positron.predictions == 2) / len(positron)

    logger.info(
        f"""
    Pion efficiency : {pion_efficiency}
    Muon efficiency : {muon_efficiency}
    """
    )
    # Positron efficiency : {positron_efficiency}
    return df


def evaluate_dgcnn():
    """Evaluate DGCNN"""

    logger.info("DGCNN evaluation starting...")

    results_path = get_config("model.dgcnn.predictions")

    model = DGCNN(
        k=get_config("model.dgcnn.k"),
        output_channels=get_config("model.dgcnn.output_channels"),
        momentum=False,
        radius=False,
        # momentum=get_config("model.dgcnn.momentum"),
        # radius=get_config("model.dgcnn.radius"),
    )

    dataset = RICHDataset(
        get_config("dataset.dgcnn.dataset"),
        val_split=get_config("dataset.dgcnn.val"),
        test_split=get_config("dataset.dgcnn.test"),
        seed=get_config("seed"),
        sample_file="/fast_scratch_1/capstone_2022/datasetC_combined.h5",
    )

    # get the data loaders
    _, val_loader, test_loader = data_loader(dataset)

    # df = evaluate(model, test_loader)
    df = evaluate(model, val_loader)

    df.to_csv(results_path, index=False)

    logger.info("DGCNN evaluation completed!")
    logger.info(f"Results successfully saved to {results_path}")


def get_predictions(
    dataloader,
    model,
    state_dict=get_config("model.dgcnn.saved_model"),
    operating_point=0.5,
    gpus=get_config("gpu"),
):
    """Evaluate the trained DGCNN model on the test set."""

    logger.info("Getting predictions with DGCNN...")

    labels, predictions, probabilities, momentum = [], [], [], []

    if torch.cuda.device_count() > 1:
        logger.info(f"Using gpus {gpus}")
        model = torch.nn.DataParallel(model, device_ids=gpus)
        device = f"cuda:{model.device_ids[0]}"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using {device}")

    model.load_state_dict(torch.load(state_dict))
    model.to(device)
    model.eval()

    logger.info(f"Model loaded from {state_dict}")

    with torch.no_grad():
        for X, y, p, r in dataloader:
            X = X.float().to(device)
            X = X.permute(0, 2, 1)
            y = y.long().to(device)
            p = p.float().to(device)
            r = r.float().to(device)

            logits = model(X, p, r).flatten()
            probs = torch.sigmoid(logits)
            preds = torch.where(probs > operating_point, 1, 0)

            y, probs, preds, p = (
                y.cpu().detach().numpy(),
                probs.squeeze().cpu().detach().numpy(),
                preds.squeeze().cpu().detach().numpy(),
                p.cpu().detach().numpy(),
            )

            # keep track of all items
            labels.extend(y)
            probabilities.extend(probs)
            predictions.extend(preds)
            momentum.extend(p)

    logger.info(f"Model evaluation complete")

    df = pd.DataFrame(
        {
            "labels": labels,
            "predictions": predictions,
            "probabilities": probabilities,
            "momentum": momentum,
        }
    )

    # unstandardize momentum
    mean_momentum = get_config("dataset.standardize.mean_momentum")
    std_momentum = get_config("dataset.standardize.std_momentum")
    df["momentum"] = df["momentum"] * std_momentum + mean_momentum

    return df


def wrangle_predictions(df, width=1):
    """Wrangle model predictions for plotting.

    Parameters
    ----------
    df : pandas DataFrame
        Initial data from model predictions (output of `get_predictions`)
    width : int, optional
        Width of momentum bins to add to data, by default 1

    Returns
    -------
    wrangled_df
        Wrangled data for plotting.
    """
    df = df.copy()
    bins = []
    bin_labels = []

    # generate bins and bin labels for momentum
    for i in range(0, 40 + width, width):
        bins.append(i)
        bin_labels.append(f"({i}, {i+width}]")

    bins.append(np.inf)
    bin_labels.pop()
    bin_labels.append("40+")

    # add momentum bins to results
    df["momentum_bin"] = pd.cut(x=df["momentum"], bins=bins, labels=bin_labels)

    # pion efficiency by momentum bin (pion recall)
    pion_effciency = df.groupby("momentum_bin").apply(
        lambda x: recall_score(
            x["labels"], x["predictions"], zero_division=0, pos_label=1
        )
    )

    # muon efficiency by momentum bin (1 - muon recall)
    muon_efficiency = 1 - (
        df.groupby("momentum_bin").apply(
            lambda x: recall_score(
                x["labels"], x["predictions"], zero_division=0, pos_label=0
            )
        )
    )

    # combine pion/muon efficiency into one df
    efficiency_df = pd.concat([pion_effciency, muon_efficiency], axis=1)

    efficiency_df.columns = ["pion_efficiency", "muon_efficiency"]

    # counts of actual/predicted muons/pions by momentum bin
    labels_df = df.groupby(["momentum_bin", "labels"]).size().unstack(fill_value=0)
    predictions_df = (
        df.groupby(["momentum_bin", "predictions"]).size().unstack(fill_value=0)
    )

    labels_df.columns = ["actual_muons", "actual_pions"]
    predictions_df.columns = ["predicted_muons", "predicted_pions"]

    # combine efficiency df with counts dfs
    wrangled_df = pd.concat([efficiency_df, labels_df, predictions_df], axis=1)
    wrangled_df = wrangled_df.query("actual_muons + actual_pions != 0")

    return wrangled_df


if __name__ == "__main__":
    # evaluate_dgcnn()

    k = 8
    gpus = [6, 7]
    operating_point = 0.5

    state_dicts = [
        "saved_models/dgcnn_k8_delta040_momentum_4epochs.pt",
        "saved_models/dgcnn_k8_delta040_momentum_radius_4epochs.pt",
        "saved_models/dgcnn_k8_delta050_momentum_4epochs.pt",
        "saved_models/dgcnn_k8_delta050_momentum_radius_4epochs.pt",
        "saved_models/dgcnn_k8_delta075_momentum_radius_4epochs.pt",
    ]

    deltas = [0.40, 0.40, 0.50, 0.50, 0.75]
    momentums = [True, True, True, True, True]
    radii = [False, True, False, True, True]

    paths = [
        "saved_models/validation_results/dgcnn_k8_delta040_momentum_4epochs.csv",
        "saved_models/validation_results/dgcnn_k8_delta040_momentum_radius_4epochs.csv",
        "saved_models/validation_results/dgcnn_k8_delta050_momentum_4epochs.csv",
        "saved_models/validation_results/dgcnn_k8_delta050_momentum_radius_4epochs.csv",
        "saved_models/validation_results/dgcnn_k8_delta075_momentum_radius_4epochs.csv",
    ]

    for state_dict, delta, momentum, radius, path in zip(
        state_dicts, deltas, momentums, radii, paths
    ):
        dataset = RICHDataset(
            dset_path=get_config("dataset.dgcnn.dataset"),
            sample_file=get_config("dataset.dgcnn.sample_file"),
            val_split=get_config("dataset.dgcnn.val"),
            test_split=get_config("dataset.dgcnn.test"),
            seed=get_config("seed"),
            delta=delta,
        )

        _, val_loader, _ = data_loader(dataset)
        model = DGCNN(k=k, output_channels=1, momentum=momentum, radius=radius)
        df = get_predictions(val_loader, model, state_dict, operating_point, gpus)
        df.to_csv(path, index=False)

    # dataset = RICHDataset(
    #     dset_path=get_config("dataset.dgcnn.dataset"),
    #     sample_file=get_config("dataset.dgcnn.sample_file"),
    #     val_split=get_config("dataset.dgcnn.val"),
    #     test_split=get_config("dataset.dgcnn.test"),
    #     seed=get_config("seed"),
    #     delta=deltas[0],
    # )

    # _, val_loader, _ = data_loader(dataset)

    # model = DGCNN(k=k, output_channels=1, momentum=True, radius=True)

    # df = get_predictions(val_loader, model, state_dicts[0], operating_point, gpus)

    # df.to_csv(paths[0], index=False)
