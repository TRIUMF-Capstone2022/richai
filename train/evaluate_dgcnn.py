import torch
import pandas as pd

from models.dgcnn import DGCNN
from dataset.rich_dataset import RICHDataset
from dataset.data_loader import data_loader
from utils.helpers import get_logger, get_config
from torch.utils.data.sampler import SubsetRandomSampler


logger = get_logger()


def get_predictions(model, dataloader, operating_point=0.5, unstandardize=True):
    """Evaluate the trained DGCNN model on the test set."""

    logger.info("Getting predictions with DGCNN...")

    labels, predictions, probabilities, momentum = [], [], [], []

    gpus = get_config("gpu")
    state_dict = get_config("model.dgcnn.saved_model")

    if torch.cuda.device_count() > 1:
        logger.info(f"Using gpus {gpus}")
        model = torch.nn.DataParallel(model, device_ids=gpus)
        device = f"cuda:{model.device_ids[0]}"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using {device}")

    model.to(device)
    model.load_state_dict(torch.load(state_dict))
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

    if unstandardize:
        mean_momentum = get_config("dataset.standardize.mean_momentum")
        std_momentum = get_config("dataset.standardize.std_momentum")
        df["momentum"] = df["momentum"] * std_momentum + mean_momentum

    return df


def evaluate_dgcnn(test_only=False):
    """Evaluate DGCNN"""

    logger.info("DGCNN evaluation starting...")

    result = pd.DataFrame()

    model = DGCNN(
        output_channels=get_config("model.dgcnn.num_classes"),
        k=get_config("model.dgcnn.k"),
        momentum=get_config("model.dgcnn.momentum"),
        radius=get_config("model.dgcnn.radius"),
    )

    files = get_config("dataset.test")

    for file_, sample_file in files.items():

        logger.info(f"Evaluating for {file_}")

        dataset = RICHDataset(
            dset_path=file_,
            # sample_file=sample_file,
            # val_split=get_config("dataset.dgcnn.val"),
            # test_split=get_config("dataset.dgcnn.test"),
            val_split=None,
            test_split=None,
            seed=get_config("seed"),
            delta=get_config("model.dgcnn.delta"),
            test_only=test_only,
        )

        # _, _, test_loader = data_loader(dataset)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=get_config("data_loader.batch_size"),
            shuffle=False,
            sampler=SubsetRandomSampler(dataset.test_indices),
            num_workers=get_config("data_loader.num_workers"),
            drop_last=get_config("data_loader.drop_last"),
        )

        df = get_predictions(model, data_loader)
        result = pd.concat([result, df])

    results_path = get_config("model.dgcnn.predictions")
    result.to_csv(results_path, index=False)

    logger.info("DGCNN evaluation completed!")
    logger.info(f"Results successfully saved to {results_path}")


if __name__ == "__main__":
    evaluate_dgcnn(test_only=True)
