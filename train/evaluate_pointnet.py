import numpy as np
import torch
from utils.helpers import get_logger, get_config
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd

from models.pointnet import PointNetFc
from dataset.rich_dataset import RICHDataset

logger = get_logger()

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, data_loader, operating_point=0.5):
    """Evaluate the trained model on the test set."""
    labels, predictions = [], []

    # enable multi GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=get_config('gpu'))
        device = f'cuda:{model.device_ids[0]}'
    model.to(device)

    model.load_state_dict(torch.load(get_config('model.pointnet.saved_model')))

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i, (X, y, p, r) in enumerate(data_loader, 0):
            X = X.float().to(device)
            y = y.long().to(device)
            p = p.float().to(device)
            r = r.float().to(device)

            logits = model(X, p, r).flatten()

            # preds = logits.max(dim=1)[1]
            predicted = torch.where(torch.sigmoid(logits) > operating_point, 1, 0)

            total += y.size(0)
            correct += (predicted == y).sum().item()

            y, predicted = (
                y.cpu().detach().numpy(),
                predicted.squeeze().cpu().detach().numpy(),
            )
            predictions.extend(predicted)
            labels.extend(y)

    val_acc = 100.0 * correct / total
    logger.info('Validation accuracy: %d %%' % val_acc)

    # convert labels and prediction in dataframe
    df = pd.DataFrame({'labels': labels, 'predictions': predictions})

    # pion efficiency
    pion = df[df.labels == 1]
    pion_efficiency = sum(pion.predictions == 1) / len(pion)

    # muon efficiency
    muon = df[df.labels == 0]
    muon_efficiency = sum(muon.predictions == 0) / len(muon)


    logger.info(
        f"""
    Pion efficiency : {pion_efficiency}
    Muon efficiency : {muon_efficiency}
    """
    )

    return df


def evaluate_pointnet():
    """Evaluate pointnet"""

    # result
    result = pd.DataFrame()

    model = PointNetFc(
        num_classes=get_config('model.pointnet.num_classes'),
        momentum=True,
        radius=True,
    )

    # data_loader
    files = get_config('dataset.test')
    for file_, sample_file in files.items():

        logger.info(f'Evaluating for {file_}')

        # get the dataset
        dataset = RICHDataset(
            dset_path=file_,
            sample_file=sample_file,
            val_split=None,
            test_split=None,
            seed=get_config('seed'),
            delta=get_config('dataset.delta'),
            # data_augmentation=True,
            test_only=True
        )
        
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=get_config('data_loader.batch_size'),
            shuffle=False,
            sampler=SubsetRandomSampler(dataset.test_indices),
            num_workers=get_config('data_loader.num_workers'),
            drop_last=get_config("data_loader.drop_last")
        )

        result = pd.concat([result, evaluate(model, data_loader)])

    result.to_csv(get_config('model.pointnet.predictions'), index=False)

    logger.info('model evaluation completed')


if __name__ == '__main__':

    evaluate_pointnet()
