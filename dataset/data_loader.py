"""
PyTorch DataLoaders for the RICH AI project.
"""

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils.helpers import get_config


def data_loader(dset):
    """Returns PyTorch DataLoader for training, validation, and testing.

    Parameters
    ----------
    dset : RICHDataset
        Input RICHDataset from `rich_dataset.py`.

    Returns
    -------
    tuple
        tuple with train, validation and test DataLoaders.
    """
    train_loader = DataLoader(
        dset,
        batch_size=get_config('data_loader.batch_size'),
        shuffle=False,
        sampler=SubsetRandomSampler(dset.train_indices),
        num_workers=get_config('data_loader.num_workers'),
        drop_last=get_config('data_loader.drop_last'),
    )

    val_loader = DataLoader(
        dset,
        batch_size=get_config('data_loader.batch_size'),
        shuffle=False,
        sampler=SubsetRandomSampler(dset.val_indices),
        num_workers=get_config('data_loader.num_workers'),
        drop_last=get_config('data_loader.drop_last'),
    )

    test_loader = None
    if hasattr(dset, 'test_indices'):
        test_loader = DataLoader(
            dset,
            batch_size=get_config('data_loader.batch_size'),
            shuffle=False,
            sampler=SubsetRandomSampler(dset.test_indices),
            num_workers=get_config('data_loader.num_workers'),
            drop_last=get_config('data_loader.drop_last'),
        )

    return (train_loader, val_loader, test_loader)
