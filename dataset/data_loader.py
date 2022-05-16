# train validation and test
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils.helpers import get_config


def data_loader(dset):
    """Create pytorch data loader"""
    train_loader = DataLoader(
        dset,
        batch_size=get_config("data_loader.batch_size"),
        shuffle=False,
        sampler=SubsetRandomSampler(dset.train_indices),
        num_workers=get_config("data_loader.num_workers"),
    )

    val_loader = DataLoader(
        dset,
        batch_size=get_config("data_loader.batch_size"),
        shuffle=False,
        sampler=SubsetRandomSampler(dset.val_indices),
        num_workers=get_config("data_loader.num_workers"),
    )

    test_loader = DataLoader(
        dset,
        batch_size=get_config("data_loader.batch_size"),
        shuffle=False,
        sampler=SubsetRandomSampler(dset.test_indices),
        num_workers=get_config("data_loader.num_workers"),
    )

    return (train_loader, val_loader, test_loader)
