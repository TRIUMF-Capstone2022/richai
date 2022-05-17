from cProfile import label
import logging
import mmap
from webbrowser import get
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from collections import defaultdict

from utils.helpers import compute_seq_id, get_config, get_logger

logger = get_logger()


class RICHDataset(Dataset):
    """RICH pytorch dataset."""

    def __init__(self, dset_path, val_split=None, test_split=None, seed=None):

        # set seed
        if seed:
            np.random.seed(seed)

        # read input dataset
        with h5py.File(dset_path, "r") as dfile:

            for key in dfile.attrs:
                logger.info("%s: %s", key, dfile.attrs[key])

            # Load the hit map into memory
            self.hit_mapping = np.asarray(dfile["HitMapping"][:])
            logger.info("hit map size: %i bytes", self.hit_mapping.nbytes)
            self.N = len(self.hit_mapping) - 1  # !!! The last cell is the sentinel !!!

            # Get the info we need to memory map the hits
            hit_ds = dfile["Hits"]
            hit_offset = hit_ds.id.get_offset()
            hit_dtype = hit_ds.dtype
            hit_shape = hit_ds.shape
            hit_length = np.prod(hit_shape)

            # Get the info we need to memory map the events
            event_ds = dfile["Events"]
            event_offset = event_ds.id.get_offset()
            event_dtype = event_ds.dtype
            event_shape = event_ds.shape
            event_length = np.prod(event_shape)

            # Add labels
            mu_off = dfile.attrs["muon_offset"]
            pi_off = dfile.attrs["pion_offset"]
            pos_off = dfile.attrs["positron_offset"]
            entries = dfile.attrs["entries"]

            if [mu_off, pi_off, pos_off] != sorted([mu_off, pi_off, pos_off]):
                raise Exception("Offsets are not correct")

            self.offsets = {
                "entries": entries,
                "muon": mu_off,
                "pion": pi_off,
                "positron": pos_off,
            }

            logger.info(f"Offsets: {self.offsets}")

            # muon: 0, pion: 1, positron: 2
            self.labels = np.zeros(entries, dtype=np.int32)
            self.labels[mu_off:pi_off] = 0
            self.labels[pi_off:pos_off] = 1
            self.labels[pos_off:] = 2

            logger.info(f"Entries: {entries}")
            logger.info(f"Muons start at index: {mu_off}")
            logger.info(f"Pions start at index: {pi_off}")
            logger.info(f"Positron start at index: {pos_off}")

            # shuffle indices
            indices = np.arange(self.N - 2)
            np.random.shuffle(indices)

            # train, validation, test
            if test_split:
                n_val = int(len(indices) * val_split)
                n_test = int(len(indices) * test_split)
                self.train_indices = indices[: -n_val - n_test]
                self.val_indices = indices[-n_test - n_val : -n_test]
                self.test_indices = indices[-n_test:]
            elif val_split:
                n_val = int(len(indices) * val_split)
                self.train_indices = indices[:-n_val]
                self.val_indices = indices[-n_val:]

        # We don't attempt to catch exception here, crash if we cannot open the file.
        with open(dset_path, "rb") as fh:
            fileno = fh.fileno()
            mapping = mmap.mmap(fileno, 0, access=mmap.ACCESS_READ)
            self.hit_array = np.frombuffer(
                mapping, dtype=hit_dtype, count=hit_length, offset=hit_offset
            ).reshape(hit_shape)
            logger.info("hit array mmap size: %i bytes", self.hit_array.nbytes)
            self.event_array = np.frombuffer(
                mapping, dtype=event_dtype, count=event_length, offset=event_offset
            ).reshape(event_shape)
            logger.info("event array mmap size: %i bytes", self.event_array.nbytes)

    def get_position_data(self):
        return np.load("rich_pmt_positions.npy")

    def get_event_pos(self, idx):

        # get hits
        idx_from = self.hit_mapping[idx]
        idx_to = self.hit_mapping[idx + 1]
        hits = self.hit_array[idx_from:idx_to]

        # load position map data
        position_map = self.get_position_data()

        # create index
        index = compute_seq_id(hits)
        event_pos = position_map[index]

        # add difference between chod_time and hit_time as 3rd dim
        event_pos[:, 2] = (
            self.event_array[idx]["chod_time"]
            - self.hit_array[idx_from:idx_to]["hit_time"]
        )

        data = np.zeros_like(position_map)
        data[: event_pos.shape[0], : event_pos.shape[1]] = event_pos

        return data

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        idx_from = self.hit_mapping[idx]
        idx_to = self.hit_mapping[idx + 1]

        self.data = {
            "event_pos": self.get_event_pos(idx),
            "label": self.labels[idx],
            "hit_time": self.hit_array[idx_from:idx_to]["hit_time"],
            "chod_time": self.event_array[idx]["chod_time"],
            "track_momentum": self.event_array[idx]["track_momentum"],
            "ring_radius": self.event_array[idx]["ring_radius"],
        }

        return (torch.tensor(self.data["event_pos"]), torch.tensor(self.data["label"]))


def combine_datset(key, **kwargs):
    """Combine all the datasets on specified path
    key: train or test
    """
    # get it from config
    dset_dirs = [
        os.path.join(get_config("dataset.base_dir"), i)
        for i in get_config(f"dataset.{key}")
    ]

    logger.info(f"Train directories: {dset_dirs}")

    # file list
    dset_dict = defaultdict()

    for dset_dir in dset_dirs:
        # check if the directory exists
        if not os.path.exists(dset_dir):
            raise Exception(f"Directory {dset_dir} does not exist")

        # get list of files
        for dset_file in os.listdir(dset_dir):
            file_ = os.path.join(dset_dir, dset_file)
            dset_dict[file_] = RICHDataset(file_, **kwargs)

    logger.info(f"Training files: {dset_dict}")

    return dset_dict


if __name__ == "__main__":
    combine_datset("test")
    # path = os.path.join(
    #     get_config("dataset.base_dir"),
    #     "A",
    #     "Run008548.EOSlist.CTRL.p.v2.0.4-01_f.v2.0.4-01.h5",
    # )
    # dset = RICHDataset(dset_path=path, val_split=0.1, test_split=0.1)
    # print(dset[0])
    # print(dset.data)
