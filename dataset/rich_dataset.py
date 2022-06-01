from curses import mousemask
import mmap
import h5py
import torch
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from torch.utils.data import Dataset
from utils.helpers import (
    compute_seq_id,
    get_config,
    get_logger,
    events_to_pandas,
)

logger = get_logger()


class RICHDataset(Dataset):
    """RICH pytorch dataset."""

    def __init__(
        self,
        dset_path,
        sample_file=None,
        val_split=None,
        test_split=None,
        seed=None,
        delta=get_config('dataset.delta'),
        data_augmentation=None,
        test_only=False,
    ):

        self.delta = delta
        self.data_augmentation = data_augmentation

        # set seed
        if seed:
            np.random.seed(seed)

        # read input dataset
        with h5py.File(dset_path, 'r') as dfile:

            for key in dfile.attrs:
                logger.info('%s: %s', key, dfile.attrs[key])

            # Load the hit map into memory
            self.hit_mapping = np.asarray(dfile['HitMapping'][:])
            logger.info('hit map size: %i bytes', self.hit_mapping.nbytes)
            self.N = len(self.hit_mapping) - 1

            # Get the info we need to memory map the hits
            hit_ds = dfile['Hits']
            hit_offset = hit_ds.id.get_offset()
            hit_dtype = hit_ds.dtype
            hit_shape = hit_ds.shape
            hit_length = np.prod(hit_shape)

            # Get the info we need to memory map the events
            event_ds = dfile['Events']
            event_offset = event_ds.id.get_offset()
            event_dtype = event_ds.dtype
            event_shape = event_ds.shape
            event_length = np.prod(event_shape)

            # Add labels
            mu_off = dfile.attrs['muon_offset']
            pi_off = dfile.attrs['pion_offset']
            pos_off = dfile.attrs['positron_offset']
            entries = dfile.attrs['entries']

            if [mu_off, pi_off, pos_off] != sorted([mu_off, pi_off, pos_off]):
                raise Exception('Offsets are not correct')

            self.offsets = {
                'entries': entries,
                'muon': mu_off,
                'pion': pi_off,
                'positron': pos_off,
            }

            logger.info(f'Offsets: {self.offsets}')

            # muon: 0, pion: 1, positron: 2
            self.labels = np.zeros(entries, dtype=np.int32)
            self.labels[mu_off:pi_off] = 0
            self.labels[pi_off:pos_off] = 1
            self.labels[pos_off:] = 2

            logger.info(f'Entries: {entries}')
            logger.info(f'Muons start at index: {mu_off}')
            logger.info(f'Pions start at index: {pi_off}')
            logger.info(f'Positron start at index: {pos_off}')

            # Get indices
            if sample_file and not test_only:
                logger.info(f'Filtering indices for sample: {sample_file}')

                # read in sample file
                df = pd.read_hdf(sample_file)
                logger.info(f'Total samples: {df.shape[0]}')

                # filter ring centre outliers (some very small or large values in data)
                df = df.query(
                    'ring_centre_pos_x < 2500 and ring_centre_pos_y < 2500'
                )
                df = df.query(
                    'ring_centre_pos_x > -2500 and ring_centre_pos_y > -2500'
                )

                # filter very large radii or negative radii
                # df = df.query("ring_radius < 500 and ring_radius > 0")

                logger.info(f'Total samples with no outliers: {df.shape[0]}')

                # self.mean_centre_x = df["ring_centre_pos_x"].mean()
                # self.mean_centre_y = df["ring_centre_pos_y"].mean()

                indices = df['original_index'].to_numpy()

                logger.info(f"Unique labels: {df['label'].unique()}")

                # remove df from memory
                del df
            else:
                indices = np.arange(self.N - 2)

            # Global Variables
            self.mean_centre_x, self.mean_centre_y = get_config(
                'dataset.centre_bias.mean_x'
            ), get_config('dataset.centre_bias.mean_y')
            (
                self.mean_momentum,
                self.std_momentum,
                self.mean_radius,
                self.std_radius,
            ) = (
                get_config('dataset.standardize.mean_momentum'),
                get_config('dataset.standardize.std_momentum'),
                get_config('dataset.standardize.mean_radius'),
                get_config('dataset.standardize.std_momentum'),
                get_config('dataset.standardize.mean_radius'),
                get_config('dataset.standardize.std_radius'),
            )

            logger.info(
                f"""
                    Mean centre locations: ({self.mean_centre_x},{self.mean_centre_y})
                    Mean/std momentum: {self.mean_momentum}, {self.std_momentum}
                    Mean/std radius: {self.mean_radius}, {self.std_radius}"""
            )

            # shuffle indices
            np.random.shuffle(indices)

            logger.info(f'Total indices: {len(indices)}')
            self.train_indices = []
            self.test_indices = []
            self.val_indices = []

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

            logger.info(f'Total Train indices: {len(self.train_indices)}')
            logger.info(f'Sample Train indices: {self.train_indices[:10]}')
            logger.info(f'Total Validation indices: {len(self.val_indices)}')
            logger.info(f'Sample Validation indices: {self.val_indices[:10]}')
            logger.info(f'Total Test indices: {len(self.test_indices)}')
            logger.info(f'Sample Test indices: {self.test_indices[:10]}')

            if test_only:

                # self.test_indices = indices

                df = events_to_pandas(dfile)

                logger.info('Testing....................')
                logger.info(f'Total events: {df.shape[0]}')

                # filter out label = 2, positron
                df = df.query('label != 2')
                logger.info(f'Total events with no positrons: {df.shape[0]}')

                # filter momentum
                momentum_lower, momentum_upper = get_config(
                    'dataset.filters.momentum_lower'
                ), get_config('dataset.filters.momentum_upper')

                df = df.query(
                    f'track_momentum >= {momentum_lower} and track_momentum < {momentum_upper}'
                )
                logger.info(
                    f'Total events with track_momentum between {momentum_lower} and {momentum_upper}: {df.shape[0]}'
                )

                # filter ring centre outliers (some very small or large values in data)
                radius_lower, radius_upper = get_config(
                    'dataset.filters.radius_lower'
                ), get_config('dataset.filters.radius_upper')
                df = df.query(
                    f'ring_centre_pos_x < {radius_upper} and ring_centre_pos_y < {radius_upper}'
                )
                df = df.query(
                    f'ring_centre_pos_x > {radius_lower} and ring_centre_pos_y > {radius_lower}'
                )
                logger.info(
                    f'Total events with no outliers (radius between {radius_lower} and {radius_upper})): {df.shape[0]}'
                )

                indices = df.index.to_numpy()

                logger.info(f'Total Test indices: {len(indices)}')
                logger.info(f"Unique labels: {df['label'].unique()}")

                self.test_indices = indices

                del df

        logger.info(f'Total Train indices: {len(self.train_indices)}')
        logger.info(f'Total Validation indices: {len(self.val_indices)}')
        logger.info(f'Total Test indices: {len(self.test_indices)}')

        # for reproducibility (to check that test set is the same)
        logger.info(f'First 5 Train indices: {self.train_indices[:5]}')
        logger.info(f'First 5 Validation indices: {self.val_indices[:5]}')
        logger.info(f'First 5 Test indices: {self.test_indices[:5]}')

        # We don't attempt to catch exception here, crash if we cannot open the file.
        with open(dset_path, 'rb') as fh:
            fileno = fh.fileno()
            mapping = mmap.mmap(fileno, 0, access=mmap.ACCESS_READ)
            self.hit_array = np.frombuffer(
                mapping, dtype=hit_dtype, count=hit_length, offset=hit_offset
            ).reshape(hit_shape)
            logger.info('hit array mmap size: %i bytes', self.hit_array.nbytes)
            self.event_array = np.frombuffer(
                mapping,
                dtype=event_dtype,
                count=event_length,
                offset=event_offset,
            ).reshape(event_shape)
            logger.info(
                'event array mmap size: %i bytes', self.event_array.nbytes
            )

    def augment_data(self, data):
        """Data Augmentation"""
        theta = np.random.uniform(0, np.pi * 2)
        rot_mat = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        data[:, [0, 2]] = data[:, [0, 2]].dot(rot_mat)  # random rotation
        data = data + np.random.normal(
            0, 0.02, size=data.shape
        )  # random jitter
        return data

    def get_position_data(self):
        return np.load('rich_pmt_positions.npy')

    def get_event_pos(self, idx):
        """Get the hits data for a single event."""
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
            self.event_array[idx]['chod_time']
            - self.hit_array[idx_from:idx_to]['hit_time']
        )

        if event_pos.shape[0] > position_map.shape[0]:
            logger.warning('Unusual event pos')
            return position_map

        # filter events for small time deltas
        event_pos = self.filter_events_delta(event_pos, self.delta)

        # demean x and y locations
        event_pos[:, :1] = event_pos[:, :1] - self.mean_centre_x
        event_pos[:, 1:2] = event_pos[:, 1:2] - self.mean_centre_y

        # pad with zeros
        data = np.zeros_like(position_map)
        data[: event_pos.shape[0], : event_pos.shape[1]] = event_pos

        # data augmentation
        if self.data_augmentation:
            data = self.augment_data(data)

        return data

    def filter_events_delta(self, event, delta):
        """Filter hits data based on a delta = hit time - chod time"""
        mask = (np.abs(event[:, 2:3]) < delta).flatten()
        return event[mask]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        idx_from = self.hit_mapping[idx]
        idx_to = self.hit_mapping[idx + 1]

        self.data = {
            'event_pos': self.get_event_pos(idx),
            'label': self.labels[idx],
            'hit_time': self.hit_array[idx_from:idx_to]['hit_time'],
            'chod_time': self.event_array[idx]['chod_time'],
            'track_momentum': self.event_array[idx]['track_momentum'],
            'ring_radius': self.event_array[idx]['ring_radius'],
        }

        # standardize momentum
        self.data['track_momentum'] -= self.mean_momentum
        self.data['track_momentum'] /= self.std_momentum

        # standardize ring radius
        self.data['ring_radius'] -= self.mean_radius
        self.data['ring_radius'] /= self.std_radius

        return (
            torch.tensor(self.data['event_pos']),
            torch.tensor(self.data['label']),
            torch.tensor(self.data['track_momentum']),
            torch.tensor(self.data['ring_radius']),
        )
