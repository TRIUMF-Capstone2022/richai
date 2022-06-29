"""
Module to create balanced dataset for the RICH AI project.
"""

import h5py
import pandas as pd
from utils.helpers import *


def undersample(df, seed=123):
    """Undersamples data as per particle class

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing entries for a momentum bin
    seed: int,optional
        Random seed. Defaults to 123
    Returns
    -------
    pandas.DataFrame
        Dataframe after undersampling entries
    """
    # Computing the count of the smaller particle class
    classes = df.label.value_counts().to_dict()
    least_class_amount = min(classes.values())

    # Balancing particle classes as per smaller particle class size
    classes_list = []
    for key in classes:
        classes_list.append(df[df["label"] == key])
    classes_sample = []
    for i in range(0, len(classes_list) - 1):
        classes_sample.append(
            classes_list[i].sample(least_class_amount, random_state=seed)
        )
    df_maybe = pd.concat(classes_sample)
    final_df = pd.concat([df_maybe, classes_list[-1]], axis=0)
    final_df = final_df.reset_index(drop=True)
    return final_df


def balance_data(dset_path=get_config("dataset.train").items(), seed=123):
    """Creates and saves balanced dataset with undersampled momentum bins and saves it to specified path

    Parameters
    ----------
    dset_path : dict, optional
        Dictionary containing source file as keys and destination file as values.
        By default get_config("dataset.train").keys()
    seed: int
        Random seed
    """

    for dset_path_raw, dset_path_bal in dset_path:
        f = h5py.File(dset_path_raw)

        # Reading source hdf5 file to pandas dataframe
        df = events_to_pandas(f)
        df["original_index"] = df.index
        df = df.drop(columns=["first_hit", "last_hit", "total_hits"])

        # Removing positrons
        df = df[(df["label"] == 0) | (df["label"] == 1)]

        # Creating momentum bins
        df["momentum-bin"] = np.where(
            (15 <= df["track_momentum"]) & (df["track_momentum"] < 25),
            "15-25",
            np.where(
                (25 <= df["track_momentum"]) & (df["track_momentum"] < 35),
                "25-35",
                np.where(
                    (35 <= df["track_momentum"]) & (df["track_momentum"] < 45),
                    "35-45",
                    ">45",
                ),
            ),
        )

        # Undersampling
        bins = ["15-25", "25-35", "35-45"]
        balanced_df = {}
        for i in bins:
            balanced_df[i] = undersample(df[df["momentum-bin"] == i], seed)

        # Combining dataframes
        combined = pd.concat(
            [balanced_df["15-25"], balanced_df["25-35"], balanced_df["35-45"]],
            ignore_index=True,
        )

        # Saving to hdf5
        combined.to_hdf(dset_path_bal, key="combined", format="table", mode="w")
