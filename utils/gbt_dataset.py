# Imports
import pandas as pd
import numpy as np
import cudf
import cupy as cp
import h5py
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from dataset.rich_dataset import RICHDataset


def events_to_cudf(dfile, gpu_id):
    """Convert events data from HDF5 to cudf DataFrame

    Parameters
    ----------
    dfile : HDF5
        HDF5 file read in using h5py.File on source HDF5 dataset
    gpu_id : int
        GPU device id

    Returns
    -------
    cudf.DataFrame
        DataFrame with Events data
    """

    cp.cuda.Device(gpu_id).use()
    df = cudf.DataFrame()

    # event features
    df["run_id"] = dfile["Events"]["run_id"]
    df["burst_id"] = dfile["Events"]["burst_id"]
    df["event_id"] = dfile["Events"]["event_id"]
    df["track_id"] = dfile["Events"]["track_id"]
    df["track_momentum"] = dfile["Events"]["track_momentum"]
    df["chod_time"] = dfile["Events"]["chod_time"]
    df["ring_radius"] = dfile["Events"]["ring_radius"]
    df["ring_centre_pos_x"] = dfile["Events"]["ring_centre_pos"][:, 0]
    df["ring_centre_pos_y"] = dfile["Events"]["ring_centre_pos"][:, 1]
    df["ring_likelihood_pion"] = dfile["Events"]["ring_likelihood"][:, 0]
    df["ring_likelihood_muon"] = dfile["Events"]["ring_likelihood"][:, 1]
    df["ring_likelihood_positron"] = dfile["Events"]["ring_likelihood"][:, 2]

    # labels
    mu_off = dfile.attrs["muon_offset"]
    pi_off = dfile.attrs["pion_offset"]
    pos_off = dfile.attrs["positron_offset"]
    entries = dfile.attrs["entries"]

    labels = np.zeros(entries, dtype="int32")
    labels[mu_off:pi_off] = 0
    labels[pi_off:pos_off] = 1
    labels[pos_off:] = 2

    df["label"] = labels

    # hit mapping values
    df["first_hit"] = np.array(dfile["HitMapping"])[:-1]  # hit n
    df["last_hit"] = np.array(dfile["HitMapping"])[1:]  # hit n + 1
    df["total_hits"] = df["last_hit"] - df["first_hit"]

    return df


def gbt_df(dset_path_raw, dset_path_bal, gpu_id, delta=0.3):
    """Creates dataset for Gradient Boosted Decision Tree model

    Args:
        dset_path_raw (str): _File path for raw HDF5 source dataset_
        dset_path_bal (str): _File path for momentum balanced HDF5 dataset_
        gpu_id(int): _GPU id to be used_
        delta(float, optional): _delta value (chod_time - hit_time) for filtering noise in hits data_

    Returns:
        pandas.DataFrame: Processed dataframe with features for Gradient Boosted Decision Tree model

    """
    # Reading in data
    print("Reading data...")
    dfile = h5py.File(dset_path_raw)
    df_raw = events_to_cudf(dfile, gpu_id)
    df_raw = df_raw.to_pandas()

    if dset_path_bal:
        df_bal = pd.read_hdf(dset_path_bal)
        df_raw = df_raw.iloc[df_bal.original_index, :]

    print("Filtering hits...")
    # Filtering hits based on delta (chod_time - hit_time)
    event_idx = df_raw.index

    total_hits_filtered = pd.Series(
        np.zeros(df_raw.shape[0], dtype="int32"), index=event_idx, dtype="int32"
    )

    dset_raw = RICHDataset(dset_path_raw)
    for idx in event_idx:
        # Finding the number of hits for each event in the hit_array from hit_mapping
        idx_from = dset_raw.hit_mapping[idx]
        idx_to = dset_raw.hit_mapping[idx + 1]
        hit_times = dset_raw.hit_array["hit_time"][idx_from:idx_to]
        delta_time = dset_raw.event_array[idx]["chod_time"] - hit_times
        total_hits_filtered[idx] = delta_time[np.abs(delta_time) < delta].shape[0]

    df_raw["total_hits_filtered"] = total_hits_filtered

    # Removing outliers or anomalous entries
    print("Removing outliers or anomalous entries")
    # ring_center_pos_x
    df_processed = df_raw[
        (df_raw["ring_centre_pos_x"] < 1000) & (df_raw["ring_centre_pos_x"] > -1000)
    ]

    # ring_center_pos_y
    df_processed = df_processed[
        (df_processed["ring_centre_pos_y"] < 1000)
        & (df_processed["ring_centre_pos_y"] > -1000)
    ]

    # Ring radius
    df_processed = df_processed[
        (df_processed["ring_radius"] > 1) & (df_processed["ring_radius"] < 1000)
    ]

    # Extracting final features
    df_processed = df_processed.loc[
        :, ["track_momentum", "ring_radius", "total_hits_filtered", "label"]
    ]
    print("Dataset preparation complete")
    return df_processed


def model_results(model, X_test, y_test):
    """Evaluates models on test data with results as classification report and confusion matrix

    Parameters
    ----------
    model : model object
        Trained decision tree model object
    X_test : pandas.DataFrame
        Test data features
    y_test : pandas.DataFrame
        Target or y of test data
    """
    print("\nClassification Report\n")
    print(
        classification_report(
            y_test, model.predict(X_test), target_names=["Muons", "Pions"]
        )
    )
    # Confusion matrix
    cm = confusion_matrix(y_test, model.predict(X_test), normalize="true")
    cm_df = pd.DataFrame(cm, index=["Muons", "Pions"], columns=["Muons", "Pions"])
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm_df, annot=True, cmap="YlGnBu")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Values")
    plt.xlabel("Predicted Values")
    plt.show()


def gbt_binwise(model, df_bal, bin_low, bin_high):
    """Creates train and test data from source dataframe and trains model locally on momentum bins

    Parameters
    ----------
    model : model object
        Untrained gradient boosted decision tree classifier instance
    df_bal : pandas.DataFrame
        Balanced and processed train data
    bin_low : int
        Lower limit value of momentum bin
    bin_high : int
        Upper limit value of momentum bin

    Returns
    -------
    model: model object
        Trained model
    X_train: pandas.DataFrame
        Training data (features) corresponding to specified momentum bin
    y_train: pandas.DataFrame)
        Training data (target) corresponding to specified momentum bin
    X_test: pandas.DataFrame
        Test data (features) corresponding to specified momentum bin
    y_test: pandas.DataFrame
        Test data (target) corresponding to specified momentum bin
    """

    # Creating df in separate momentum bins
    df_bal_bin = df_bal[
        (df_bal.track_momentum < bin_high) & (df_bal.track_momentum > bin_low)
    ]

    # Selecting X & y
    X = df_bal_bin.loc[:, ["track_momentum", "ring_radius", "total_hits_filtered"]]
    y = df_bal_bin.loc[:, "label"]

    # Training and validation split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Model training
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test
