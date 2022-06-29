"""
Plotting functions for the RICH AI project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    RocCurveDisplay,
    recall_score,
)


def plot_cm(
    y_true,
    y_pred,
    target_names=["muon", "pion"],
    normalize="true",
    cmap="cividis",
    title=None,
    save=None,
):
    """Plot a confusion matrix for a classifier.

    Parameters
    ----------
    y_true : pandas Series
        The actual label values.
    y_pred : pandas Series
        The predicted label values.
    target_names : list of str, by default ["muon", "pion"]
        The class labels.
    normalize : str or None
        "true" normalizes by rows, "pred" normalizes by columns.
    cmap : str or None
        Matplotlib colormap.
    title : str or None
        Plot title.
    save: str or None, optional
        Path where to save figure, if desired.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    disp.plot(ax=ax, cmap=cmap, values_format=".4f")

    ax.set_ylabel("Actual class", fontsize=18)
    ax.set_xlabel("Predicted class", fontsize=18)
    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)

    if title:
        plt.title(title, fontsize=20)

    for labels in disp.text_.ravel():
        labels.set_fontsize(15)

    plt.tight_layout()

    if save:
        fig.savefig(save)

    plt.show()


def show_results(path, title=None):
    """Show the classification report and confusion matrix for a classifier.

    Parameters
    ----------
    path : str
        The path to the model results .csv file.
    title : str or None
        The title for the confusion matrix.

    Returns
    -------
    None
    """
    results = pd.read_csv(path)
    target_names = ["muon", "pion"]

    # classification report
    print(
        classification_report(
            y_true=results["labels"],
            y_pred=results["predictions"],
            target_names=target_names,
        )
    )

    # confusion matrix
    plot_cm(
        y_true=results["labels"],
        y_pred=results["predictions"],
        target_names=target_names,
        title=title,
    )


def plot_roc_curves(models, title=None, op_point=None, save=None):
    """Plot ROC curves for multiple models.

    Parameters
    ----------
    models : dict
        Dictionary where the key: values are "name: model path".
        "name" will be shown in the legend of the plot for that model.
        "model path" is the file path to the csv results for that model.
    title : str
        Plot title.
    op_point : float, optional
        The operating point to plot on the ROC curve, by default None.
    save: str or None, optional
        Path where to save figure, if desired.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    for name, path in models.items():
        df = pd.read_csv(path)

        fpr, tpr, thresholds = roc_curve(
            y_true=df["labels"],
            y_score=df["probabilities"],
        )

        if op_point is not None:
            # location of operating point
            op_point_loc = np.argmin(np.abs(thresholds - op_point))

            # plot operating point
            ax.plot(fpr[op_point_loc], tpr[op_point_loc], "o", markersize=8, c="r")

        # plot roc curve
        disp = RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name=name)
        disp.plot(ax=ax)

    # 45 degree line is equivalent of a random classifier
    ax.plot([0, 1], [0, 1], color="black", linestyle="--")
    ax.set_xlabel("False Positive Rate (Muon/Pion)", fontsize=20)
    ax.set_ylabel("True Positive Rate (Pion/Pion)", fontsize=20)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(18)

    if title:
        ax.set_title(title, fontsize=25)

    plt.tight_layout()
    plt.legend(fontsize=15)

    if save:
        fig.savefig(save)

    plt.show()


def wrangle_predictions(path, width=1, op_point=None):
    """Wrangle model predictions in order to plot efficiency curves.

    Parameters
    ----------
    path : str
        Path to .csv file that contains model predictions to wrangle.
        Columns must be: labels, predictions, probabilities, momentum.
    width : int, optional
        Width of momentum bins to add to data, by default 1

    Returns
    -------
    wrangled_df: pd.DataFrame
        Wrangled data for plotting.
    """
    df = pd.read_csv(path)
    bins = []
    bin_labels = []

    if op_point is not None:
        df["predictions"] = np.where(df["probabilities"] > op_point, 1, 0)

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


def plot_efficiencies(
    path,
    title=None,
    cern_scale=True,
    pion_axlims=(0.4, 1),
    muon_axlims=(0, 200),
    pion_axticks=(0.05, 0.01),
    muon_axticks=(10, 1),
    save=None,
    op_point=None,
):
    """Plot pion and muon efficiencies by momentum bin.

    Parameters
    ----------
    path : str
        Path to .csv file that contains model predictions.
        Columns must be: labels, predictions, probabilities, momentum.
    title : str, optional
        Overall title for the plot.
    pion_axlims : tuple of float
        Bounds for the pion y-axis given as: (lower, upper).
    muon_axlims : tuple of float
        Bounds for the muon y-axis given as: (lower, upper).
    pion_axticks : tuple of float
        Tick locations for the pion y-axis given as: (major, minor).
    muon_axticks : tuple of float
        Tick locations for the muon y-axis given as: (major, minor).
    save: str or None, optional
        Path where to save figure, if desired.

    Returns
    -------
    None
    """
    results_df = wrangle_predictions(path, op_point=op_point)

    fig, ax1 = plt.subplots(figsize=(9, 7))

    # global plot variables
    pion_color = "black"  # pion plotting color
    muon_color = "blue"  # muon plotting color
    label_fs = 17  # axis label font size
    tick_fs = 13  # axis tick font size
    major_length, major_width = 7, 2  # major axis tick sizes
    minor_length, minor_width = 4, 1  # minor axis tick sizes
    main_ms, line_ms = 7, 9  # circle/square/line marker sizes
    labelpad = 10  # padding for axis labels

    # plot pion efficiency as circles with horizontal lines
    results_df["pion_efficiency"].plot(
        ax=ax1, color=pion_color, style="o", markersize=main_ms
    )
    results_df["pion_efficiency"].plot(
        ax=ax1, marker=0, linestyle="", markersize=line_ms, color=pion_color
    )
    results_df["pion_efficiency"].plot(
        ax=ax1, marker=1, linestyle="", markersize=line_ms, color=pion_color
    )

    # momentum x-axis customization (shared axis)
    ax1.set_xlabel(
        "Particle momentum [GeV/c]",
        fontsize=label_fs,
        fontweight="bold",
        labelpad=labelpad,
    )
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax1.grid(True, linestyle="--")

    for label in ax1.get_xticklabels():
        label.set_fontweight("bold")

    # pion y-axis customization
    ax1.set_ylim(pion_axlims[0], pion_axlims[1])
    ax1.set_ylabel(
        "Pion efficiency (recall)",
        color=pion_color,
        fontsize=label_fs,
        fontweight="bold",
        labelpad=labelpad,
    )
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(pion_axticks[0]))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(pion_axticks[1]))
    ax1.tick_params(
        which="major",
        length=major_length,
        width=major_width,
        labelsize=tick_fs,
    )
    ax1.tick_params(which="minor", length=minor_length, width=minor_width)

    for label in ax1.get_yticklabels():
        label.set_fontweight("bold")

    # second x-axis for muon
    ax2 = ax1.twinx()

    # whether or not to match muon scale to original NA62 CERN plot (10^-3)
    if cern_scale:
        results_df["muon_efficiency"] *= 10**3
        ax2.text(
            28.5,
            muon_axlims[1],
            r"x$10^{-3}$",
            fontsize=label_fs,
            color=muon_color,
        )

    # plot muon efficiency as squares with horizontal lines
    results_df["muon_efficiency"].plot(
        ax=ax2, color=muon_color, style="s", markersize=main_ms
    )
    results_df["muon_efficiency"].plot(
        ax=ax2, marker=0, linestyle="", markersize=line_ms, color=muon_color
    )
    results_df["muon_efficiency"].plot(
        ax=ax2, marker=1, linestyle="", markersize=line_ms, color=muon_color
    )

    # muon x-axis customization
    ax2.set_ylim(muon_axlims[0], muon_axlims[1])
    ax2.set_ylabel(
        "Muon Efficiency (1 - recall)",
        color=muon_color,
        fontsize=label_fs,
        fontweight="bold",
        labelpad=labelpad,
    )
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(muon_axticks[0]))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(muon_axticks[1]))
    ax2.tick_params(
        which="major",
        length=major_length,
        width=major_width,
        labelsize=tick_fs,
    )
    ax2.tick_params(which="minor", length=minor_length, width=minor_width)
    ax2.tick_params(which="both", color=muon_color, labelcolor=muon_color)

    for label in ax2.get_yticklabels():
        label.set_fontweight("bold")

    if title:
        plt.title(title, fontsize=20)

    plt.tight_layout()

    if save:
        fig.savefig(save)

    plt.show()
