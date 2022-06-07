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
)


# TODO add option to save figures and save paths


def plot_cm(
    y_true, y_pred, target_names=["muon", "pion"], normalize="true", title=None
):
    """Plot a confusion matrix for a classifier.

    Parameters
    ----------
    y_true : pandas Series
        The actual label values.
    y_pred : pandas Series
        The predicted label values.
    target_names : list of str, by default ["muon", "pion"]
        The class labels
    normalize : str, optional
        Whether or not to normalize the CM, by default "true"
    title : str, optional
        Plot title, by default None
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    disp.plot(ax=ax, cmap="cividis", values_format=".4f")

    ax.set_ylabel("Actual class", fontsize=18)
    ax.set_xlabel("Predicted class", fontsize=18)
    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)

    if title:
        plt.suptitle(title, y=0.92, fontsize=20)

    for labels in disp.text_.ravel():
        labels.set_fontsize(15)

    plt.show()


def show_results(path):
    """Show the classification report and confusion matrix for a classifier.

    Parameters
    ----------
    path : str
        The path to the model results .csv file.
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
    )


def plot_roc_curves(models, title, op_point=None):
    """Plot ROC curves for multiple models.

    Parameters
    ----------
    models : dict
        Dictionary where the key: values are "name: model path".
        "name" will be shown in the legend of the plot for that model.
        "model path" is the file path to the csv results for that model.
    title : str
        Plot titile.
    op_point : float, optional
        The operating point to plot on the ROC curve, by default None.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # iterate through models and plot ROC curve for each
    for name, path in models.items():
        df = pd.read_csv(path)

        fpr, tpr, thresholds = roc_curve(
            y_true=df["labels"],
            y_score=df["probabilities"],
        )

        # optionally plot operating points
        if op_point is not None:
            op_point_loc = np.argmin(np.abs(thresholds - op_point))
            ax.plot(fpr[op_point_loc], tpr[op_point_loc], "o", markersize=8, c="r")

        # plot roc curve
        disp = RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name=name)
        disp.plot(ax=ax)

    ax.set_title(title, fontsize=20)
    ax.set_xlabel("False Positive Rate (Muon/Pion)", fontsize=16)
    ax.set_ylabel("True Positive Rate (Pion/Pion)", fontsize=16)

    # 45 degree line is equivalent of a random classifier
    ax.plot([0, 1], [0, 1], color="black", linestyle="--")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(16)

    plt.show()


def plot_efficiencies(results_df, title=None):
    """Plot pion and muon efficiencies by momentum bin.

    Parameters
    ----------
    results_df : pandas DataFrame
        Prediction data containing momentum bins and efficiencies.
    title : str, optional
        Overall title for the plot.
    """
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
    ax1.set_ylim(0.7)
    ax1.set_ylabel(
        "Pion efficiency",
        color=pion_color,
        fontsize=label_fs,
        fontweight="bold",
        labelpad=labelpad,
    )
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax1.tick_params(
        which="major", length=major_length, width=major_width, labelsize=tick_fs
    )
    ax1.tick_params(which="minor", length=minor_length, width=minor_width)

    for label in ax1.get_yticklabels():
        label.set_fontweight("bold")

    # second x-axis for muon
    ax2 = ax1.twinx()

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
    ax2.set_ylim(0, 0.3)
    ax2.set_ylabel(
        "Muon efficiency",
        color=muon_color,
        fontsize=label_fs,
        fontweight="bold",
        labelpad=labelpad,
    )
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax2.tick_params(
        which="major", length=major_length, width=major_width, labelsize=tick_fs
    )
    ax2.tick_params(which="minor", length=minor_length, width=minor_width)
    ax2.tick_params(which="both", color=muon_color, labelcolor=muon_color)

    for label in ax2.get_yticklabels():
        label.set_fontweight("bold")

    if title:
        plt.suptitle(title, fontsize=20, y=0.94)

    plt.show()
