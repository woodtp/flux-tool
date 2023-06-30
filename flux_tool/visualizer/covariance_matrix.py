import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.pyplot import Axes, Figure
from numpy import arange

from flux_tool.visualizer.style import neutrino_labels


def plot_covariance(
    Matrix: pd.DataFrame, cbar_label=None, **kwargs
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots()

    params = {
        "ax": ax,
        "cmap": "bwr",
        "cbar": True,
        "vmin": -1.0,
        "vmax": 1.0,
        "square": False,
        "cbar_kws": {"shrink": 0.81},
    }

    params.update(kwargs)

    sns.heatmap(Matrix, **params)

    ax.invert_yaxis()
    ax.set_aspect(Matrix.shape[1] / Matrix.shape[0])

    if cbar_label is not None:
        cbar = ax.collections[0].colorbar

        cbar.set_label(cbar_label, loc="center", rotation=270)

    # nrows = Matrix.shape[0]
    #
    # axis_labels = [
    #     neutrino_labels["nue"] + r"$^{fhc}$",
    #     neutrino_labels["nuebar"] + r"$^{fhc}$",
    #     neutrino_labels["numu"] + r"$^{fhc}$",
    #     neutrino_labels["numubar"] + r"$^{fhc}$",
    #     neutrino_labels["nue"] + r"$^{rhc}$",
    #     neutrino_labels["nuebar"] + r"$^{rhc}$",
    #     neutrino_labels["numu"] + r"$^{rhc}$",
    #     neutrino_labels["numubar"] + r"$^{rhc}$",
    # ]
    #
    # n_boxes = len(axis_labels)
    #
    # nbins = int(nrows / n_boxes)
    #
    # divisions = arange(len(axis_labels), nrows, step=nbins)
    #
    # ax.set_xticks(divisions)
    # ax.set_yticks(divisions)
    #
    # ax.set_xticklabels(
    #     axis_labels,
    #     rotation=0,
    #     verticalalignment="top",
    # )
    # ax.set_yticklabels(axis_labels)
    #
    # ax.set_xlabel("")
    # ax.set_ylabel("")
    #
    # for i in range(1, n_boxes + 1):
    #     lw = 2 if i == int(n_boxes / 2) else 1
    #     ax.axvline(x=i * nbins, linewidth=lw, color="k")
    #     ax.axhline(y=i * nbins, linewidth=lw, color="k")

    return fig, ax
