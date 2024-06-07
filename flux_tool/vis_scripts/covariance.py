import itertools
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from flux_tool.vis_scripts.helper import save_figure
from flux_tool.vis_scripts.spectra_reader import SpectraReader
from flux_tool.vis_scripts.style import neutrino_labels


def plot_matrices(
    matrices: dict[str, Any],
    horn_currents: list[str],
    nbins: dict[str, int],
    vlim: tuple[float, float] | str = (-1, 1),
):
    bin_ordering = [nbins["nue"], nbins["nuebar"], nbins["numu"], nbins["numubar"]]

    if len(horn_currents) == 2:
        bin_ordering += bin_ordering[:-1]

    line_positions = list(itertools.accumulate(bin_ordering))

    nue = neutrino_labels["nue"]
    nueb = neutrino_labels["nuebar"]
    numu = neutrino_labels["numu"]
    numub = neutrino_labels["numubar"]

    for key, mat in matrices.items():
        fig, ax = plt.subplots()  # layout="constrained")

        ax.set_box_aspect(1)

        m = mat.to_numpy()[0]

        if isinstance(vlim, tuple):
            vmin, vmax = vlim
        elif vlim == "auto":
            vmax = np.amax(m)
            vmin = -vmax
        else:
            raise ValueError(f"Unrecognized argument passed to plot_matrices: {vlim=}")

        heatmap_kwargs = {
            "ax": ax,
            "cmap": "bwr",
            "cbar": True,
            "vmin": vmin,
            "vmax": vmax,
            "square": False,
            "cbar_kws": {"shrink": 0.81},
        }

        sns.heatmap(m, **heatmap_kwargs)

        ax.invert_yaxis()  # type: ignore
        # ax.set_aspect(m.shape[1] / m.shape[0])  # type: ignore

        ax.tick_params(  # type: ignore
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
            which="both",
        )

        kwargs = {
            "transform": ax.transAxes,  # type: ignore
            "fontsize": 32,
            "fontweight": "bold",
            # "ha": "center",
            # "va": "top",
        }
        kwargs2 = {
            "transform": ax.transAxes,  # type: ignore
            "fontsize": 32,
            "fontweight": "bold",
            # "va": "center",
            # "ha": "right",
        }

        # ax.annotate(  # type: ignore
        #     "ICARUS Preliminary",
        #     (0.0, 1.0),
        #     xytext=(0, 3),
        #     xycoords="axes fraction",
        #     textcoords="offset points",
        #     ha="left",
        #     va="bottom",
        #     fontweight="bold",
        # )  # type: ignore

        if len(horn_currents) == 1:
            xpos = -0.015
            nu_coords = [[xpos, 0.09], [xpos, 0.25], [xpos, 0.50], [xpos, 0.85]]
            horn = horn_currents[0].upper()
            ax.text(0.50, -0.085, horn, ha="center", va="top", **kwargs)  # type: ignore
            ax.text(-0.085, 0.50, horn, rotation=90, ha="right", va="center", **kwargs2)  # type: ignore

            for (x, y), nu in zip(nu_coords, neutrino_labels.values()):
                ax.text(x, y, nu, ha="right", va="center", **kwargs)  # type: ignore
                ax.text(y, x, nu, ha="center", va="top", **kwargs)  # type: ignore

            for pos in line_positions:
                lw = 1
                ax.axvline(pos, color="k", lw=lw)
                ax.axhline(pos, color="k", lw=lw)
        else:
            ax.text(0.245, -0.085, "FHC", ha="center", va="top", **kwargs)  # type: ignore
            ax.text(0.765, -0.085, "RHC", ha="center", va="top", **kwargs)  # type: ignore
            ax.text(  # type: ignore
                -0.085, 0.245, "FHC", rotation=90, ha="right", va="center", **kwargs2  # type: ignore
            )
            ax.text(  # type: ignore
                -0.085, 0.765, "RHC", rotation=90, ha="right", va="center", **kwargs2  # type: ignore
            )

            xpos = -0.015
            nu_coords = [
                [xpos, 0.05, nue],
                [xpos, 0.18, nueb],
                [xpos, 0.31, numu],
                [xpos, 0.44, numub],
                [xpos, 0.57, nue],
                [xpos, 0.70, nueb],
                [xpos, 0.83, numu],
                [xpos, 0.96, numub],
            ]

            for x, y, nu in nu_coords:
                ax.text(x, y, nu, ha="right", va="center", **kwargs2)  # type: ignore
                ax.text(y, x, nu, ha="center", va="top", **kwargs)  # type: ignore

            for i, pos in enumerate(line_positions):
                lw = 2 if i == 3 else 1
                ax.axvline(pos, color="k", lw=lw)
                ax.axhline(pos, color="k", lw=lw)

        yield key, fig


def plot_hadron_correlation_matrices(
    reader: SpectraReader, output_dir: Optional[Path] = None
):
    horn_currents = reader.horn_current
    matrices = reader.hadron_correlation_matrices

    nbins = {k: len(v) - 1 for k, v in reader.binning.items()}

    figures = plot_matrices(matrices, horn_currents, nbins)

    if output_dir is not None:
        for key, fig in figures:
            category = key.split("/")[1]
            file_stem = f"{category}_correlation_matrix"
            tex_caption = ""
            tex_label = file_stem
            save_figure(fig, file_stem, output_dir, tex_caption, tex_label)  # type: ignore
            plt.close(fig)


def plot_hadron_covariance_matrices(
    reader: SpectraReader, output_dir: Optional[Path] = None
):
    horn_currents = reader.horn_current
    matrices = reader.hadron_covariance_matrices

    nbins = {k: len(v) - 1 for k, v in reader.binning.items()}

    figures = plot_matrices(matrices, horn_currents, nbins, vlim="auto")

    if output_dir is not None:
        for key, fig in figures:
            category = key.split("/")[1]
            file_stem = f"{category}_covariance_matrix"
            tex_caption = ""
            tex_label = file_stem
            save_figure(fig, file_stem, output_dir, tex_caption, tex_label)  # type: ignore
            plt.close(fig)


def plot_beam_correlation_matrices(
    reader: SpectraReader, output_dir: Optional[Path] = None
):
    horn_currents = reader.horn_current
    matrices = reader.beam_correlation_matrices

    nbins = {k: len(v) - 1 for k, v in reader.binning.items()}

    figures = plot_matrices(matrices, horn_currents, nbins)

    if output_dir is not None:
        for key, fig in figures:
            category = key
            if "/" in category:
                category = category.split("/")[1]
            category = category.split("_", 1)[1]
            file_stem = f"{category}_correlation_matrix"
            tex_caption = ""
            tex_label = file_stem
            save_figure(fig, file_stem, output_dir, tex_caption, tex_label)  # type: ignore
            plt.close(fig)


def plot_beam_covariance_matrices(
    reader: SpectraReader, output_dir: Optional[Path] = None
):
    horn_currents = reader.horn_current
    matrices = reader.beam_covariance_matrices

    nbins = {k: len(v) - 1 for k, v in reader.binning.items()}

    figures = plot_matrices(matrices, horn_currents, nbins, vlim="auto")

    if output_dir is not None:
        for key, fig in figures:
            category = key
            if "/" in category:
                category = category.split("/")[1]
            category = category.split("_", 1)[1]
            file_stem = f"{category}_covariance_matrix"
            tex_caption = ""
            tex_label = file_stem
            save_figure(fig, file_stem, output_dir, tex_caption, tex_label)  # type: ignore
            plt.close(fig)
