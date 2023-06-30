from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
from numpy import ndarray
from pandas import DataFrame

from flux_tool.visualizer.style import neutrino_labels, okabe_ito, xlabel_enu, ylabel_flux


def plot_ppfx_universes(
    universes: DataFrame,
    nominal_flux: DataFrame,
    ppfx_correction: DataFrame,
    bins: ndarray,
    nu: str,
) -> list[Path]:
    fig, ax = plt.subplots()

    ax.plot([], [], " ", label="PPFX Input")

    nu_label = neutrino_labels[nu]

    hep.histplot(
        H=nominal_flux.to_numpy(),
        label=f"{nu_label} nominal",
        bins=bins,
        color="k",
        lw=3,
        yerr=False,
        edges=False,
        binwnorm=1,
        ax=ax,
    )

    ax.plot([], [], " ", label="PPFX Output")

    universes_np = universes.to_numpy()

    hep.histplot(
        H=universes_np[0],
        bins=bins,
        label=f"{nu_label} universes",
        color=okabe_ito["skyblue"],
        lw=1,
        yerr=False,
        edges=False,
        binwnorm=1,
        ax=ax,
    )
    hep.histplot(
        H=universes_np[1:],
        bins=bins,
        color=okabe_ito["skyblue"],
        lw=1,
        yerr=False,
        edges=False,
        binwnorm=1,
        ax=ax,
    )

    hep.histplot(
        H=ppfx_correction["mean"].to_numpy(),
        yerr=ppfx_correction["sigma"].to_numpy(),
        bins=bins,
        label=f"{neutrino_labels[nu]} mean",
        color=okabe_ito["blue"],
        histtype="errorbar",
        xerr=True,
        capsize=4,
        elinewidth=4,
        marker=None,
        binwnorm=1,
        ax=ax,
    )

    ax.set_xlim(0, 6)
    ax.set_xlabel(xlabel_enu)
    ax.set_ylabel(ylabel_flux)

    handles, labels = ax.get_legend_handles_labels()

    leg = ax.legend(
        handles,
        labels,
        loc="best",
    )

    for item, label in zip(leg.legendHandles, leg.texts):
        if label._text in ["PPFX Input", "PPFX Output"]:
            width = item.get_window_extent(fig.canvas.get_renderer()).width
            label.set_ha("left")
            label.set_position((-2 * width, 0))

    fig.tight_layout()

    return fig, ax
