from functools import partial
from typing import Optional

import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.pyplot import Axes, Figure
from numpy import ndarray

from flux_tool.visualizer.vis_helpers import create_ylabel_with_scale, get_hist_scale, scale_hist
from flux_tool.visualizer.style import neutrino_labels, place_header, xlabel_enu


def plot_ppfx_correction(
    nominal_flux: list[ndarray],
    flux_correction: list[ndarray],
    flux_uncertainty: list[ndarray],
    bins: ndarray,
    neutrinos: list[str],
    is_fhc: bool = True,
    ax: Optional[Axes] = None,
    draw_legend: bool = True,
    make_header: bool = True,
    xlim: tuple[float, float] = (0.0, 20.0),
) -> Axes | tuple[Figure, Axes]:
    fig = None
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)

    nu_labels = [neutrino_labels[nu] for nu in neutrinos]

    scale_factor = -1 * get_hist_scale(nominal_flux[0])

    ylabel = create_ylabel_with_scale(scale_factor)

    scale = partial(scale_hist, scale_factor=scale_factor)

    nominal_flux = list(map(scale, nominal_flux))
    flux_correction = list(map(scale, flux_correction))
    flux_uncertainty = list(map(scale, flux_uncertainty))

    nominal_labels = []
    correction_labels = []
    correction_suffix = r" PPFX Mean $\pm$ (stat. $\oplus$ syst.)"

    for label in nu_labels:
        nominal_labels.append(f"{label} Nominal")
        correction_labels.append(label + correction_suffix)

    correction_opts = {
            "color": None,
            "marker": "o",
            }

    if any(["nue" in nu for nu in neutrinos]):
        correction_opts.update({
            "color": ["C2", "C3"],
            "marker": "s"
        })

    hep.histplot(
        H=nominal_flux,
        bins=bins,
        binwnorm=True,
        histtype="step",
        label=nominal_labels,
        color=["k", "gray"],
        lw=2,
        ax=ax,
    )

    hep.histplot(
        H=flux_correction,
        bins=bins,
        yerr=flux_uncertainty,
        histtype="errorbar",
        label=correction_labels,
        binwnorm=True,
        elinewidth=3,
        capsize=4,
        markersize=14,
        markerfacecolor=[None, "none"],
        ax=ax,
        **correction_opts
    )

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel_enu)
    ax.set_xlim(*xlim)

    if make_header:
        header_str = "Forward Horn Current" if is_fhc else "Reverse Horn Current"
        place_header(ax, header_str)

    if draw_legend:
        ax.legend(loc="best", fontsize=18)

    return ax if fig is None else (fig, ax)


def plot_ppfx_correction_inset(
    nue_nominal: list[ndarray],
    numu_nominal: list[ndarray],
    nue_correction: list[ndarray],
    numu_correction: list[ndarray],
    nue_uncert: list[ndarray],
    numu_uncert: list[ndarray],
    bins: ndarray,
    is_fhc: bool = True,
    xlim: tuple[float, float] = (0.0, 20.0),
):
    fig, ax = plt.subplots(constrained_layout=True)

    ax_inset = ax.inset_axes([0.40, 0.25, 0.55, 0.55])
    ax_inset.set_xlim(*xlim)
    ax_inset.tick_params(labelsize=14)

    plot_ppfx_correction(
        nue_nominal,
        nue_correction,
        nue_uncert,
        bins,
        ["nue", "nuebar"],
        is_fhc=is_fhc,
        ax=ax,
    )
    plot_ppfx_correction(
        numu_nominal,
        numu_correction,
        numu_uncert,
        bins,
        ["numu", "numubar"],
        is_fhc=is_fhc,
        ax=ax_inset,
        make_header=False,
        # draw_legend=False,
    )

    ax_inset.set_ylabel(ax_inset.get_ylabel(), fontsize=18)
    ax_inset.set_xlabel(ax_inset.get_xlabel(), fontsize=18)

    return fig, (ax, ax_inset)
