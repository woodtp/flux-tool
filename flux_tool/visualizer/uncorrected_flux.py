import mplhep as hep
import numpy as np
from matplotlib.pyplot import Axes, Figure, subplots
from numpy import ndarray

from flux_tool.visualizer.vis_helpers import make_legend_no_errorbars
from flux_tool.visualizer.style import (neutrino_labels, neutrino_parent_labels, place_header,
                    xlabel_enu)


def plot_uncorrected_flux(
    ax: Axes,
    nu_flux: list[ndarray],
    bins: ndarray,
    fhc: bool = True,
    scale_factor=1,
    ylim=None,
    **kwargs,
) -> None:
    fake_errors = np.zeros(shape=nu_flux[0].shape)

    opts = dict(
        bins=bins,
        histtype="errorbar",
        yerr=fake_errors,
        markersize=14,
        markeredgewidth=2,
        markerfacecolor=[None, "none"],
        marker="o",
        label=[neutrino_labels["numu"], neutrino_labels["numubar"]],
    )

    opts.update(kwargs)

    H = list(map(lambda x: scale_factor * x, nu_flux))
    hep.histplot(
        H=H,
        ax=ax,
        **opts,
    )

    if fhc:
        header = "NuMI Simulation (Uncorrected, FHC)"
    else:
        header = "NuMI Simulation (Uncorrected, RHC)"

    if scale_factor == 1:
        ylabel = r"$\mathrm{\phi_\nu}$ [m$^{-2}$ POT$^{-1}$ GeV$^{-1}$]"
    else:
        power = int(np.log10(scale_factor))
        ylabel = f"$\\mathrm{{\\phi_\\nu}} \\times{{}} 10^{{{power}}}$ [m$^{{-2}}$ POT$^{{-1}}$ GeV$^{{-1}}$]"

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlim(0, 6)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel_enu)

    make_legend_no_errorbars(ax, fontsize=36)

    place_header(ax, header)


def plot_uncorrected_flux_logscale(
    numu_flux: ndarray,
    nue_flux: ndarray,
    numu_ratio: ndarray,
    nue_ratio: ndarray,
    fhc: bool = True,
) -> tuple[Figure, tuple[Axes, Axes]]:
    fig, (ax1, ax2) = subplots(
        2,
        1,
        sharex=True,
        height_ratios=[3, 1],
        gridspec_kw={"hspace": 0.025},
        # figsize=(12, 12),
        constrained_layout=True,
    )

    opts = dict(
        histtype="errorbar",
        elinewidth=0,
        markersize=14,
        binwnorm=True,
        markeredgewidth=2,
        markerfacecolor=[None, "none"],
    )

    hep.histplot(
        H=numu_flux,
        marker="o",
        label=[neutrino_labels["numu"], neutrino_labels["numubar"]],
        ax=ax1,
        **opts,
    )

    hep.histplot(
        H=nue_flux,
        label=[neutrino_labels["nue"], neutrino_labels["nuebar"]],
        marker="s",
        ax=ax1,
        **opts,
    )

    if fhc:
        header = "NuMI Simulation (Uncorrected, FHC)"
        ratio_ylabel = r"$\mathrm{\phi_{\nu}}$ / $\mathrm{\phi_{\bar{\nu}}}$"
        ratio_labels = [
            r"$\mathrm{\nu_{\mu}/\bar{\nu}_{\mu}}$",
            r"$\mathrm{\nu_{e}/\bar{\nu}_{e}}$",
        ]
    else:
        ratio_ylabel = r"$\mathrm{\phi_{\bar{\nu}}}$ / $\mathrm{\phi_{\nu}}$"
        header = "NuMI Simulation (Uncorrected, RHC)"
        ratio_labels = [
            r"$\mathrm{\bar{\nu}_{\mu}/\nu_{\mu}}$",
            r"$\mathrm{\bar{\nu}_{e}/\nu_{e}}$",
        ]

    hep.histplot(
        H=[numu_ratio, nue_ratio],
        label=ratio_labels,
        edges=False,
        linewidth=3,
        ax=ax2,
    )

    fontsize = 36

    ax1.set_xlim(0, 6)
    ax1.set_ylim(1e-11, 1e-4)
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$\mathrm{\phi_\nu}$ [m$^{-2}$ GeV$^{-1}$ POT$^{-1}$]", fontsize=fontsize)
    ax1.legend(loc="best", ncol=2)

    place_header(ax1, header)

    ax2.set_xlabel(xlabel_enu, fontsize=fontsize)
    ax2.set_ylabel(ratio_ylabel, fontsize=fontsize)

    ax2.legend(loc="upper right", fontsize=26)

    ax2.set_ylim(0, 2.75)
    ax2.axhline(y=1, ls="--", color="k", lw=2)

    return fig, (ax1, ax2)


def plot_neutrino_parents(
    ax: Axes,
    nominal: tuple[ndarray, ndarray],
    parents: dict[str, ndarray],
    nu: str,
    horn: str,
    scale_factor: int = 1,
    **kwargs,
) -> None:
    histogram_opts = dict(ax=ax, linewidth=2, yerr=False)
    histogram_opts.update(kwargs)

    nu_label = neutrino_labels[nu]

    to_nu = r"$\mathrm{\to}$" + nu_label

    parent_labels = [neutrino_parent_labels[key] + to_nu for key in parents]

    hep.histplot(H=nominal, label=f"Total {nu_label}", color="k", **histogram_opts)
    hep.histplot(H=list(parents.values()), label=parent_labels, **histogram_opts)

    ax.set_xlim(0, 6)
    ax.legend()

    place_header(ax, f"NuMI Simulation (Uncorrected, {horn})")

    ylabel = r"$\mathrm{\phi_{\nu}}$"

    if scale_factor != 1:
        ylabel += f" $\\times 10^{int(np.log10(scale_factor))}$"

    ylabel += r" [m$^{-2}$ POT$^{-1}$]"

    ax.set_ylabel(ylabel)

    ax.set_xlabel(xlabel_enu)
