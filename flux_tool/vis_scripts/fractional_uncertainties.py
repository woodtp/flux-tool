from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from ROOT import TH1D  # type: ignore

from flux_tool.vis_scripts.helper import absolute_uncertainty, save_figure
from flux_tool.vis_scripts.spectra_reader import SpectraReader
from flux_tool.vis_scripts.style import (beam_syst_colors, beam_syst_labels,
                                         neutrino_labels, place_header,
                                         ppfx_colors, ppfx_labels,
                                         ppfx_mesinc_colors, xlabel_enu)


@dataclass
class PlotComponents:
    horn: str
    nu: str
    total_uncertainty: TH1D
    output_name: str
    xlim: tuple[float, float] = field(default_factory=tuple)
    ylim: tuple[float, float] = field(default_factory=tuple)
    uncertainties: list[TH1D] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    legend_kwargs: dict[str, Any] = field(
        default_factory=lambda: dict(
            loc="upper center", ncol=2, fontsize=20, columnspacing=0.8
        )
    )


def create_figure(comps: PlotComponents) -> Figure:
    n_spectra = len(comps.uncertainties)

    half = n_spectra // 2

    ls = (half) * ["-"] + (n_spectra - half) * ["--"] if n_spectra > 6 else "-"

    fig, ax = plt.subplots()  # , layout="constrained")

    ax.set_box_aspect(1)

    if comps.total_uncertainty is not None:
        hep.histplot(
            ax=ax,
            H=comps.total_uncertainty,
            yerr=False,
            histtype="step",
            linestyle="-",
            lw=3,
            color="k",
            label="Total",
            edges=False,
        )

    hep.histplot(
        ax=ax,
        H=comps.uncertainties,
        yerr=False,
        histtype="step",
        edges=False,
        color=comps.colors,
        lw=3,
        ls=ls,
        label=comps.labels,
    )

    header = {
        "fhc": f"FHC {neutrino_labels[comps.nu]}",
        "rhc": f"RHC {neutrino_labels[comps.nu]}",
    }

    ax.set_yticks(np.arange(0, 0.22, step=0.02))
    ax.set_xlim(*comps.xlim)
    ax.set_ylim(*comps.ylim)
    ax.legend(**comps.legend_kwargs)
    ax.set_xlabel(xlabel_enu)
    ax.set_ylabel(r"Fractional Uncertainty $\mathrm{\left( \sigma / \phi \right)}$")
    # ax.tick_params(labelsize=28)
    # place_header(ax, f"{header[horn]} {neutrino_labels[nu]}", xy=(1.0, 1.0), ha="right")
    hep.label.exp_label(llabel="", rlabel=header[comps.horn])
    # hep.label.exp_label(exp="DUNE", label="", ax=ax)
    # icarus_preliminary(ax)

    return fig


def plot_uncertainties(
    reader: SpectraReader,
    fn: Callable[[SpectraReader, tuple[float, float], tuple[float, float]], Iterator],
    output_dir: Optional[Path] = None,
    xlim: tuple[float, float] = (0, 20),
    ylim: tuple[float, float] = (0, 0.22),
    flux_overlay: bool = False,
):
    for comps in fn(reader, xlim, ylim):
        fig = create_figure(comps)

        # if flux_overlay:
        #     create_flux_overlay(reader, horn, nu)

        if output_dir is not None:
            fig_name = comps.output_name
            tex_label = fig_name
            tex_caption = ""

            save_figure(fig, fig_name, output_dir, tex_caption, tex_label)

        plt.close(fig)


def rebin_within_xlim(hist: TH1D, binning: NDArray, xlim: tuple[float, float]) -> TH1D:
    new_binning = binning[(binning >= xlim[0]) & (binning <= xlim[1])]
    hist = hist.Rebin(len(new_binning) - 1, hist.GetName(), new_binning)
    return hist


def plot_hadron_fractional_uncertainties(
    reader: SpectraReader,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> Iterator[PlotComponents]:
    ppfx_correction = reader.ppfx_correction
    hadron_uncertainties = reader.hadron_uncertainties

    for horn, nu in reader.horns_and_nus:
        # binning = reader.binning[nu]
        # new_binning = binning[(binning >= xlim[0]) & (binning <= xlim[1])]

        flux = ppfx_correction[f"htotal_{horn}_{nu}"].to_pyroot()
        flux = rebin_within_xlim(flux, reader.binning[nu], xlim)
        # flux = flux.Rebin(len(new_binning) - 1, flux.GetName(), new_binning)

        total_uncertainty = hadron_uncertainties[
            f"total/hfrac_hadron_total_{horn}_{nu}"
        ].to_pyroot()

        total_uncertainty = rebin_within_xlim(
            total_uncertainty, reader.binning[nu], xlim
        )

        uncerts = {
            k.split("/")[0]: v.to_pyroot()
            for k, v in hadron_uncertainties.items()
            if k.endswith(nu)
            and horn in k
            and "total" not in k
            and "projectile" not in k
            and "daughter" not in k
        }

        for k, h in uncerts.items():
            uncerts[k] = rebin_within_xlim(h, reader.binning[nu], xlim)

        sorted_uncerts = sort_uncertainties(uncerts, flux)

        hadron_labels = [ppfx_labels[k] for k in sorted_uncerts]

        colors = [ppfx_colors[k] for k in sorted_uncerts]

        yield PlotComponents(
            horn,
            nu,
            total_uncertainty,
            f"{horn}_{nu}_hadron_fractional_uncertainties",
            xlim,
            ylim,
            list(sorted_uncerts.values()),
            hadron_labels,
            colors,
        )


def plot_hadron_fractional_uncertainties_mesinc_breakout(
    reader: SpectraReader,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> Iterator[PlotComponents]:
    version = {"projectile": "incoming", "daughter": "outgoing"}

    all_versions = product(
        reader.horns_and_nus,
        ["daughter", "projectile"],
    )
    all_versions_list = [(*v[0], v[1]) for v in all_versions]

    ppfx_correction = reader.ppfx_correction
    hadron_uncertainties = reader.hadron_uncertainties

    for horn, nu, ver in all_versions_list:
        binning = reader.binning[nu]
        new_binning = binning[(binning >= xlim[0]) & (binning <= xlim[1])]

        flux = ppfx_correction[f"htotal_{horn}_{nu}"].to_pyroot()
        flux = flux.Rebin(len(new_binning) - 1, flux.GetName(), new_binning)

        total_uncertainty = hadron_uncertainties[
            f"total/hfrac_hadron_total_{horn}_{nu}"
        ].to_pyroot()

        total_uncertainty = total_uncertainty.Rebin(
            len(new_binning) - 1, total_uncertainty.GetName(), new_binning
        )

        uncerts = {
            k.split("/")[0]: v.to_pyroot()
            for k, v in hadron_uncertainties.items()
            if k.endswith(nu)
            and horn in k
            and "total" not in k
            and ver in k
            and "mesinc/" not in k
        }

        for k, h in uncerts.items():
            uncerts[k] = h.Rebin(len(new_binning) - 1, h.GetName(), new_binning)

        sorted_uncerts = sort_uncertainties(uncerts, flux)

        hadron_labels = [ppfx_labels[k] for k in sorted_uncerts]

        colors = [ppfx_mesinc_colors[k] for k in sorted_uncerts]

        yield PlotComponents(
            horn,
            nu,
            total_uncertainty,
            f"{version[ver]}/{horn}_{nu}_{version[ver]}_hadron_fractional_uncertainties",
            xlim,
            ylim,
            list(sorted_uncerts.values()),
            hadron_labels,
            colors,
        )


# def plot_hadron_fractional_uncertainties_mesinc_only(
#     reader: SpectraReader,
#     output_dir: Optional[Path] = None,
#     xlim: tuple[float, float] = (0, 6),
#     ylim: tuple[float, float] = (0, 0.2),
# ) -> None:
#     version = {"projectile": "incoming", "daughter": "outgoing"}
#
#     all_versions = product(
#         reader.horns_and_nus,
#         ["daughter", "projectile"],
#     )
#     all_versions_list = [(*v[0], v[1]) for v in all_versions]
#
#     ppfx_correction = reader.ppfx_correction
#     hadron_uncertainties = reader.hadron_uncertainties
#
#     for horn, nu, ver in all_versions_list:
#         flux = ppfx_correction[f"htotal_{horn}_{nu}"].to_pyroot()
#
#         uncerts = {
#             k.split("/")[0]: v.to_pyroot()
#             for k, v in hadron_uncertainties.items()
#             if k.endswith(nu)
#             and horn in k
#             and "total" not in k
#             and ver not in k
#             and "mesinc" in k
#         }
#
#         if ver == "daughter":
#             actual = "projectile"
#         else:
#             actual = "daughter"
#
#         # absolute_uncertainties = {
#         #     k: absolute_uncertainty(flux, h) for k, h in hadron_uncerts.items()
#         # }
#         #
#         # sorted_hadron_uncerts = {
#         #     k: v
#         #     for k, v in sorted(
#         #         hadron_uncerts.items(),
#         #         key=lambda kv: absolute_uncertainties[kv[0]].Integral(1, 11),
#         #         reverse=True,
#         #     )
#         # }
#
#         sorted_uncerts = sort_uncertainties(uncerts, flux)
#
#         hadron_labels = [ppfx_labels[k] for k in sorted_uncerts]
#
#         # fig = create_figure(
#         #     uncerts=list(sorted_uncerts.values()),
#         #     horn=horn,
#         #     nu=nu,
#         #     labels=hadron_labels,
#         #     xlim=xlim,
#         #     ylim=ylim,
#         # )
#
#         if output_dir is not None:
#             ver = version[actual]
#
#             out_dir = output_dir / ver
#             prefix = f"{horn}_{nu}"
#             fig_name = f"{prefix}_{ver}_hadron_fractional_uncertainties"
#             tex_label = fig_name
#             tex_caption = ""
#
#             save_figure(fig, fig_name, out_dir, tex_caption, tex_label)
#
#         plt.close(fig)


def plot_beam_fractional_uncertainties(
    reader: SpectraReader,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> Iterator[PlotComponents]:
    ppfx_correction = reader.ppfx_correction
    uncertainties = reader.beam_uncertainties

    for horn, nu in reader.horns_and_nus:
        flux = ppfx_correction[f"htotal_{horn}_{nu}"].to_pyroot()
        flux = rebin_within_xlim(flux, reader.binning[nu], xlim)

        total_uncertainty = uncertainties[
            f"total/hfrac_beam_total_{horn}_{nu}"
        ].to_pyroot()

        total_uncertainty = rebin_within_xlim(
            total_uncertainty, reader.binning[nu], xlim
        )

        uncerts = {
            k.split("/")[0]: v.to_pyroot()
            for k, v in uncertainties.items()
            if k.endswith(nu)
            and horn in k
            and "total" not in k
            and "beam_power" not in k
        }
        for k, h in uncerts.items():
            uncerts[k] = rebin_within_xlim(h, reader.binning[nu], xlim)

        sorted_uncerts = sort_uncertainties(uncerts, flux)

        labels = [beam_syst_labels[k] for k in sorted_uncerts]

        colors = [beam_syst_colors[k] for k in sorted_uncerts]

        comps = PlotComponents(
            horn,
            nu,
            total_uncertainty,
            f"{horn}_{nu}_beam_fractional_uncertainties",
            xlim,
            ylim,
            list(sorted_uncerts.values()),
            labels,
            colors,
        )

        comps.legend_kwargs["fontsize"] = 18

        yield comps


def create_flux_overlay(reader: SpectraReader, horn: str, nu: str) -> None:
    flux = reader.ppfx_correction[f"htotal_{horn}_{nu}"].to_pyroot()
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    max = flux.GetMaximum()

    h = hep.histplot(
        flux,
        ax=ax2,
        histtype="fill",
        yerr=False,
        label=("Flux Shape (A.U.)"),
        ls="-.",
        color="gray",
        edgecolor="k",
        lw=3,
        zorder=0,
        alpha=0.3,
    )

    handles, labels = ax1.get_legend_handles_labels()

    handles += [h[0][0]]
    labels += ["Flux Shape (A.U.)"]

    ax1.legend(handles, labels, ncol=2, loc="upper center", fontsize=26)

    ax2.set_ylim(0, 1.4 * max)

    ax2.get_yaxis().set_visible(False)


def plot_beam_systematic_shifts(
    reader: SpectraReader,
    output_dir: Optional[Path] = None,
    xlim: tuple[float, float] = (0, 20),
    # ylim: tuple[float, float] = (0, 0.1),
) -> None:
    shifts = reader.beam_systematic_shifts

    for key, shift in shifts.items():
        if "beam_power" in key:
            continue
        tmp, horn, nu = key.rsplit("_", 2)
        _, _, syst = tmp.split("_", 2)

        label = beam_syst_labels[syst]
        color = beam_syst_colors[syst]

        stats, bins = reader[f"statistical_uncertainties/hstat_{horn}_{nu}"].to_numpy()  # type: ignore

        fig, ax = plt.subplots(layout="constrained", figsize=(11, 11))

        hep.histplot(
            H=shift,
            label=label,
            ax=ax,
            yerr=False,
            edges=False,
            lw=3,
            zorder=10,
            color=color,
        )
        hep.histplot(
            H=stats,
            bins=bins,
            ax=ax,
            histtype="fill",
            label=r"$\mathrm{\sigma}_\mathsf{stat}$",
            color="C0",
            alpha=0.4,
            zorder=0,
        )
        hep.histplot(
            H=-1 * stats,
            bins=bins,
            ax=ax,
            histtype="fill",
            color="C0",
            alpha=0.4,
            zorder=0,
        )

        ax.axhline(0, ls="--", color="k", zorder=1)

        ax.set_xlim(xlim)
        ax.set_ylim(-0.1, 0.1)
        ax.legend(loc="best")
        ax.set_xlabel(xlabel_enu)
        ax.set_ylabel(
            r"$\mathrm{\phi}_x - \mathrm{\phi}_\mathsf{nom}$ / $\mathrm{\phi}_\mathsf{nom}$"
        )

        hep.label.exp_label(
            exp="", llabel="", rlabel=f"{horn.upper()} {neutrino_labels[nu]}"
        )

        if output_dir is not None:
            prefix = f"{horn}_{nu}"
            fig_name = f"{prefix}_{syst}_systematic_shift"
            tex_label = fig_name
            tex_caption = ""

            save_figure(fig, fig_name, output_dir, tex_caption, tex_label)

        plt.close(fig)


def sort_uncertainties(uncertainties: dict[str, TH1D], flux: TH1D) -> dict[str, TH1D]:
    absolute_uncertainties = {
        k: absolute_uncertainty(flux, h).Integral(1, 11)
        for k, h in uncertainties.items()
    }

    sorted_uncertainties = {
        k: v
        for k, v in sorted(
            uncertainties.items(),
            key=lambda kv: absolute_uncertainties[kv[0]],
            reverse=True,
        )
    }

    return sorted_uncertainties
