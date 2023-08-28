from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from flux_tool.vis_scripts.helper import absolute_uncertainty, save_figure
from flux_tool.vis_scripts.spectra_reader import SpectraReader
from flux_tool.vis_scripts.style import (beam_syst_labels, icarus_preliminary,
                                         neutrino_labels, place_header,
                                         ppfx_labels, xlabel_enu)


def plot_hadron_fractional_uncertainties(
    reader: SpectraReader,
    output_dir: Optional[Path] = None,
    xlim: tuple[float, float] = (0, 20),
    ylim: tuple[float, float] = (0, 0.2),
) -> None:
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)

    ppfx_correction = reader.ppfx_correction
    hadron_uncertainties = reader.hadron_uncertainties

    for horn, nu in reader.horns_and_nus:
        flux = ppfx_correction[f"htotal_{horn}_{nu}"].to_pyroot()  # type: ignore

        total_uncert = hadron_uncertainties[
            f"total/hfrac_hadron_total_{horn}_{nu}"
        ].to_pyroot()  # type: ignore

        hadron_uncerts = {
            k.split("/")[0]: v.to_pyroot()
            for k, v in hadron_uncertainties.items()
            if k.endswith(nu)
            and horn in k
            and "total" not in k
            and "projectile" not in k
            and "daughter" not in k
        }

        absolute_uncertainties = {
            k: absolute_uncertainty(flux, h) for k, h in hadron_uncerts.items()
        }

        sorted_hadron_uncerts = {
            k: v
            for k, v in sorted(
                hadron_uncerts.items(),
                key=lambda kv: absolute_uncertainties[kv[0]].Integral(1, 11),
                reverse=True,
            )
        }

        hadron_labels = [ppfx_labels[k] for k in sorted_hadron_uncerts]

        fig = create_figure(
            list(sorted_hadron_uncerts.values()),
            horn,
            nu,
            hadron_labels,
            total_uncert,
            xlim,
            ylim,
        )

        if output_dir is not None:
            prefix = f"{horn}_{nu}"
            fig_name = f"{prefix}_hadron_fractional_uncertainties"
            tex_label = fig_name
            tex_caption = ""

            save_figure(fig, fig_name, output_dir, tex_caption, tex_label)

        plt.close(fig)


def plot_hadron_fractional_uncertainties_mesinc_breakout(
    reader: SpectraReader,
    output_dir: Optional[Path] = None,
    xlim: tuple[float, float] = (0, 6),
    ylim: tuple[float, float] = (0, 0.2),
) -> None:
    version = {"projectile": "incoming", "daughter": "outgoing"}

    all_versions = product(
        reader.horns_and_nus,
        ["daughter", "projectile"],
    )
    all_versions_list = [(*v[0], v[1]) for v in all_versions]

    ppfx_correction = reader.ppfx_correction
    hadron_uncertainties = reader.hadron_uncertainties

    for horn, nu, ver in all_versions_list:
        flux = ppfx_correction[f"htotal_{horn}_{nu}"].to_pyroot()  # type: ignore

        total_uncert = hadron_uncertainties[
            f"total/hfrac_hadron_total_{horn}_{nu}"
        ].to_pyroot()  # type: ignore

        hadron_uncerts = {
            k.split("/")[0]: v.to_pyroot()
            for k, v in hadron_uncertainties.items()
            if k.endswith(nu)
            and horn in k
            and "total" not in k
            and ver not in k
            and "mesinc/" not in k
        }

        if ver == "daughter":
            actual = "projectile"
        else:
            actual = "daughter"

        absolute_uncertainties = {
            k: absolute_uncertainty(flux, h) for k, h in hadron_uncerts.items()
        }

        sorted_hadron_uncerts = {
            k: v
            for k, v in sorted(
                hadron_uncerts.items(),
                key=lambda kv: absolute_uncertainties[kv[0]].Integral(1, 11),
                reverse=True,
            )
        }

        hadron_labels = [ppfx_labels[k] for k in sorted_hadron_uncerts]

        fig = create_figure(
            list(sorted_hadron_uncerts.values()),
            horn,
            nu,
            hadron_labels,
            total_uncert,
            xlim,
            ylim,
        )

        if output_dir is not None:
            ver = version[actual]

            out_dir = output_dir / ver
            if not out_dir.exists():
                out_dir.mkdir(parents=True)
            prefix = f"{horn}_{nu}"
            fig_name = f"{prefix}_{ver}_hadron_fractional_uncertainties"
            tex_label = fig_name
            tex_caption = ""

            save_figure(fig, fig_name, out_dir, tex_caption, tex_label)  # type: ignore

        plt.close(fig)


def plot_hadron_fractional_uncertainties_mesinc_only(
    reader: SpectraReader,
    output_dir: Optional[Path] = None,
    xlim: tuple[float, float] = (0, 6),
    ylim: tuple[float, float] = (0, 0.2),
) -> None:
    version = {"projectile": "incoming", "daughter": "outgoing"}

    all_versions = product(
        reader.horns_and_nus,
        ["daughter", "projectile"],
    )
    all_versions_list = [(*v[0], v[1]) for v in all_versions]

    ppfx_correction = reader.ppfx_correction
    hadron_uncertainties = reader.hadron_uncertainties

    for horn, nu, ver in all_versions_list:
        flux = ppfx_correction[f"htotal_{horn}_{nu}"].to_pyroot()  # type: ignore

        hadron_uncerts = {
            k.split("/")[0]: v.to_pyroot()
            for k, v in hadron_uncertainties.items()
            if k.endswith(nu)
            and horn in k
            and "total" not in k
            and ver not in k
            and "mesinc" in k
        }

        if ver == "daughter":
            actual = "projectile"
        else:
            actual = "daughter"

        absolute_uncertainties = {
            k: absolute_uncertainty(flux, h) for k, h in hadron_uncerts.items()
        }

        sorted_hadron_uncerts = {
            k: v
            for k, v in sorted(
                hadron_uncerts.items(),
                key=lambda kv: absolute_uncertainties[kv[0]].Integral(1, 11),
                reverse=True,
            )
        }

        hadron_labels = [ppfx_labels[k] for k in sorted_hadron_uncerts]

        fig = create_figure(
            uncerts=list(sorted_hadron_uncerts.values()),
            horn=horn,
            nu=nu,
            labels=hadron_labels,
            xlim=xlim,
            ylim=ylim,
        )

        if output_dir is not None:
            ver = version[actual]

            out_dir = output_dir / ver
            if not out_dir.exists():
                out_dir.mkdir(parents=True)
            prefix = f"{horn}_{nu}"
            fig_name = f"{prefix}_{ver}_hadron_fractional_uncertainties"
            tex_label = fig_name
            tex_caption = ""

            save_figure(fig, fig_name, out_dir, tex_caption, tex_label)  # type: ignore

        plt.close(fig)


def plot_beam_fractional_uncertainties(
    reader: SpectraReader,
    output_dir: Optional[Path] = None,
    xlim: tuple[float, float] = (0, 20),
    ylim: tuple[float, float] = (0, 0.2),
) -> None:
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)

    ppfx_correction = reader.ppfx_correction
    uncertainties = reader.beam_uncertainties

    for horn, nu in reader.horns_and_nus:
        flux = ppfx_correction[f"htotal_{horn}_{nu}"].to_pyroot()  # type: ignore

        total_uncert = uncertainties[
            f"total/hfrac_beam_total_{horn}_{nu}"
        ].to_pyroot()  # type: ignore

        uncerts = {
            k.split("/")[0]: v.to_pyroot()
            for k, v in uncertainties.items()
            if k.endswith(nu)
            and horn in k
            and "total" not in k
            and "beam_power" not in k
        }

        absolute_uncertainties = {
            k: absolute_uncertainty(flux, h) for k, h in uncerts.items()
        }

        sorted_uncerts = {
            k: v
            for k, v in sorted(
                uncerts.items(),
                key=lambda kv: absolute_uncertainties[kv[0]].Integral(1, 11),
                reverse=True,
            )
        }

        labels = [beam_syst_labels[k] for k in sorted_uncerts]

        fig = create_figure(
            list(sorted_uncerts.values()), horn, nu, labels, total_uncert, xlim, ylim
        )

        if output_dir is not None:
            prefix = f"{horn}_{nu}"
            fig_name = f"{prefix}_beam_fractional_uncertainties"
            tex_label = fig_name
            tex_caption = ""

            save_figure(fig, fig_name, output_dir, tex_caption, tex_label)

        plt.close(fig)


def create_figure(
    uncerts,
    horn,
    nu,
    labels,
    total_uncert=None,
    xlim: tuple[float, float] = (0, 20),
    ylim: tuple[float, float] = (0, 0.2),
):
    n_spectra = len(uncerts)

    half = n_spectra // 2

    ls = (half) * ["-"] + (n_spectra - half) * ["--"]

    fig, ax = plt.subplots(figsize=(14, 14), layout="constrained")

    if total_uncert is not None:
        hep.histplot(
            ax=ax,
            H=total_uncert,
            yerr=False,
            histtype="step",
            linestyle="-",
            lw=3,
            color="k",
            label="Total",
        )

    hep.histplot(
        ax=ax,
        H=uncerts,
        yerr=False,
        histtype="step",
        edges=False,
        lw=3,
        ls=ls,
        label=labels,
    )

    header = {"fhc": "Forward Horn Current", "rhc": "Reverse Horn Current"}

    ax.set_yticks(np.arange(0, 0.22, step=0.02))  # type: ignore
    ax.set_xlim(*xlim)  # type: ignore
    ax.set_ylim(*ylim)  # type: ignore
    ax.legend(loc="upper center", fontsize=24, ncol=2)  # type: ignore
    ax.set_xlabel(xlabel_enu)  # type: ignore
    ax.set_ylabel(r"Fractional Uncertainty $\mathrm{\left( \sigma / \phi \right)}$")  # type: ignore
    ax.tick_params(labelsize=28)  # type: ignore
    place_header(ax, f"{header[horn]} {neutrino_labels[nu]}", xy=(1.0, 1.0), ha="right")  # type: ignore
    icarus_preliminary(ax)  # type: ignore

    return fig


def plot_beam_systematic_shifts(
    reader: SpectraReader,
    output_dir: Optional[Path] = None,
    xlim: tuple[float, float] = (0, 20),
    # ylim: tuple[float, float] = (0, 0.1),
):
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)

    shifts = reader.beam_systematic_shifts

    for key, shift in shifts.items():
        if "beam_power" in key:
            continue
        tmp, horn, nu = key.rsplit("_", 2)
        _, _, syst = tmp.split("_", 2)

        label = beam_syst_labels[syst]

        stats, bins = reader[f"statistical_uncertainties/hstat_{horn}_{nu}"].to_numpy()  # type: ignore

        fig, ax = plt.subplots(layout="constrained", figsize=(11, 11))

        hep.histplot(
            H=shift, label=label, ax=ax, yerr=False, edges=False, lw=3, zorder=10
        )
        hep.histplot(
            H=stats,
            bins=bins,
            ax=ax,
            histtype="fill",
            label=r"$\mathrm{\sigma}_\mathsf{stat}$",
            color="C5",
            alpha=0.4,
            zorder=0,
        )
        hep.histplot(
            H=-1 * stats,
            bins=bins,
            ax=ax,
            histtype="fill",
            color="C5",
            alpha=0.4,
            zorder=0,
        )

        ax.axhline(0, ls="--", color="k", zorder=1)  # type: ignore

        ax.set_xlim(xlim)  # type: ignore
        ax.set_ylim(-0.1, 0.1)  # type: ignore
        ax.legend(loc="best")  # type: ignore
        ax.set_xlabel(xlabel_enu)  # type: ignore
        ax.set_ylabel(r"$\mathrm{\phi}_x - \mathrm{\phi}_\mathsf{nom}$ / $\mathrm{\phi}_\mathsf{nom}}$")  # type: ignore

        icarus_preliminary(ax)  # type: ignore

        place_header(
            ax,  # type: ignore
            f"NuMI Simulation ({horn.upper()} {neutrino_labels[nu]})",
            xy=(1.0, 1.0),
            ha="right",
        )

        if output_dir is not None:
            prefix = f"{horn}_{nu}"
            fig_name = f"{prefix}_{syst}_systematic_shift"
            tex_label = fig_name
            tex_caption = ""

            save_figure(fig, fig_name, output_dir, tex_caption, tex_label)  # type: ignore

        plt.close(fig)
