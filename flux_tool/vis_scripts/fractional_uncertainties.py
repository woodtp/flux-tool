import sys
from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from flux_tool.vis_scripts.helper import absolute_uncertainty, save_figure
from flux_tool.vis_scripts.spectra_reader import SpectraReader
from flux_tool.vis_scripts.style import (icarus_preliminary, neutrino_labels,
                                         place_header, ppfx_labels, style,
                                         xlabel_enu)

def plot_hadron_fractional_uncertainties(
    reader: SpectraReader,
    output_dir: Optional[Path] = None,
    xlim: tuple[int | float, int | float] = (0, 20),
    ylim: tuple[int | float, int | float] = (0, 0.2),
) -> None:
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)

    plt.style.use(style)

    header = {"fhc": "Forward Horn Current", "rhc": "Reverse Horn Current"}

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

        n_spectra = len(hadron_uncerts)

        half = n_spectra // 2

        ls = (half) * ["-"] + (n_spectra - half) * ["--"]

        fig, ax = plt.subplots(figsize=(14, 14))

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
            H=list(sorted_hadron_uncerts.values()),
            yerr=False,
            histtype="step",
            edges=False,
            lw=3,
            ls=ls,
            label=hadron_labels,
        )

        ax.set_yticks(np.arange(0, 0.22, step=0.02))

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.legend(loc="upper center", fontsize=24, ncol=2)
        ax.set_xlabel(xlabel_enu)

        ax.set_ylabel(r"Fractional Uncertainty $\mathrm{\left( \sigma / \phi \right)}$")

        ax.tick_params(labelsize=28)

        place_header(ax, f"{header[horn]} {neutrino_labels[nu]}", x_pos=0.55)

        # icarus_preliminary(ax)

        if output_dir is not None:
            prefix = f"{horn}_{nu}"
            fig_name = f"{prefix}_hadron_fractional_uncertainties"
            tex_label = fig_name
            tex_caption = ""

            save_figure(fig, fig_name, output_dir, tex_caption, tex_label)

        plt.close(fig)


def plot_hadron_fractional_uncertainties_mesinc_breakout(
    reader: SpectraReader, output_dir: Optional[Path] = None
) -> None:
    plt.style.use(style)

    xaxis_lim = (0, 6)
    yaxis_lim = (0, 0.2)

    header = {"fhc": "Forward Horn Current", "rhc": "Reverse Horn Current"}
    version = {"projectile": "incoming", "daughter": "outgoing"}

    all_versions = product(
        product(["fhc", "rhc"], ["numu", "numubar", "nue", "nuebar"]),
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

        n_spectra = len(hadron_uncerts)

        half = n_spectra // 2

        ls = (half) * ["-"] + (n_spectra - half) * ["--"]

        fig, ax = plt.subplots(figsize=(14, 14))

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
            H=list(sorted_hadron_uncerts.values()),
            yerr=False,
            histtype="step",
            edges=False,
            lw=3,
            ls=ls,
            label=hadron_labels,
        )

        ax.set_yticks(np.arange(0, 0.22, step=0.02))

        ax.set_xlim(*xaxis_lim)
        ax.set_ylim(*yaxis_lim)
        ax.legend(loc="upper center", fontsize=24, ncol=2)
        ax.set_xlabel(xlabel_enu)

        ax.set_ylabel(r"Fractional Uncertainty $\mathrm{\left( \sigma / \phi \right)}$")

        ax.tick_params(labelsize=28)

        icarus_preliminary(ax)
        place_header(ax, f"{header[horn]} {neutrino_labels[nu]}", x_pos=0.55)

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
    reader: SpectraReader, output_dir: Optional[Path] = None
) -> None:
    plt.style.use(style)

    xaxis_lim = (0, 6)
    yaxis_lim = (0, 0.2)

    header = {"fhc": "Forward Horn Current", "rhc": "Reverse Horn Current"}
    version = {"projectile": "incoming", "daughter": "outgoing"}

    all_versions = product(
        product(["fhc", "rhc"], ["numu", "numubar", "nue", "nuebar"]),
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

        n_spectra = len(hadron_uncerts)

        half = n_spectra // 2

        ls = (half) * ["-"] + (n_spectra - half) * ["--"]

        fig, ax = plt.subplots(figsize=(14, 14))

        hep.histplot(
            ax=ax,
            H=list(sorted_hadron_uncerts.values()),
            yerr=False,
            histtype="step",
            edges=False,
            lw=3,
            ls=ls,
            label=hadron_labels,
        )

        ax.set_yticks(np.arange(0, 0.22, step=0.02))

        ax.set_xlim(*xaxis_lim)
        ax.set_ylim(*yaxis_lim)
        ax.legend(loc="upper center", fontsize=24, ncol=2)
        ax.set_xlabel(xlabel_enu)

        ax.set_ylabel(r"Fractional Uncertainty $\mathrm{\left( \sigma / \phi \right)}$")

        ax.tick_params(labelsize=28)

        place_header(ax, f"{header[horn]} {neutrino_labels[nu]}", x_pos=0.55)

        icarus_preliminary(ax)

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
