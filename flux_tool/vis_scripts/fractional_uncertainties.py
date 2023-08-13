from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from flux_tool.vis_scripts.helper import absolute_uncertainty
from flux_tool.vis_scripts.style import (icarus_preliminary, neutrino_labels,
                                         place_header, ppfx_labels, style,
                                         xlabel_enu)


def plot_hadron_fractional_uncertainties(
    products_file: Path | str, output_dir: str = "plots/had_systs"
) -> None:
    plt.style.use(style)

    xaxis_lim = (0, 6)
    yaxis_lim = (0, 0.2)

    header = {"fhc": "Forward Horn Current", "rhc": "Reverse Horn Current"}

    all_versions = list(product(["fhc", "rhc"], ["numu", "numubar", "nue", "nuebar"]))

    with uproot.open(products_file) as f:
        for horn, nu in all_versions:
            flux = f[f"ppfx_corrected_flux/total/htotal_{horn}_{nu}"].to_pyroot()

            total_uncert = f[
                f"fractional_uncertainties/hadron/total/hfrac_hadron_total_{horn}_{nu}"
            ].to_pyroot()

            hadron_uncerts = {
                key.split("/")[0]: h.to_pyroot()
                for key, h in f["fractional_uncertainties/hadron/"].items(
                    filter_name=f"*{horn}*{nu}", cycle=False
                )
                if "total" not in key
                and "projectile" not in key
                and "daughter" not in key
                and "qel" not in key.lower()
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

            _, ax = plt.subplots(figsize=(14, 14))

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

            ax.set_ylabel(
                r"Fractional Uncertainty $\mathrm{\left( \sigma / \phi \right)}$"
            )

            ax.tick_params(labelsize=28)

            place_header(ax, f"{header[horn]} {neutrino_labels[nu]}")

            icarus_preliminary(ax)

            for ext in ("pdf", "png"):
                plt.savefig(
                    f"{output_dir}/{horn}_{nu}_hadron_fractional_uncertainties.{ext}"
                )


def plot_hadron_fractional_uncertainties_mesinc_breakout(
    products_file: Path | str, output_dir: str = "plots/had_systs"
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
    with uproot.open(products_file) as f:
        for horn, nu, ver in all_versions_list:
            flux = f[f"ppfx_corrected_flux/total/htotal_{horn}_{nu}"].to_pyroot()

            total_uncert = f[
                f"fractional_uncertainties/hadron/total/hfrac_hadron_total_{horn}_{nu}"
            ].to_pyroot()

            hadron_uncerts = {
                key.split("/")[0]: h.to_pyroot()
                for key, h in f["fractional_uncertainties/hadron/"].items(
                    filter_name=f"*{horn}*{nu}", cycle=False
                )
                if "total" not in key
                and ver not in key
                and "mesinc/" not in key
                and "qel" not in key.lower()
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

            _, ax = plt.subplots(figsize=(14, 14))

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

            ax.set_ylabel(
                r"Fractional Uncertainty $\mathrm{\left( \sigma / \phi \right)}$"
            )

            ax.tick_params(labelsize=28)

            place_header(ax, f"{header[horn]} {neutrino_labels[nu]}")
            icarus_preliminary(ax)

            for ext in ("pdf", "png"):
                plt.savefig(
                    f"{output_dir}/{version[actual]}_meson/{horn}_{nu}_{version[actual]}_hadron_fractional_uncertainties.{ext}"
                )


def plot_hadron_fractional_uncertainties_mesinc_only(
    products_file: Path | str, output_dir: str = "plots/had_systs"
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
    with uproot.open(products_file) as f:
        for horn, nu, ver in all_versions_list:
            flux = f[f"ppfx_corrected_flux/total/htotal_{horn}_{nu}"].to_pyroot()

            # total_uncert = f[
            #     f"fractional_uncertainties/hadron/total/hfrac_hadron_total_{horn}_{nu}"
            # ].to_pyroot()

            hadron_uncerts = {
                key.split("/")[0]: h.to_pyroot()
                for key, h in f["fractional_uncertainties/hadron/"].items(
                    filter_name=f"*{horn}*{nu}", cycle=False
                )
                if "total" not in key and ver not in key and "mesinc" in key
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

            _, ax = plt.subplots(figsize=(14, 14))

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

            ax.set_ylabel(
                r"Fractional Uncertainty $\mathrm{\left( \sigma / \phi \right)}$"
            )

            ax.tick_params(labelsize=28)

            place_header(ax, f"{header[horn]} {neutrino_labels[nu]}")
            icarus_preliminary(ax)

            for ext in ("pdf", "png"):
                plt.savefig(
                    f"{output_dir}/{version[actual]}_meson_only/{horn}_{nu}_{version[actual]}_hadron_fractional_uncertainties.{ext}"
                )
