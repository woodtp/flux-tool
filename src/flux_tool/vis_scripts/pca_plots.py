from functools import reduce
from itertools import product, repeat
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from flux_tool.vis_scripts.helper import absolute_uncertainty, save_figure
from flux_tool.vis_scripts.spectra_reader import SpectraReader
from flux_tool.vis_scripts.style import (neutrino_labels, place_header,
                                         ppfx_labels, xlabel_enu)


def scree_plot(reader: SpectraReader, output_dir: Optional[Path] = None):
    eigenvalues, _ = reader.pca_eigenvalues.to_numpy()  # type: ignore
    x = [i for i in range(1, len(eigenvalues) + 1)]

    cumul = np.cumsum(eigenvalues)

    fig, ax = plt.subplots()

    ax.set_box_aspect(1)

    ax.set_yscale("log")
    ax.set_xlabel("PCA Component Number")
    ax.set_ylabel("Fraction of Total Variance")
    # ax2 = ax.twinx()
    ax.scatter(x, eigenvalues, label="Eigenvalue")
    # ax2.scatter(x, cumul, color="C1")
    # ax2.set_ylabel("Cumulative Sum")
    idx = np.where(cumul <= 0.99)[0][-1]

    ax.axvline(
        idx,
        label=f"$\\mathrm{{\\lambda_{{{idx}}}}}$, 99% Explained Variance",
        c="k",
        ls="--",
    )

    ax.legend(loc="best", fontsize=26)
    if output_dir is not None:
        file_stem = "pca_scree_plot"
        tex_caption = ""
        tex_label = "scree_plot"

        save_figure(fig, file_stem, output_dir, tex_caption, tex_label)  # type: ignore

    plt.close(fig)


def pca_mesinc_overlay(reader: SpectraReader, output_dir: Optional[Path] = None):
    pcs = np.abs(
        np.hstack(
            [
                reader["pca/principal_components/hpc_1_fhc_nue"].to_numpy()[0],  # type: ignore
                reader["pca/principal_components/hpc_1_fhc_nuebar"].to_numpy()[0],  # type: ignore
                reader["pca/principal_components/hpc_1_fhc_numu"].to_numpy()[0],  # type: ignore
                reader["pca/principal_components/hpc_1_fhc_numubar"].to_numpy()[0],  # type: ignore
                reader["pca/principal_components/hpc_1_rhc_nue"].to_numpy()[0],  # type: ignore
                reader["pca/principal_components/hpc_1_rhc_nuebar"].to_numpy()[0],  # type: ignore
                reader["pca/principal_components/hpc_1_rhc_numu"].to_numpy()[0],  # type: ignore
                reader["pca/principal_components/hpc_1_rhc_numubar"].to_numpy()[0],  # type: ignore
            ]
        )
    )

    mesinc = np.sqrt(
        np.diag(
            reader.hadron_covariance_matrices["mesinc/hcov_mesinc"].to_numpy()[0]  # type: ignore
        )
    )

    bin_lengths = {k: len(b) for k, b in reader.binning.items()}

    x = [i for i in range(1, pcs.shape[0] + 1)]

    fig, ax = plt.subplots(figsize=(12, 12))

    ax.set_box_aspect(1)

    ax.plot(x, mesinc, label=ppfx_labels["mesinc"], lw=3)
    ax.plot(x, pcs, label="PC2 Overlay", ls="--", lw=3)

    ax.set_ylabel(r"Fractional Uncertainty ($\mathrm{\sigma / \phi}$)")
    ax.set_xlabel("Neutrino Flavor-Energy Bin No.")
    ax.set_ylim(0, 0.2)

    ax.legend(loc="upper left", fontsize=28)

    ax.axvline(59, c="gray", lw=1.5, ls="--", ymax=0.90)

    ax.text(57.5, 0.155, "FHC", fontweight="bold", horizontalalignment="right")
    ax.text(60.5, 0.155, "RHC", fontweight="bold", horizontalalignment="left")

    prev = 5
    for i, x in enumerate(repeat(bin_lengths.items(), 2)):
        for key, length in x:
            ax.annotate(
                neutrino_labels[key],
                [prev, 0.125],
                textcoords="offset points",
                xytext=[0, 0],
                va="bottom",
                ha="left",
            )
            prev += length - 1
            if i == 1 and key == "numubar":
                continue
            ax.axvline(prev - 4, c="gray", ls="--", lw=0.85, ymax=0.80)

    if output_dir is not None:
        file_stem = "pca_mesinc_overlay"
        tex_caption = ""
        tex_label = "mesinc_overlay"

        save_figure(fig, file_stem, output_dir, tex_caption, tex_label)  # type: ignore

    plt.close(fig)


def plot_top_principal_components(
    reader: SpectraReader, output_dir: Optional[Path] = None, n_components: int = 4
):
    pcs = np.vstack(
        [
            np.hstack(
                [
                    reader[f"pca/principal_components/hpc_{i}_fhc_nue"].to_numpy()[0],  # type: ignore
                    reader[f"pca/principal_components/hpc_{i}_fhc_nuebar"].to_numpy()[0],  # type: ignore
                    reader[f"pca/principal_components/hpc_{i}_fhc_numu"].to_numpy()[0],  # type: ignore
                    reader[f"pca/principal_components/hpc_{i}_fhc_numubar"].to_numpy()[0],  # type: ignore
                    reader[f"pca/principal_components/hpc_{i}_rhc_nue"].to_numpy()[0],  # type: ignore
                    reader[f"pca/principal_components/hpc_{i}_rhc_nuebar"].to_numpy()[0],  # type: ignore
                    reader[f"pca/principal_components/hpc_{i}_rhc_numu"].to_numpy()[0],  # type: ignore
                    reader[f"pca/principal_components/hpc_{i}_rhc_numubar"].to_numpy()[0],  # type: ignore
                ]
            )
            for i in range(n_components)
        ]
    )

    eigenvals, _ = reader["pca/heigenvals_frac"].to_numpy()  # type: ignore

    eigenvals = eigenvals[:n_components]

    x = [i for i in range(1, pcs.shape[1] + 1)]

    labels = [f"PC{i} ({ev:0.1%})" for i, ev in enumerate(eigenvals, start=1)]

    fig, ax = plt.subplots(figsize=(12, 12))

    ax.set_box_aspect(1)

    ax.axhline(0, lw=0.5, ls="--", c="k")

    ax.set_ylim(-0.25, 0.25)

    ax.plot(x, pcs.T, label=labels)

    ax.set_ylabel(r"Component Weight ($\mathrm{\sqrt{\lambda_k} \hat{u}_{k, i}}$)")
    ax.set_xlabel(r"Neutrino Flavor-Energy Bin No., $\mathrm{i}$")

    ax.legend(loc="upper center", ncol=2, columnspacing=0.8, fontsize=28)

    ax.axvline(59, c="gray", lw=1.5, ls="--", ymax=0.90)

    ax.text(57.5, 0.155, "FHC", fontweight="bold", horizontalalignment="right")
    ax.text(60.5, 0.155, "RHC", fontweight="bold", horizontalalignment="left")

    bin_lengths = {k: len(b) for k, b in reader.binning.items()}

    prev = 5
    for i, x in enumerate(repeat(bin_lengths.items(), 2)):
        for key, length in x:
            ax.annotate(
                neutrino_labels[key],
                [prev, 0.125],
                textcoords="offset points",
                xytext=[0, 0],
                va="bottom",
                ha="left",
            )
            prev += length - 1
            if i == 1 and key == "numubar":
                continue
            ax.axvline(prev - 4, c="gray", ls="--", lw=0.85, ymax=0.80)

    if output_dir is not None:
        file_stem = "pca_top_components"
        tex_caption = ""
        tex_label = "top_components"

        save_figure(fig, file_stem, output_dir, tex_caption, tex_label)  # type: ignore

    plt.close(fig)


def plot_hadron_systs_and_pca_variances(
    reader: SpectraReader,
    output_dir: Optional[Path] = None,
    xlim=(0.0, 20.0),
    ylim=(0.0, 0.05),
) -> None:
    print("[TODO] This is broken right now :(")
    return

    npcs = 7

    header = {"fhc": "Forward Horn Current", "rhc": "Reverse Horn Current"}

    version = {"projectile": "incoming", "daughter": "outgoing"}

    all_versions = product(
        reader.horns_and_nus,
        ["daughter", "projectile"],
    )

    all_versions_list = [(*v[0], v[1]) for v in all_versions]

    ppfx_correction = reader.ppfx_correction
    hadron_uncertainties = reader.hadron_uncertainties
    principal_components = reader.principal_components

    for horn, nu, ver in all_versions_list:
        flux = ppfx_correction[f"htotal_{horn}_{nu}"].to_pyroot()  # type: ignore

        total_uncert = hadron_uncertainties[
            f"total/hfrac_hadron_total_{horn}_{nu}"
        ].to_pyroot()  # type: ignore

        # print(hadron_uncertainties.keys())

        hadron_uncerts = {
            key.split("/")[0]: h.to_pyroot()
            for key, h in hadron_uncertainties.items()
            if key.endswith(nu)
            and horn in key
            and "total" not in key
            and version[ver] not in key
            # and "outgoing" not in key
            and "mesinc/" not in key
            and "oops" not in key.lower()
            and "Ascale" not in key
            and "inps" not in key.lower()
        }

        pcs = [
            principal_components[f"hpc_{x}_{horn}_{nu}"].to_pyroot()  # type: ignore
            for x in range(npcs)
        ]

        eigenvals, _ = reader["pca/heigenvals_frac"].to_numpy()  # type: ignore

        eigenvals = eigenvals[:npcs]

        absolute_uncertainties = {
            k: absolute_uncertainty(flux, h) for k, h in hadron_uncerts.items()
        }

        total_variance = total_uncert * total_uncert

        hadron_variances = {k: h * h for k, h in hadron_uncerts.items()}

        sorted_hadron_variances = {
            k: v
            for k, v in sorted(
                hadron_variances.items(),
                key=lambda kv: absolute_uncertainties[kv[0]].Integral(1, 11),
                reverse=True,
            )
        }

        total_variance_from_had_systs = reduce(
            lambda h1, h2: h1 + h2, hadron_variances.values()
        )

        hadron_labels = [ppfx_labels[k] for k in sorted_hadron_variances]

        pcs_variances = [pc * pc for pc in pcs]

        pc_labels = [f"PC{i} ({var*100:0.1f}%)" for i, var in enumerate(eigenvals, 1)]

        if ver == "daughter":
            actual = "projectile"
        else:
            actual = "daughter"

        fig, axs = plt.subplots(
            1,
            2,
            sharey=True,
            figsize=(20, 10),
            gridspec_kw={"wspace": 0.03}
            # 1, 2, sharey=True, figsize=(26, 14), gridspec_kw={"wspace": 0.03}
        )

        for ax in axs:
            ax.set_box_aspect(1)

        # hep.histplot(
        #     ax=axs[0],
        #     H=total_variance_from_had_systs,
        #     yerr=False,
        #     histtype="fill",
        #     linestyle="--",
        #     lw=3,
        #     color="w",
        #     edgecolor="gray",
        #     hatch="//",
        #     zorder=0,
        #     label=r"Total of all channels",
        #     edges=False,
        # )

        hep.histplot(
            ax=axs[0],
            H=total_variance,
            yerr=False,
            histtype="step",
            linestyle="--",
            lw=3,
            color="k",
            label="PPFX Total",
            zorder=10,
            edges=False,
        )

        hep.histplot(
            ax=axs[0],
            H=list(sorted_hadron_variances.values())[:npcs],
            yerr=False,
            histtype="fill",
            stack=True,
            edges=False,
            lw=3,
            label=hadron_labels[:npcs],
        )

        hep.histplot(
            ax=axs[1],
            H=total_variance,
            yerr=False,
            # histtype="fill",
            linestyle="--",
            lw=3,
            color="w",
            edgecolor="k",
            # hatch="//",
            zorder=0,
            label=r"PPFX Total",
            edges=False,
        )

        # hep.histplot(
        #     ax=axs[1],
        #     H=total_variance,
        #     yerr=False,
        #     histtype="step",
        #     linestyle="--",
        #     lw=3,
        #     color="k",
        #     zorder=10,
        #     edges=False,
        # )

        hep.histplot(
            ax=axs[1],
            H=pcs_variances,
            yerr=False,
            histtype="fill",
            stack=True,
            edges=False,
            lw=3,
            label=pc_labels,
        )

        for ax in axs:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.legend(loc="upper center", columnspacing=0.8, fontsize=22, ncol=2)
            ax.set_xlabel(xlabel_enu)

        axs[0].set_ylabel(
            r"Fractional Variance $\mathrm{\left( \sigma / \phi \right)^2}$"
        )

        place_header(
            axs[0],
            f"{header[horn]} {neutrino_labels[nu]}",
            (1.0, 1.0),
            ha="right",
        )

        # icarus_preliminary(axs[0])

        place_header(
            axs[1],
            r"$\mathrm{\sum \, \lambda_n =}$" + f" {eigenvals.sum()*100:0.1f}%",
            (1.0, 1.0),
            ha="right",
        )

        if output_dir is not None:
            prefix = f"{horn}_{nu}_{version[actual]}"

            file_stem = f"{prefix}_hadron_systs_and_pca_variances"

            tex_caption = ""
            tex_label = f"variance_{prefix}"

            save_figure(fig, file_stem, output_dir, tex_caption, tex_label)  # type: ignore

        plt.close(fig)


def plot_pca_systematic_shifts(
    reader: SpectraReader,
    output_dir: Optional[Path] = None,
    xlim: tuple[float, float] = (0, 20),
    ylim: tuple[float, float] = (-0.1, 0.1),
):
    shifts = reader.principal_components

    n_comps = 4

    for horn, nu in reader.horns_and_nus:
        comps: list[Any] = []
        labels: list[str] = []

        for n in range(n_comps):
            comps.append(shifts[f"hpc_{n}_{horn}_{nu}"])
            labels.append(f"PC{n+1}")

        fig, ax = plt.subplots()

        ax.set_box_aspect(1)

        hep.histplot(
            H=comps, label=labels, ax=ax, yerr=False, edges=False, lw=3, zorder=10
        )

        ax.axhline(0, ls="--", color="k", zorder=1)  # type: ignore

        ax.set_xlim(xlim)  # type: ignore
        ax.set_ylim(ylim)  # type: ignore
        ax.legend(loc="best", ncol=2)  # type: ignore
        ax.set_xlabel(xlabel_enu)  # type: ignore
        ax.set_ylabel(r"Principal Component Weight ($\mathrm{\sqrt{\lambda_n} \hat{e}_n}$)")  # type: ignore

        # icarus_preliminary(ax)  # type: ignore

        place_header(
            ax,  # type: ignore
            f"NuMI Simulation ({horn.upper()} {neutrino_labels[nu]})",
            xy=(1.0, 1.0),
            ha="right",
        )

        if output_dir is not None:
            prefix = f"{horn}_{nu}"
            fig_name = f"{prefix}_pca_systematic_shift"
            tex_label = fig_name
            tex_caption = ""

            save_figure(fig, fig_name, output_dir, tex_caption, tex_label)  # type: ignore

        plt.close(fig)
