import logging
import tarfile
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mplhep as hep
from numpy.typing import NDArray

import flux_tool.vis_scripts as vis
from flux_tool.vis_scripts.spectra_reader import SpectraReader
from flux_tool.vis_scripts.style import style


def plot_all(
    products_file: Path | str,
    output_dir: Path,
    plot_opts: dict[str, Any],
    binning: dict[str, NDArray],
):
    plt.style.use(style)

    reader = SpectraReader(products_file, binning)

    if plot_opts["draw_label"]:
        label_drawer = partial(
            hep.label.exp_label,
            exp=plot_opts["experiment"],
            llabel=plot_opts["stage"],
            rlabel="",
        )
    else:
        label_drawer = None

    xlim: tuple[float, float] = plot_opts["xlim"]

    enabled_plots = plot_opts["enabled"]

    if all(enabled_plots.values()):
        logging.info(f"Pre-caching ROOT objects from {products_file}...")
        reader.load_cache()
        logging.info("Done.")

    if enabled_plots["uncorrected_flux"]:
        logging.info("Plotting uncorrected flux...")
        vis.plot_flux_uncorrected_logarithmic(
            reader, output_dir / "flux_spectra/uncorrected_flux", xlim
        )

    if enabled_plots["flux_prediction"]:
        logging.info("Plotting flux prediction...")
        vis.plot_flux_prediction(
            reader,
            output_dir / "flux_spectra/flux_prediction",
            xlim,
            label_drawer,
            bullets=plot_opts["flux_prediction_bullets"],
        )

    if enabled_plots["flux_prediction_parent_spectra"]:
        logging.info("Plotting parent spectra...")
        vis.plot_parents(reader, output_dir / "flux_spectra/parents", xlim)

    if enabled_plots["flux_prediction_parent_spectra_stacked"]:
        logging.info("Plotting parent spectra (stacked)...")
        vis.plot_parents(
            reader,
            output_dir / "flux_spectra/parents",
            xlim,
            stacked=True,
        )

    if enabled_plots["ppfx_universes"]:
        logging.info("Plotting PPFX universes...")
        vis.plot_ppfx_universes(reader, output_dir / "flux_spectra/universes", xlim)

    if enabled_plots["hadron_uncertainties"]:
        logging.info("Plotting hadron uncertainties...")
        vis.plot_uncertainties(
            reader,
            vis.plot_hadron_fractional_uncertainties,
            output_dir / "hadron_uncertainties",
            xlim,
            label_drawer=label_drawer,
        )

    if enabled_plots["hadron_uncertainties_meson"]:
        logging.info("Plotting hadron uncertainties (mesinc)...")
        vis.plot_uncertainties(
            reader,
            vis.plot_hadron_fractional_uncertainties_mesinc_breakout,
            output_dir / "hadron_uncertainties/meson_breakout",
            xlim,
        )
    if enabled_plots["hadron_uncertainties_meson_only"]:
        logging.info("Plotting hadron uncertainties (mesinc only)...")
        logging.warning("Deprecated: Use hadron_uncertainties_meson instead.")
        # plot_hadron_fractional_uncertainties_mesinc_only(
        #     reader, output_dir / "hadron_uncertainties/meson_only", xlim
        # )

    if enabled_plots["hadron_uncertainties_nua"]:
        logging.info("Plotting hadron uncertainties (nua bands only)...")
        vis.plot_uncertainties(
            reader,
            vis.plot_hadron_fractional_uncertainties_nua_breakout,
            output_dir / "hadron_uncertainties",
            xlim,
        )

    if enabled_plots["pca_scree_plot"]:
        logging.info("Plotting Eigenvalues Scree Plot")
        vis.scree_plot(reader, output_dir / "pca")

    if enabled_plots["pca_mesinc_overlay"]:
        logging.info("Plotting PCA Uncertainty Comparison")
        vis.pca_mesinc_overlay(reader, output_dir / "pca")

    if enabled_plots["pca_top_components"]:
        logging.info("Plotting top principal components")
        vis.plot_top_principal_components(reader, output_dir / "pca")

    if enabled_plots["pca_variances"]:
        logging.info("Plotting PCA variances...")
        vis.plot_hadron_systs_and_pca_variances(
            reader, output_dir / "pca/variances", xlim
        )

    if enabled_plots["pca_components"]:
        logging.info("Plotting princpal components...")
        vis.plot_pca_systematic_shifts(
            reader, output_dir / "pca/components", xlim, (-0.12, 0.12)
        )

    if enabled_plots["beam_uncertainties"]:
        logging.info("Plotting beam uncertainties...")
        vis.plot_uncertainties(
            reader,
            vis.plot_beam_fractional_uncertainties,
            output_dir / "beam_uncertainties",
            xlim,
            (0, 0.12),
        )

    if status := enabled_plots["hadron_covariance_matrices"]:
        logging.info("Plotting hadron covariance matrices...")
        vis.plot_hadron_covariance_matrices(
            reader,
            output_dir / "covariance_matrices/hadron",
            which=status if isinstance(status, str) else None,
        )

    if status := enabled_plots["beam_covariance_matrices"]:
        logging.info("Plotting beamline focusing covariance matrices...")
        vis.plot_beam_covariance_matrices(
            reader,
            output_dir / "covariance_matrices/beam",
            which=status if isinstance(status, str) else None,
        )

    if status := enabled_plots["hadron_correlation_matrices"]:
        logging.info("Plotting hadron correlation matrices...")
        vis.plot_hadron_correlation_matrices(
            reader,
            output_dir / "covariance_matrices/hadron",
            which=status if isinstance(status, str) else None,
        )

    if status := enabled_plots["beam_correlation_matrices"]:
        logging.info("Plotting beamline focusing correlation matrices...")
        vis.plot_beam_correlation_matrices(
            reader,
            output_dir / "covariance_matrices/beam",
            which=status if isinstance(status, str) else None,
        )

    if enabled_plots["beam_systematic_shifts"]:
        logging.info("Plotting beamline systematic shifts...")
        vis.plot_beam_systematic_shifts(
            reader, output_dir / "beam_systematic_shifts", xlim
        )


def compress_directory(directory: Path):
    logging.info(f"Compressing {directory}...")
    with tarfile.open("plots.tar.xz", "w:xz") as tar:
        for d in directory.iterdir():
            tar.add(d, arcname=d.stem)
