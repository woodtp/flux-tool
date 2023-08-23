import logging
import tarfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from flux_tool.vis_scripts.covariance import (plot_beam_correlation_matrices,
                                              plot_hadron_correlation_matrices)
from flux_tool.vis_scripts.flux_prediction import (
    plot_flux_prediction, plot_flux_uncorrected_logarithmic)
from flux_tool.vis_scripts.fractional_uncertainties import (
    plot_beam_fractional_uncertainties, plot_beam_systematic_shifts,
    plot_hadron_fractional_uncertainties,
    plot_hadron_fractional_uncertainties_mesinc_breakout,
    plot_hadron_fractional_uncertainties_mesinc_only)
from flux_tool.vis_scripts.parent_spectra import plot_parents
from flux_tool.vis_scripts.pca_plots import plot_hadron_systs_and_pca_variances
from flux_tool.vis_scripts.ppfx_universes import plot_ppfx_universes
from flux_tool.vis_scripts.spectra_reader import SpectraReader
from flux_tool.vis_scripts.style import style


def plot_all(
    products_file: Path | str,
    output_dir: Path,
    plot_opts: dict[str, Any],
):
    plt.style.use(style)

    reader = SpectraReader(products_file)

    logging.info(f"Reading ROOT objects from {products_file}...")
    reader.load_cache()
    logging.info("Done.")

    xlim: tuple[float, float] = plot_opts["xlim"]

    logging.info("Plotting uncorrected flux...")
    plot_flux_uncorrected_logarithmic(
        reader, output_dir / "flux_spectra/uncorrected_flux", xlim
    )

    logging.info("Plotting flux prediction...")
    plot_flux_prediction(reader, output_dir / "flux_spectra/flux_prediction", xlim)

    logging.info("Plotting parent spectra...")
    plot_parents(reader, output_dir / "flux_spectra/parents", xlim)

    logging.info("Plotting parent spectra (stacked)...")
    plot_parents(reader, output_dir / "flux_spectra/parents", xlim, stacked=True)

    logging.info("Plotting PPFX universes...")
    plot_ppfx_universes(reader, output_dir / "flux_spectra/universes")

    logging.info("Plotting hadron uncertainties...")
    plot_hadron_fractional_uncertainties(
        reader, output_dir / "hadron_uncertainties", xlim, (0, 0.20)
    )

    logging.info("Plotting hadron uncertainties (mesinc)...")
    plot_hadron_fractional_uncertainties_mesinc_breakout(
        reader, output_dir / "hadron_uncertainties/meson_breakout", xlim
    )
    plot_hadron_fractional_uncertainties_mesinc_only(
        reader, output_dir / "hadron_uncertainties/meson_only", xlim
    )

    logging.info("Plotting PCA variances...")
    plot_hadron_systs_and_pca_variances(reader, output_dir / "pca", xlim)

    logging.info("Plotting beam uncertainties...")
    plot_beam_fractional_uncertainties(
        reader, output_dir / "beam_uncertainties", xlim, (0, 0.18)
    )

    logging.info("Plotting hadron correlation matrices...")
    plot_hadron_correlation_matrices(reader, output_dir / "covariance_matrices/hadron")

    logging.info("Plotting beamline focusing correlation matrices...")
    plot_beam_correlation_matrices(reader, output_dir / "covariance_matrices/beam")

    logging.info("Plotting beamline systematic shifts...")
    plot_beam_systematic_shifts(reader, output_dir / "beam_systematic_shifts", xlim)


def compress_directory(directory: Path):
    logging.info(f"Compressing {directory}...")
    with tarfile.open("plots.tar.xz", "w:xz") as tar:
        for d in directory.iterdir():
            tar.add(d, arcname=d.stem)
