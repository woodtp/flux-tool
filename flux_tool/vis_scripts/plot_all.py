import logging
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt

from flux_tool.vis_scripts.covariance import plot_hadron_correlation_matrices
from flux_tool.vis_scripts.flux_prediction import plot_flux_prediction
from flux_tool.vis_scripts.fractional_uncertainties import (
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
    xlim: tuple[int, int] = (0, 20),
):
    plt.style.use(style)

    reader = SpectraReader(products_file)

    # reader.load_cache()

    jobs = (
        (
            plot_flux_prediction,
            (reader, output_dir / "flux_spectra/flux_prediction", xlim),
        ),
        # (plot_parents, (reader, output_dir / "flux_spectra/parents")),
        # (
        #     plot_parents,
        #     (reader, output_dir / "flux_spectra/parents", True),
        # ),
        # (plot_ppfx_universes, (reader, output_dir / "flux_spectra/universes")),
        # (
        #     plot_hadron_fractional_uncertainties,
        #     (reader, output_dir / "hadron_uncertainties", (0, 10), (0, 0.15)),
        # ),
        # (
        #     plot_hadron_correlation_matrices,
        #     (reader, output_dir / "covariance_matrices/hadron"),
        # ),
        # (
        #     plot_hadron_fractional_uncertainties_mesinc_breakout,
        #     (reader, output_dir / "hadron_uncertainties/meson_breakout"),
        # ),
        # (
        #     plot_hadron_fractional_uncertainties_mesinc_only,
        #     (reader, output_dir / "hadron_uncertainties/meson_only"),
        # ),
        # (plot_hadron_systs_and_pca_variances, (reader, output_dir / "pca")),
    )

    for fn, args in jobs:
        fn(*args)  # type: ignore


def compress_directory(directory: Path):
    logging.info(f"Compressing {directory}...")
    with tarfile.open("plots.tar.xz", "w:xz") as tar:
        for d in directory.iterdir():
            tar.add(d, arcname=d.stem)
