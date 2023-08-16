from pathlib import Path

from flux_tool.vis_scripts.fractional_uncertainties import (
    plot_hadron_fractional_uncertainties,
    plot_hadron_fractional_uncertainties_mesinc_breakout,
    plot_hadron_fractional_uncertainties_mesinc_only)
from flux_tool.vis_scripts.pca_plots import plot_hadron_systs_and_pca_variances


def plot_all(products_file: Path | str, output_dir: Path):
    plot_hadron_fractional_uncertainties(
        products_file, output_dir / "hadron_uncertainties"
    )
    plot_hadron_fractional_uncertainties_mesinc_breakout(
        products_file, output_dir / "hadron_uncertainties/meson_breakout"
    )
    plot_hadron_fractional_uncertainties_mesinc_only(
        products_file, output_dir / "hadron_uncertainties/meson_only"
    )
    plot_hadron_systs_and_pca_variances(products_file, output_dir / "pca")
