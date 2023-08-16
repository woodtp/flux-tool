import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

from flux_tool.config import AnalysisConfig
from flux_tool.exporter import Exporter
from flux_tool.flux_systematics_analysis import FluxSystematicsAnalysis
from flux_tool.preprocessor import Preprocessor
# from flux_tool.visualizer import Visualizer
from flux_tool.vis_scripts.pca_plots import plot_hadron_systs_and_pca_variances

try:
    import ROOT
except ImportError:
    raise ImportError(
        "PyROOT is not installed. Please install ROOT before using this package."
    )


def run(cfg_path: str):
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"The configuration file {cfg_path} was not found...")
    cfg = AnalysisConfig.from_file(cfg_path)

    if not cfg.sources_path.exists():
        raise FileNotFoundError(
            f"The directory {cfg.sources_path} does not exist. Exiting..."
        )
    elif not any(cfg.sources_path.iterdir()):
        raise FileNotFoundError(f"No files found in {cfg.sources_path}. Exiting...")

    if not cfg.results_path.exists():
        opt = input(f"{cfg.results_path} does not exist. Create it? (y/n) ").lower()
        if opt == "y":
            cfg.results_path.mkdir()
        else:
            print("Results directory not created. Exiting...")
            sys.exit()

    preprocessor = Preprocessor(cfg=cfg)

    analysis = FluxSystematicsAnalysis(
        nominal_flux_df=preprocessor.nominal_flux_df,
        ppfx_correction_df=preprocessor.ppfx_correction_df,
        bin_edges=cfg.bin_edges,
    )

    analysis.run(pca_threshold=1)

    exporter = Exporter(cfg, analysis)

    exporter.export_products()

    exporter.export_ppfx_output()

    with open(
        exporter.products_file.parent / "flux_covariance_binning_NuMI_GeV.txt", "w"
    ) as f:
        f.write(analysis.matrix_binning_str)

    logging.info("Beginning plot generation...")

    pca_plots_dir = cfg.plots_path / "pca"
    pca_plots_dir.mkdir(exist_ok=True)

    plot_hadron_systs_and_pca_variances(
        exporter.products_file, output_dir=pca_plots_dir
    )

    logging.info("Done.")

    # TODO
    # vis = Visualizer(config=cfg, analysis=analysis)
    #
    # vis.save_plots()


def main():
    from time import time

    start = time()

    parser = ArgumentParser(
        prog="flux_uncertainties",
        description="Interpret PPFX output into a neutrino flux prediction with uncertainties",
    )
    parser.add_argument(
        "-c", "--config", help="specify the path to a toml configuration file"
    )

    args = parser.parse_args()

    logging.basicConfig(format="%(message)s", level=logging.INFO)

    cfg = args.config

    if cfg is None:
        parser.print_help()
        sys.exit(1)

    run(args.config)
    end = time()

    print(f"Finished in {end-start:0.2f} s")


if __name__ == "__main__":
    main()
