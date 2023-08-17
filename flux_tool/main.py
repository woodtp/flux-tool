import logging
import sys
from argparse import ArgumentParser
from importlib.util import find_spec

from flux_tool.config import AnalysisConfig
from flux_tool.exporter import Exporter
from flux_tool.flux_systematics_analysis import FluxSystematicsAnalysis
from flux_tool.preprocessor import Preprocessor
from flux_tool.vis_scripts.plot_all import plot_all


def check_for_ROOT() -> None:
    try:
        spec = find_spec("ROOT")
    except ValueError:
        """For some reason ROOT doesn't have a defined __spec__,
        and a ValueError exception is thrown.
        But it's still importable so we can catch the exception here.
        """
        pass
    else:
        if not spec:
            msg = (
                "PyROOT is not installed."
                "Please install ROOT before using this package."
            )
            raise ImportError(msg)


def run(cfg_path: str):
    check_for_ROOT()

    try:
        cfg = AnalysisConfig.from_file(cfg_path)
    except FileNotFoundError:
        print(f'The configuration file "{cfg_path}" was not found. Exiting...')
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

    plot_all(exporter.products_file, cfg.plots_path)

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
