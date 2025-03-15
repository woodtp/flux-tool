import logging
import signal
import sys
from argparse import ArgumentParser
from importlib.util import find_spec
from time import time

from flux_tool.config import AnalysisConfig
from flux_tool.exporter import Exporter
from flux_tool.flux_systematics_analysis import FluxSystematicsAnalysis
from flux_tool.preprocessor import Preprocessor
from flux_tool.vis_scripts.plot_all import compress_directory, plot_all


def timer(fn):
    def wrap():
        start = time()
        fn()
        end = time()
        logging.info(f"Finished in {end-start:0.2f} s")

    return wrap


def signal_handler(sig, frame):
    print("Interrupted. Exiting...")
    exit(0)


signal.signal(signal.SIGINT, signal_handler)


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


def load_config(cfg_path: str) -> AnalysisConfig:
    try:
        cfg = AnalysisConfig.from_file(cfg_path)
    except FileNotFoundError:
        print(f'The configuration file "{cfg_path}" was not found. Exiting...')
        sys.exit(1)
    return cfg


def run_analysis(cfg: AnalysisConfig):
    preprocessor = Preprocessor(cfg=cfg)

    analysis = FluxSystematicsAnalysis(
        nominal_flux_df=preprocessor.nominal_flux_df,
        ppfx_correction_df=preprocessor.ppfx_correction_df,
        bin_edges=cfg.bin_edges,
        cfg=cfg,
    )

    analysis.run(pca_threshold=1)

    exporter = Exporter(cfg, analysis)

    exporter.export_products()

    exporter.export_ppfx_output()

    with open(
        exporter.products_file.parent / "flux_covariance_binning_NuMI_GeV.txt", "w"
    ) as f:
        f.write(analysis.matrix_binning_str)

    with open(exporter.products_file.parent / "uncertainties_table.txt", "w") as f:
        f.write(analysis.total_uncertainty_table_latex)

    return exporter.products_file


@timer
def main():
    check_for_ROOT()

    parser = ArgumentParser(
        prog="flux_uncertainties",
        description="This package coerces PPFX output into a neutrino flux prediction with uncertainties, and stores various spectra related to the flux, e.g., fractional uncertainties, covariance matrices, etc.",
    )
    parser.add_argument(
        "-c", "--config", help="specify the path to a toml configuration file"
    )

    parser.add_argument(
        "-p",
        "--plots-only",
        dest="plot",
        metavar="PRODUCTS_FILE",
        help="Specify path to an existing ROOT file for which to produce plots",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.INFO,
    )

    parser.add_argument(
        "-z",
        "--enable-compression",
        action="store_true",
        dest="compression",
        help="Enable compression of the output plots directory",
    )

    parser.add_argument("--example-config", action="store_true", help="Print an example configuration file")

    args = parser.parse_args()

    logging.basicConfig(format="%(message)s", level=args.loglevel)

    if args.example_config:
        with open("./config.toml", "w", encoding="utf-8") as f:
            f.write(AnalysisConfig.dump_default_config())
        logging.info("Example configuration file written to config.toml")
        sys.exit(0)

    cfg_str = args.config

    if cfg_str is None:
        parser.print_help()
        sys.exit(1)

    cfg = load_config(cfg_str)

    plot = args.plot

    products_file = plot if plot is not None else run_analysis(cfg)

    logging.info("\n=============== MAKING PLOTS ===============")

    plot_all(products_file, cfg.nominal_id, cfg.plots_path, cfg.plot_opts, cfg.bin_edges)

    if args.compression:
        compress_directory(cfg.plots_path)

    logging.info("Done.")


if __name__ == "__main__":
    main()
