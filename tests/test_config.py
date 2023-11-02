import pathlib
import tempfile
import tomllib

import numpy as np
import pytest

from flux_tool.config import AnalysisConfig

with open("config.toml", "rb") as cfg:
    project_config = tomllib.load(cfg)

analysis_config = AnalysisConfig(project_config)


def test_create_analysis_config():
    assert analysis_config.plot_opts["draw_label"]
    assert analysis_config.plot_opts["experiment"] == "ICARUS"
    assert analysis_config.plot_opts["stage"] == "Preliminary"
    assert analysis_config.plot_opts["xlim"] == [0.0, 6.0]

    for value in analysis_config.plot_opts["enabled"].values():
        assert value

    assert (analysis_config.bin_edges["nue"] == np.linspace(0, 20, num=201)).all()
    assert (
        analysis_config.bin_edges["nuebar"]
        == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0, 12.0]
    ).all()
    assert (analysis_config.bin_edges["numu"] == np.linspace(0, 20, num=201)).all()
    assert (
        analysis_config.bin_edges["numubar"]
        == [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            6.0,
            8.0,
            12.0,
            20.0,
        ]
    ).all()

    ppfx_enabled = analysis_config.ppfx["enabled"]
    assert ppfx_enabled["attenuation"]
    assert ppfx_enabled["mesinc"]
    assert ppfx_enabled["mesinc_parent_K0"]
    assert ppfx_enabled["mesinc_parent_Km"]
    assert ppfx_enabled["mesinc_parent_Kp"]
    assert ppfx_enabled["mesinc_parent_pim"]
    assert ppfx_enabled["mesinc_parent_pip"]
    assert ppfx_enabled["mesinc_daughter_K0"]
    assert ppfx_enabled["mesinc_daughter_Km"]
    assert ppfx_enabled["mesinc_daughter_Kp"]
    assert ppfx_enabled["mesinc_daughter_pim"]
    assert ppfx_enabled["mesinc_daughter_pip"]
    assert not ppfx_enabled["mippnumi"]
    assert ppfx_enabled["nua"]
    assert not ppfx_enabled["pCfwd"]
    assert ppfx_enabled["pCk"]
    assert ppfx_enabled["pCpi"]
    assert ppfx_enabled["pCnu"]
    assert not ppfx_enabled["pCQEL"]
    assert ppfx_enabled["others"]
    assert not ppfx_enabled["thintarget"]

    assert analysis_config.sources_path == pathlib.Path(
        "/path/to/directory/containing/input/histograms"
    )
    assert analysis_config.results_path == pathlib.Path(
        "/path/to/directory/containing/input/"
    )
    assert analysis_config.plots_path == pathlib.Path(
        "/path/to/directory/containing/input/plots"
    )
    assert analysis_config.products_file.endswith("out.root")


def test_ignored_histogram_names():
    ignored_names = list(analysis_config.ignored_histogram_names)

    assert "hpot" in ignored_names
    assert "hthin_nue" in ignored_names  # thintarget is disabled
    assert "mipp" in ignored_names  # mippnumi is disabled
    assert "other" not in ignored_names  # other is enabled


def test_verify_paths_dont_exist(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    with pytest.raises(FileNotFoundError):
        analysis_config.verify_paths()


def test_parse_filename():
    filenames = [("fhc", 15, "out_0015_200i.root"), ("rhc", 15, "out_0015_-200i.root")]
    for true_horn, true_run_id, file in filenames:
        horn, run_id = AnalysisConfig.parse_filename(file)
        assert horn == true_horn
        assert run_id == true_run_id


def test_ignored_hist_filter():
    assert not analysis_config.ignored_hist_filter("hpot")
    assert not analysis_config.ignored_hist_filter("hthin_numu")
    assert not analysis_config.ignored_hist_filter("mipp")
    assert analysis_config.ignored_hist_filter("hatt")


sample_config = """
output_file_name = "out.root"
sources = "/path/to/sources"

[Plotting]
draw_label = true
experiment = "TestExperiment"
stage = "TestStage"
neutrino_energy_range = [0.0, 20.0]

[Binning]
nue = 200
nuebar = 201
numu = 100
numubar = 150

[PPFX]
[PPFX.enabled]
attenuation = true

[Plotting.enabled]
uncorrected_flux = true
flux_prediction = false
"""


def test_from_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmpfile:
        tmpfile.write(sample_config)

    config_file = tmpfile.name

    analysis_config = AnalysisConfig.from_file(config_file)

    assert analysis_config.plot_opts["draw_label"]
    assert analysis_config.plot_opts["experiment"] == "TestExperiment"
    assert (analysis_config.bin_edges["nue"] == np.linspace(0, 20, num=201)).all()
    assert (analysis_config.bin_edges["nuebar"] == np.linspace(0, 20, num=202)).all()
    assert (analysis_config.bin_edges["numu"] == np.linspace(0, 20, num=101)).all()
    assert (analysis_config.bin_edges["numubar"] == np.linspace(0, 20, num=151)).all()


def test_from_string():
    analysis_config = AnalysisConfig.from_str(sample_config)

    assert analysis_config.plot_opts["draw_label"]
    assert analysis_config.plot_opts["experiment"] == "TestExperiment"
    assert (analysis_config.bin_edges["nue"] == np.linspace(0, 20, num=201)).all()
    assert (analysis_config.bin_edges["nuebar"] == np.linspace(0, 20, num=202)).all()
    assert (analysis_config.bin_edges["numu"] == np.linspace(0, 20, num=101)).all()
    assert (analysis_config.bin_edges["numubar"] == np.linspace(0, 20, num=151)).all()
