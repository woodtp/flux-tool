from functools import cached_property
from pathlib import Path

import uproot


class SpectraReader:
    def __init__(self, products_file: Path | str):
        self._f = uproot.open(products_file)

    def __del__(self):
        self._f.close()  # type: ignore

    @cached_property
    def flux_weights(self):
        return {
            key: h for key, h in self._f["ppfx_flux_weights"].items(cycle=False)  # type: ignore
        }

    @cached_property
    def hadron_uncertainties(self):
        return {
            key.split("/")[0]: h.to_pyroot()
            for key, h in self._f["fractional_uncertainties/hadron/"].items(
                cycle=False  # type: ignore
            )
            if "total" not in key
            and "projectile" not in key
            and "daughter" not in key
            and "qel" not in key.lower()
        }

    @cached_property
    def nominal_spectra(self):
        return {
            key: h
            for key, h in self._f["ppfx_output"].items(
                cycle=False, filter_name="*/nom/hnom*"  # type: ignore
            )
        }

    @cached_property
    def parent_spectra(self):
        return {
            key: h
            for key, h in self._f["ppfx_output"].items(
                cycle=False, filter_name="*/nom/parent/*"  # type: ignore
            )
        }

    @cached_property
    def pot(self):
        return {
            "fhc": self._f["ppfx_output/fhc/pot"].values().max(),  # type: ignore
            "rhc": self._f["ppfx_output/rhc/pot"].values().max(),  # type: ignore
        }

    @cached_property
    def ppfx_correction(self):
        return {
            key: h for key, h in self._f["ppfx_corrected_flux/total"].items(cycle=False)  # type: ignore
        }

    @cached_property
    def universes(self):
        return {
            key: h
            for key, h in self._f["ppfx_output"].items(
                filter_name="*/*_total/*", cycle=False  # type: ignore
            )
        }
