from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from ..config.models import ClusteringConfig


class PPXFPipeline:
    """
    Wrapper around the legacy pPXF fitting providing a typed interface.
    """

    def __init__(self, config: ClusteringConfig):
        self.config = config

    def _group_stacked_fits(self) -> Dict[str, List[str]]:
        stacked_dir = self.config.paths.stacked_fits_dir
        fits_files = [
            f
            for f in stacked_dir.iterdir()
            if f.is_file() and f.name.startswith("stacked_") and f.suffix == ".fits"
        ]

        methods: Dict[str, List[str]] = {}
        for fits_file in fits_files:
            base_name = fits_file.stem[len("stacked_") :]  # remove prefix
            method = base_name.rsplit("_", 1)[0]
            methods.setdefault(method, []).append(fits_file.as_posix())

        for files in methods.values():
            files.sort()

        return methods

    def run(self) -> None:
        """
        Execute pPXF fitting on all stacked spectra present in the configured directory.
        """
        from legacy.scripts.stacked_ppxf_fitting import make_catalogue, combine_catalogues

        methods = self._group_stacked_fits()
        if not methods:
            raise RuntimeError("No stacked spectra found to fit.")

        for method, file_names in methods.items():
            make_catalogue(file_names, method, self.config.ppxf.nrand)

        combine_catalogues()
