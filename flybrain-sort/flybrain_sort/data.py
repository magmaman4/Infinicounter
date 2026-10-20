"""Download and cache the FlyWire (v783) whole-brain connectome.

Sources
-------
* Connectivity + neuron list: Shiu et al. (2024), "A Drosophila computational
  brain model reveals sensorimotor processing", Nature 634, 210-219.
  https://github.com/philshiu/Drosophila_brain_model
* Cell-type annotations: Schlegel et al. (2024), "Whole-brain annotation and
  multi-connectome cell typing of Drosophila", Nature 634, 139-152.
  https://github.com/flyconnectome/flywire_annotations

Both derive from the FlyWire Consortium connectome (Dorkenwald et al. 2024,
Nature 634, 124-138). Please cite them if you use this project.
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path

_SHIU = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/"
_ANN = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
    "supplemental_files/"
)

FILES = {
    "Completeness_783.csv": _SHIU + "Completeness_783.csv",
    "Connectivity_783.parquet": _SHIU + "Connectivity_783.parquet",
    "Supplemental_file1_neuron_annotations.tsv": _ANN
    + "Supplemental_file1_neuron_annotations.tsv",
}


def data_dir() -> Path:
    """Directory holding the connectome files (``$FLYBRAIN_DATA`` or ``./data``)."""
    return Path(os.environ.get("FLYBRAIN_DATA", "data")).expanduser()


def fetch(name: str, directory: Path | None = None, quiet: bool = False) -> Path:
    """Return the local path of ``name``, downloading it on first use."""
    directory = directory or data_dir()
    path = directory / name
    if path.exists():
        return path
    directory.mkdir(parents=True, exist_ok=True)
    url = FILES[name]
    if not quiet:
        print(f"downloading {name} ...", file=sys.stderr)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.rename(path)
    return path


def fetch_all(directory: Path | None = None, quiet: bool = False) -> dict[str, Path]:
    return {name: fetch(name, directory, quiet) for name in FILES}
