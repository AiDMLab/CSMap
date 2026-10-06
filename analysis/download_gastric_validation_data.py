#!/usr/bin/env python3
"""Download the independent gastric-cancer validation subset.

The files are a deliberately small, reproducible subset of GSE270678/GSE270680:
three Visium sections spanning the three reported histological subtypes and five
cross-donor tumour scRNA-seq references.  Existing figure/data directories are
never touched.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "gastric_cancer"
ST_DIR = DATA / "raw_st"
SC_DIR = DATA / "raw_sc"

GEO_SAMPLES = "https://ftp.ncbi.nlm.nih.gov/geo/samples"

# Indeterminate/core, diffuse/core and intestinal/in-between.
ST_SAMPLES = {
    "D02T1": "GSM8349863_D02T1",
    "D18T2": "GSM8349879_D18T2",
    "D24T2": "GSM8349886_D24T2",
}
ST_SUFFIXES = (
    "_filtered_feature_bc_matrix.h5",
    "_scalefactors_json.json.gz",
    "_tissue_hires_image.png.gz",
    "_tissue_positions_list.csv.gz",
)

# All are tumour samples and none is Donor 02, 12 or 24.
SC_SAMPLES = {
    "D03T": "GSM8349955_D03T",
    "D04T": "GSM8349958_D04T",
    "D05T": "GSM8349962_D05T",
}
SC_SUFFIXES = ("_barcodes.tsv.gz", "_features.tsv.gz", "_matrix.mtx.gz")


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"skip {destination.name} ({destination.stat().st_size:,} bytes)")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"download {destination.name}")
    with urlopen(url, timeout=120) as response, partial.open("wb") as handle:
        while block := response.read(1024 * 1024):
            handle.write(block)
    partial.replace(destination)


def sample_url(prefix: str, filename: str) -> str:
    accession = prefix.split("_", 1)[0]
    family = accession[:-3] + "nnn"
    return f"{GEO_SAMPLES}/{family}/{accession}/suppl/{filename}"


def main() -> None:
    for sample, prefix in ST_SAMPLES.items():
        directory = ST_DIR / sample
        for suffix in ST_SUFFIXES:
            filename = prefix + suffix
            download(sample_url(prefix, filename), directory / filename)
    for sample, prefix in SC_SAMPLES.items():
        directory = SC_DIR / sample
        for suffix in SC_SUFFIXES:
            filename = prefix + suffix
            download(sample_url(prefix, filename), directory / filename)


if __name__ == "__main__":
    main()
