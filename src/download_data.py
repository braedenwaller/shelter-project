"""
Download the two raw Austin Animal Center files into data/raw/.

Run from the project root:

    python src/download_data.py

Source: City of Austin Open Data Portal (public domain)
  Intakes  https://data.austintexas.gov/d/wter-evkm
  Outcomes https://data.austintexas.gov/d/9t4d-g238

The raw files are ~25 MB and ~20 MB and are not committed to the repository.
The processed file the app uses (data/processed/stays.parquet) *is* committed,
so you only need this script if you want to rebuild it from scratch with
`python src/prepare_data.py`.  Note that the portal updates continuously, so a
fresh download will contain newer rows than the snapshot used here
(downloaded 2026-10-21; records run through 2025-05-05).
"""

from pathlib import Path
from urllib.request import urlretrieve

FILES = {
    "intakes.csv": "https://data.austintexas.gov/api/views/wter-evkm/rows.csv?accessType=DOWNLOAD",
    "outcomes.csv": "https://data.austintexas.gov/api/views/9t4d-g238/rows.csv?accessType=DOWNLOAD",
}

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = RAW / name
        print(f"Downloading {name} ...", end=" ", flush=True)
        urlretrieve(url, target)
        print(f"{target.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
