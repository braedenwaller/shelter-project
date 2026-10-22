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
