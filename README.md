# Who Waits Longest?

A Streamlit app for CMSE 830 (Fall 2026) about how long dogs and cats wait to be adopted at Austin Animal Center.

**Question:** Which dogs and cats wait longest for a home, and what affects the wait?

**Audience:** Shelter staff and foster volunteers deciding where to focus their effort, and people thinking about adopting.

## How to run the app

**Option 1: open it online.** Live app: _link coming soon_

**Option 2: run it on your own computer.** This needs Python 3.10 or newer. Open a terminal and run these four commands, one at a time:

```
git clone https://github.com/braedenwaller/shelter-project.git
cd shelter-project
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens in your browser at http://localhost:8501. To stop it, press Ctrl+C in the terminal.

The cleaned data is included in this repository, so nothing else needs to be downloaded. If `python` is not found, try `python3` instead.

To rebuild the cleaned data from the original files (optional):

```
python src/download_data.py
python src/prepare_data.py
```

## Data

Two public tables from the City of Austin Open Data Portal, downloaded October 21, 2026. They cover October 2013 to May 2025.

- [Austin Animal Center Intakes](https://data.austintexas.gov/d/wter-evkm): 173,812 rows
- [Austin Animal Center Outcomes](https://data.austintexas.gov/d/9t4d-g238): 173,775 rows

Each row of the cleaned data is one stay: one intake matched to the outcome that ended it. The main cleaning steps were:

- Matching each intake to the next outcome for the same animal, because 13,421 animals came in more than once.
- Keeping stays with no outcome yet instead of dropping them.
- Converting ages from text to days, and treating "0 years" and negative ages as unknown.
- Keeping dogs and cats only.
- Using medians, because stay lengths are heavily skewed.

The app's Data & decisions tab and `notebooks/01_data_quality_and_eda.ipynb` explain each step.

## Files

- `app.py`: the Streamlit app
- `src/download_data.py`: downloads the original data files
- `src/prepare_data.py`: cleans them into `data/processed/stays.parquet`
- `notebooks/`: data quality checks and exploration
- `data/processed/`: the cleaned data the app uses

## Limitations

- This is one shelter. Austin releases 96% of its dogs and cats alive, which is higher than many shelters.
- Days to adoption only exists for animals that were adopted.
- The findings are comparisons, not causes.
- 29% of stays end in a transfer to a rescue group, and the data doesn't show what happened after.

## AI use

See [AI_DISCLOSURE.md](AI_DISCLOSURE.md).
