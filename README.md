# Who Waits Longest? Length of stay at Austin Animal Center

An interactive Streamlit data story built for **CMSE 830 (Fall 2026, MSU)**.

> **Live app:** _add your Streamlit Community Cloud link here after deploying_
> **Run locally:** see [Setup](#setup-and-run) below (takes about two minutes).

## The question and who it is for

Austin Animal Center is the largest no-kill shelter in the United States and publishes a record of every
animal that comes in and every animal that leaves. This app joins those two records into one table of
**shelter stays** and asks:

> **Which dogs and cats wait longest for a home, and what moves the wait?**

It is built for three kinds of user:

| User | What they can do with it |
|---|---|
| Shelter staff and rescue partners | See which animals (age, breed, condition, intake route) sit longest, so marketing and foster outreach can start on day one |
| Foster volunteers and would-be adopters | Learn which animals are overlooked, and how the picture has changed since the pandemic |
| Students of data science | Walk through a real, messy two-table integration problem with every decision explained in the app |

## What the app shows

| Tab | Content |
|---|---|
| **Overview** | How to use the app, headline metrics, median wait by age band for dogs and cats, and the outcome mix (adopted / returned to owner / transferred / euthanised / died) |
| **Explore a factor** | Pick any factor (age, intake type, condition, spay/neuter status, name, breed, colour, season, repeat visit) and see the median wait, a day-by-day "share still at the shelter" curve, and the outcome mix per group |
| **Trends over time** | Monthly intakes (kitten season, COVID), median wait by year of intake with a follow-up toggle, live-release rate |
| **Data & decisions** | Every cleaning step with the counts it affected, including a live demonstration of why dropping unfinished stays is a mistake |
| **Findings & limits** | Six evidence-based findings, the limitations and alternative explanations, and next steps for the final project |

Sidebar filters (species, intake years, intake type, measure, minimum group size) apply to every chart.

## Key findings (Oct 2013 - May 2025)

1. Half of all dogs and cats leave within 6 days, but mostly by going home to an owner or to a rescue partner.
   Among animals that are **adopted**, the median wait is 9 days for dogs and 25 days for cats.
2. **Breed is the biggest single factor for dogs.** Adopted pit-bull-type dogs wait a median 28 days vs 8 days for all other dogs.
3. **Senior dogs are the most at-risk group**: only 29% are adopted, and those that are wait 21 days.
4. **Kittens "wait" 30 days, mostly to grow up** to adoption age; intact cats wait twice as long as already-fixed cats.
5. **Dog waits tripled after the pandemic**: median days to adoption for dogs rose from 7 (2019 intakes) to 27 (2023 intakes).
6. **"Black dog syndrome" does not appear here** (9 days vs 9 days); black cats wait slightly longer (29 vs 24 days).

## Data

| Source | Rows | Link |
|---|---|---|
| Austin Animal Center Intakes | 173,812 | https://data.austintexas.gov/d/wter-evkm |
| Austin Animal Center Outcomes | 173,775 | https://data.austintexas.gov/d/9t4d-g238 |

Both are public-domain datasets from the City of Austin Open Data Portal, downloaded on 2026-10-21. The
snapshot contains records from 2013-10-01 to 2025-05-05. The portal updates continuously, so a fresh download
will contain more rows than the numbers quoted in the app.

The raw files (45 MB) are **not** committed. The processed table the app reads,
`data/processed/stays.parquet` (163,895 stays, 5.8 MB), **is** committed together with
`data/processed/quality_report.json`, which holds every count the app and this README quote.

### Key data-preparation decisions (full detail in the app's *Data & decisions* tab)

- **Two date formats + stray UTC offsets.** Intakes use `MM/DD/YYYY hh:mm:ss AM`; outcomes use ISO 8601, and
  3,584 outcome rows carry a `-05:00` offset with a midnight time (date only). Each file is parsed with an explicit
  format and treated as Austin local time.
- **Ages are text** (`2 years`, `3 weeks`). Converted to days and binned. `0 years` (a placeholder, not a
  newborn) and negative ages (date of birth after intake) are treated as unknown, not imputed.
- **Matching intakes to outcomes.** 13,421 animals have multiple visits; a plain join on Animal ID produces
  211,165 pairs, 24,776 of them with the outcome before the intake. Instead each animal's events are walked in time
  order: every intake opens a stay and the first following outcome closes it.
- **Unfinished stays are censored, not dropped.** 1,537 intakes have no outcome yet. Dropping them removes exactly
  the animals that wait longest; the app's "still waiting" curves use a Kaplan-Meier estimate and the trend tab
  offers a 180-day follow-up rule.
- **Missing values are informative or structural**, so nothing is imputed: a missing name marks a stray, a missing
  outcome subtype is normal for most outcome types.
- **Categories consolidated**: 20 intake-condition labels to 7 groups, 11 outcome types to 6, 3,000 breed strings
  to a primary breed + mixed flag.
- **No outlier removal, scaling or encoding**: medians and quartiles handle the skew; there is no model yet.

## Repository layout

```
shelter-project/
├── app.py                      Streamlit app (single file, five tabs)
├── requirements.txt
├── README.md
├── AI_DISCLOSURE.md            Generative-AI use statement (required by the course)
├── PRESENTATION.md             7-minute talk outline and demo click-path
├── data/
│   ├── raw/                    intakes.csv, outcomes.csv  (git-ignored; see src/download_data.py)
│   └── processed/
│       ├── stays.parquet       one row per shelter stay (committed)
│       └── quality_report.json every count quoted in the app (committed)
├── notebooks/
│   └── 01_data_quality_and_eda.ipynb   first look, quality investigation, EDA (executed, outputs saved)
└── src/
    ├── download_data.py        fetch the two raw CSVs from the Austin portal
    └── prepare_data.py         raw CSVs -> stays.parquet + quality_report.json (about 2 s)
```

## Setup and run

Requires Python 3.10 or newer.

```bash
git clone https://github.com/<your-username>/shelter-project.git
cd shelter-project
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501 and reads the committed `data/processed/stays.parquet`, so no download is
needed. To rebuild the processed data from scratch (e.g. with a fresh snapshot from the portal):

```bash
python src/download_data.py     # ~45 MB into data/raw/
python src/prepare_data.py      # writes data/processed/stays.parquet and quality_report.json
```

To deploy on Streamlit Community Cloud: push the repository to GitHub, go to https://share.streamlit.io, pick the
repo, and set the main file to `app.py`. No secrets are required.

## Limitations

- One shelter, and an unusually well-resourced one (96% live release). Results will not transfer to every shelter.
- "Days to adoption" is only observed for animals that were adopted; it must be read next to the outcome mix.
- 29% of stays end in transfers to partner rescues, and what happened afterwards is not in the data.
- Comparisons are descriptive, not causal: e.g. named animals wait longer largely because they are owner surrenders.
- Time in foster care counts as time "in care"; ages and breeds are staff estimates; 3,788 outcomes have a date but
  no time; the data ends 2025-05-05.

## Generative AI use

See [AI_DISCLOSURE.md](AI_DISCLOSURE.md).
