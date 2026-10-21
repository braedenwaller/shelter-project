"""
Build the stay-level analysis table from the two raw Austin Animal Center files.

Run from the project root:

    python src/prepare_data.py

Input  : data/raw/intakes.csv, data/raw/outcomes.csv   (see src/download_data.py)
Output : data/processed/stays.parquet                   (one row per shelter stay)
         data/processed/quality_report.json             (every count quoted in the app/README)

Each numbered step below is a data decision that is explained in the app's
"Data & decisions" tab and in the README.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# --------------------------------------------------------------------------
# Category groupings.  The raw files use 20 intake-condition labels and 11
# outcome labels, several of which were added over the years and have only a
# handful of rows.  Grouping them keeps every chart readable and every group
# large enough to summarise.
# --------------------------------------------------------------------------
CONDITION_GROUPS = {
    "Normal": "Normal",
    "Injured": "Injured",
    "Sick": "Sick / medical",
    "Medical": "Sick / medical",
    "Med Attn": "Sick / medical",
    "Med Urgent": "Sick / medical",
    "Parvo": "Sick / medical",
    "Panleuk": "Sick / medical",
    "Neurologic": "Sick / medical",
    "Agonal": "Sick / medical",
    "Congenital": "Sick / medical",
    "Nursing": "Nursing / neonatal / pregnant",
    "Neonatal": "Nursing / neonatal / pregnant",
    "Pregnant": "Nursing / neonatal / pregnant",
    "Aged": "Aged",
    "Behavior": "Behavioral / feral",
    "Feral": "Behavioral / feral",
    "Other": "Other / unknown",
    "Unknown": "Other / unknown",
    "Space": "Other / unknown",
}

OUTCOME_GROUPS = {
    "Adoption": "Adoption",
    "Transfer": "Transfer to partner",
    "Return to Owner": "Return to owner",
    "Rto-Adopt": "Return to owner",      # owner surrendered, then adopted their own animal back
    "Euthanasia": "Euthanasia",
    "Died": "Died in care",
    "Disposal": "Died in care",          # arrived dead / died before intake processing
    "Missing": "Other / unknown",
    "Stolen": "Other / unknown",
    "Lost": "Other / unknown",
    "Relocate": "Other / unknown",
}

# Age-group edges in days.  "1 year" in the raw data means 1 <= age < 2, so a
# "1 year" animal lands in the 6 months - 2 years band.
AGE_EDGES = [-np.inf, 182, 730, 2556, np.inf]
AGE_LABELS = ["Under 6 months", "6 months - 2 years", "2 - 7 years", "7+ years"]

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def parse_age_to_days(text: str | float) -> float:
    """'2 years' -> 730.5, '3 weeks' -> 21.  '0 years' and negative ages -> NaN.

    "0 years" is a placeholder: for 80 % of those rows the recorded date of
    birth equals the intake date, i.e. staff did not estimate an age.  Negative
    ages come from a date of birth typed in after the intake date.  Neither is a
    real age, so both become missing rather than 0.
    """
    if not isinstance(text, str):
        return np.nan
    number, unit = text.split()
    number = int(number)
    if number <= 0:
        return np.nan
    days_per_unit = {"year": 365.25, "month": 30.44, "week": 7.0, "day": 1.0}
    return number * days_per_unit[unit.rstrip("s")]


def split_sex(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'Neutered Male' -> ('Male', 'Fixed'); 'Intact Female' -> ('Female', 'Intact')."""
    s = series.fillna("Unknown")
    sex = np.select([s.str.contains("Male"), s.str.contains("Female")], ["Male", "Female"], "Unknown")
    fixed = np.select([s.str.contains("Neutered|Spayed"), s.str.contains("Intact")], ["Fixed", "Intact"], "Unknown")
    return pd.Series(sex, index=series.index), pd.Series(fixed, index=series.index)


def primary_breed(series: pd.Series) -> pd.Series:
    """'Labrador Retriever/Pit Bull Mix' -> 'Labrador Retriever'. Keeps the first listed breed."""
    return series.str.split("/").str[0].str.replace(" Mix", "", regex=False).str.strip()


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def main() -> None:
    report: dict = {}

    # 1. Load -----------------------------------------------------------------
    intakes = pd.read_csv(RAW / "intakes.csv")
    outcomes = pd.read_csv(RAW / "outcomes.csv")
    report["raw_rows"] = {"intakes": len(intakes), "outcomes": len(outcomes)}
    report["raw_missing_intakes"] = intakes.isna().sum().to_dict()
    report["raw_missing_outcomes"] = outcomes.isna().sum().to_dict()
    report["animal_type_counts"] = intakes["Animal Type"].value_counts().to_dict()

    # 2. Dates --------------------------------------------------------------
    # The two files use different formats: intakes "05/05/2025 07:22:00 AM",
    # outcomes ISO "2025-05-05T07:22:00".  3,584 outcome rows also carry a UTC
    # offset ("-05:00") and all of those are stamped exactly midnight, i.e. the
    # time of day was never recorded.  We strip the offset and treat every
    # timestamp as Austin local time, which is what the intake file already is.
    has_offset = outcomes["DateTime"].str.contains(r"[+-]\d\d:\d\d$", regex=True)
    report["outcome_rows_with_utc_offset"] = int(has_offset.sum())
    intakes["dt"] = pd.to_datetime(intakes["DateTime"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    outcomes["dt"] = pd.to_datetime(
        outcomes["DateTime"].str.replace(r"[+-]\d\d:\d\d$", "", regex=True),
        format="%Y-%m-%dT%H:%M:%S", errors="coerce",
    )
    report["unparsed_dates"] = {"intakes": int(intakes["dt"].isna().sum()),
                                "outcomes": int(outcomes["dt"].isna().sum())}
    report["outcomes_stamped_midnight"] = int((outcomes["dt"].dt.hour.eq(0) & outcomes["dt"].dt.minute.eq(0)
                                               & outcomes["dt"].dt.second.eq(0)).sum())
    report["date_range"] = {"intakes": [str(intakes["dt"].min()), str(intakes["dt"].max())],
                            "outcomes": [str(outcomes["dt"].min()), str(outcomes["dt"].max())]}

    # 3. Duplicates ---------------------------------------------------------
    report["exact_duplicate_rows"] = {"intakes": int(intakes.duplicated().sum()),
                                      "outcomes": int(outcomes.duplicated().sum())}
    intakes = intakes.drop_duplicates()
    outcomes = outcomes.drop_duplicates()
    # A second record of the same animal at the same minute is a double entry
    # even when one field differs (e.g. a retyped found-location).
    report["same_animal_same_time_duplicates"] = {
        "intakes": int(intakes.duplicated(["Animal ID", "dt"]).sum()),
        "outcomes": int(outcomes.duplicated(["Animal ID", "dt"]).sum())}
    intakes = intakes.drop_duplicates(["Animal ID", "dt"])
    outcomes = outcomes.drop_duplicates(["Animal ID", "dt"])

    # 4. Dogs and cats only -------------------------------------------------
    # "Other" is mostly wildlife (bats, raccoons) brought in for rabies testing,
    # which is euthanised almost immediately and is not part of the adoption
    # question.  Birds and livestock are too few to summarise.
    keep = intakes["Animal Type"].isin(["Dog", "Cat"])
    report["rows_dropped_not_dog_or_cat"] = int((~keep).sum())
    intakes = intakes[keep].copy()
    outcomes = outcomes[outcomes["Animal Type"].isin(["Dog", "Cat"])].copy()

    # 5. Ages ---------------------------------------------------------------
    age_text = intakes["Age upon Intake"]
    report["age_zero_years_rows"] = int(age_text.eq("0 years").sum())
    report["age_negative_rows"] = int(age_text.str.startswith("-").sum())
    intakes["age_days"] = age_text.map(parse_age_to_days)
    report["age_unknown_after_parse"] = int(intakes["age_days"].isna().sum())

    # 6. Match each intake to its outcome ------------------------------------
    # One animal can come in many times (one dog has 33 intakes).  Joining on
    # Animal ID alone pairs every intake with every outcome.  Instead we list
    # every event for each animal in time order; each intake starts a new
    # "stay", and the first outcome inside that stay is its exit.
    intakes = intakes.sort_values(["Animal ID", "dt"]).reset_index(drop=True)
    outcomes = outcomes.sort_values(["Animal ID", "dt"]).reset_index(drop=True)

    naive = intakes[["Animal ID", "dt"]].merge(outcomes[["Animal ID", "dt"]], on="Animal ID",
                                               suffixes=("_in", "_out"))
    report["naive_join"] = {"rows": len(naive),
                            "pairs_with_outcome_before_intake": int((naive["dt_out"] < naive["dt_in"]).sum())}
    del naive

    events = pd.concat([
        pd.DataFrame({"animal_id": intakes["Animal ID"], "dt": intakes["dt"], "kind": 0, "row": intakes.index}),
        pd.DataFrame({"animal_id": outcomes["Animal ID"], "dt": outcomes["dt"], "kind": 1, "row": outcomes.index}),
    ], ignore_index=True)
    # kind=0 (intake) sorts before kind=1 (outcome) at an identical timestamp.
    events = events.sort_values(["animal_id", "dt", "kind"], kind="mergesort").reset_index(drop=True)
    events["stay"] = (events["kind"] == 0).groupby(events["animal_id"]).cumsum()
    events["pos"] = events.groupby(["animal_id", "stay"]).cumcount()

    orphan_outcomes = (events["kind"] == 1) & (events["stay"] == 0)   # outcome before any recorded intake
    extra_outcomes = (events["kind"] == 1) & (events["pos"] >= 2)      # second outcome with no intake between
    first_outcomes = (events["kind"] == 1) & (events["pos"] == 1) & (events["stay"] > 0)
    report["matching"] = {
        "outcomes_before_first_intake": int(orphan_outcomes.sum()),
        "extra_outcomes_in_same_stay": int(extra_outcomes.sum()),
    }

    intake_rows = events[events["kind"] == 0][["animal_id", "stay", "row"]].rename(columns={"row": "intake_row"})
    outcome_rows = events[first_outcomes][["animal_id", "stay", "row"]].rename(columns={"row": "outcome_row"})
    pairs = intake_rows.merge(outcome_rows, on=["animal_id", "stay"], how="left")

    stays = intakes.loc[pairs["intake_row"].values].reset_index(drop=True)
    stays["stay_number"] = pairs["stay"].values
    out_cols = outcomes[["dt", "Outcome Type", "Outcome Subtype", "Sex upon Outcome"]].rename(columns={
        "dt": "outcome_dt", "Outcome Type": "outcome_type_raw", "Outcome Subtype": "outcome_subtype",
        "Sex upon Outcome": "sex_upon_outcome"})
    matched = out_cols.reindex(pairs["outcome_row"].values).reset_index(drop=True)
    stays = pd.concat([stays, matched], axis=1)

    # 7. Derived columns ----------------------------------------------------
    as_of = outcomes["dt"].max().normalize()          # last day with any record
    report["as_of_date"] = str(as_of.date())

    stays["has_outcome"] = stays["outcome_dt"].notna()
    stays["los_days"] = (stays["outcome_dt"] - stays["dt"]).dt.total_seconds() / 86400
    # For open stays, the time from intake to the data's end date is a *lower
    # bound* on the eventual length of stay (right-censoring).
    stays["los_lower_bound_days"] = np.where(
        stays["has_outcome"], stays["los_days"], (as_of - stays["dt"]).dt.total_seconds() / 86400)

    stays["age_years"] = stays["age_days"] / 365.25
    stays["age_group"] = pd.cut(stays["age_days"], AGE_EDGES, labels=AGE_LABELS, right=False).astype(object)
    stays["age_group"] = stays["age_group"].fillna("Unknown")

    stays["condition_group"] = stays["Intake Condition"].map(CONDITION_GROUPS).fillna("Other / unknown")
    stays["outcome_group"] = stays["outcome_type_raw"].map(OUTCOME_GROUPS)
    stays.loc[stays["has_outcome"] & stays["outcome_group"].isna(), "outcome_group"] = "Other / unknown"
    stays.loc[~stays["has_outcome"], "outcome_group"] = "No outcome recorded"
    stays["adopted"] = stays["outcome_group"].eq("Adoption")

    stays["sex"], stays["fixed_at_intake"] = split_sex(stays["Sex upon Intake"])
    _, stays["fixed_at_outcome"] = split_sex(stays["sex_upon_outcome"])
    stays["arrived_with_name"] = stays["Name"].notna()
    stays["primary_breed"] = primary_breed(stays["Breed"])
    stays["is_mix"] = stays["Breed"].str.contains("Mix|/", regex=True)
    stays["primary_color"] = stays["Color"].str.split("/").str[0].str.strip()
    stays["intake_year"] = stays["dt"].dt.year
    stays["intake_month"] = stays["dt"].dt.month
    stays["season"] = stays["intake_month"].map(SEASONS)

    stays = stays.rename(columns={
        "Animal ID": "animal_id", "dt": "intake_dt", "Animal Type": "species",
        "Intake Type": "intake_type", "Intake Condition": "intake_condition_raw",
        "Breed": "breed_raw", "Color": "color_raw", "Age upon Intake": "age_text",
    })
    keep_cols = [
        "animal_id", "stay_number", "species", "intake_dt", "outcome_dt", "has_outcome",
        "los_days", "los_lower_bound_days", "intake_type", "intake_condition_raw", "condition_group",
        "age_text", "age_days", "age_years", "age_group", "sex", "fixed_at_intake", "fixed_at_outcome",
        "arrived_with_name", "breed_raw", "primary_breed", "is_mix", "color_raw", "primary_color",
        "intake_year", "intake_month", "season", "outcome_type_raw", "outcome_subtype", "outcome_group", "adopted",
    ]
    stays = stays[keep_cols].sort_values(["intake_dt", "animal_id"]).reset_index(drop=True)

    # 8. Report + save ------------------------------------------------------
    open_stays = stays[~stays["has_outcome"]]
    report["stays"] = {
        "total": len(stays),
        "with_outcome": int(stays["has_outcome"].sum()),
        "open": int((~stays["has_outcome"]).sum()),
        "open_by_intake_year": open_stays["intake_year"].value_counts().sort_index().to_dict(),
        "animals": int(stays["animal_id"].nunique()),
        "animals_with_repeat_stays": int((stays["animal_id"].value_counts() > 1).sum()),
        "max_stays_one_animal": int(stays["animal_id"].value_counts().max()),
        "negative_los": int((stays["los_days"] < 0).sum()),
        "los_over_365": int((stays["los_days"] > 365).sum()),
    }
    report["los_summary"] = {
        "median": float(stays["los_days"].median()), "mean": float(stays["los_days"].mean()),
        "p90": float(stays["los_days"].quantile(0.9)), "max": float(stays["los_days"].max()),
    }
    report["missing_name_share_by_intake_type"] = (
        stays.groupby("intake_type")["arrived_with_name"].apply(lambda s: round(1 - s.mean(), 3)).to_dict())
    report["missing_subtype_share_by_outcome"] = (
        stays[stays["has_outcome"]].groupby("outcome_type_raw")["outcome_subtype"]
        .apply(lambda s: round(s.isna().mean(), 3)).to_dict())
    report["condition_groups"] = CONDITION_GROUPS
    report["outcome_groups"] = OUTCOME_GROUPS

    OUT.mkdir(parents=True, exist_ok=True)
    stays.to_parquet(OUT / "stays.parquet", index=False)
    with open(OUT / "quality_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"Wrote {len(stays):,} stays to {OUT / 'stays.parquet'}")
    print(json.dumps({k: report[k] for k in ["raw_rows", "exact_duplicate_rows", "rows_dropped_not_dog_or_cat",
                                               "naive_join", "matching", "stays", "los_summary"]},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
