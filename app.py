"""
Who Waits Longest?  -  Length of stay at Austin Animal Center
CMSE 830 midterm project (Fall 2026)

Run with:  streamlit run app.py
Data:      data/processed/stays.parquet  (built by src/prepare_data.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup and constants
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Who Waits Longest?", page_icon="🐾", layout="wide")

DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"

# Colour follows the entity everywhere in the app: dogs are always blue, cats
# always orange.  Palette slots are taken in the validated (colour-blind-safe)
# order of the course data-viz reference palette.
SPECIES_ORDER = ["Dog", "Cat"]
SPECIES_COLORS = ["#2a78d6", "#eb6834"]

OUTCOME_ORDER = ["Adoption", "Transfer to partner", "Return to owner",
                 "Euthanasia", "Died in care", "Other / unknown"]
OUTCOME_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

LEVEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # for up-to-4 user-chosen groups

AGE_ORDER = ["Under 6 months", "6 months - 2 years", "2 - 7 years", "7+ years", "Unknown"]
SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
CONDITION_ORDER = ["Normal", "Injured", "Sick / medical", "Nursing / neonatal / pregnant",
                   "Aged", "Behavioral / feral", "Other / unknown"]

# The factors a user can explore, and how each one is labelled.
FACTORS = {
    "Age at intake": "age_group",
    "Intake type": "intake_type",
    "Condition at intake": "condition_group",
    "Spayed/neutered at intake": "fixed_at_intake",
    "Sex": "sex",
    "Arrived with a name": "name_label",
    "Primary breed (dogs)": "primary_breed",
    "Primary colour": "primary_color",
    "Season of intake": "season",
    "First visit or repeat visit": "visit_label",
}
FACTOR_ORDER = {
    "age_group": AGE_ORDER, "season": SEASON_ORDER, "condition_group": CONDITION_ORDER,
    "fixed_at_intake": ["Intact", "Fixed", "Unknown"], "sex": ["Male", "Female", "Unknown"],
    "name_label": ["Had a name", "No name"], "visit_label": ["First visit", "Repeat visit"],
}

MEASURES = {
    "Days until adoption (adopted animals only)": "adoption",
    "Days until leaving the shelter (any outcome)": "any",
}


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading shelter stays...")
def load_data() -> tuple[pd.DataFrame, dict]:
    stays = pd.read_parquet(DATA_DIR / "stays.parquet")
    with open(DATA_DIR / "quality_report.json") as fh:
        report = json.load(fh)
    # Small display helpers that are easier to build here than in the pipeline.
    stays["name_label"] = np.where(stays["arrived_with_name"], "Had a name", "No name")
    stays["visit_label"] = np.where(stays["stay_number"] == 1, "First visit", "Repeat visit")
    stays["intake_month_start"] = stays["intake_dt"].dt.to_period("M").dt.to_timestamp()
    return stays, report


if not (DATA_DIR / "stays.parquet").exists():
    st.error("`data/processed/stays.parquet` is missing. Run `python src/download_data.py` "
             "then `python src/prepare_data.py` from the project folder first.")
    st.stop()

stays, report = load_data()
AS_OF = pd.Timestamp(report["as_of_date"])


# ----------------------------------------------------------------------------
# Small analysis helpers
# ----------------------------------------------------------------------------
def summarise(df: pd.DataFrame, by: list[str], value: str = "los_days") -> pd.DataFrame:
    """Median, quartiles and count of `value` per group.  Medians, not means:
    stays are heavily right-skewed (mean 22 days vs median 6)."""
    g = df.groupby(by, observed=True)[value]
    out = g.agg(median="median", n="size").reset_index()
    out["q25"] = g.quantile(0.25).values
    out["q75"] = g.quantile(0.75).values
    for col in by:
        out[col] = out[col].astype(str)
    return out


def outcome_mix(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Share of each outcome group within each `by` group (closed stays only)."""
    closed = df[df["has_outcome"]]
    counts = closed.groupby(by + ["outcome_group"], observed=True).size().rename("n").reset_index()
    counts["share"] = counts["n"] / counts.groupby(by)["n"].transform("sum")
    counts["outcome_rank"] = counts["outcome_group"].map({o: i for i, o in enumerate(OUTCOME_ORDER)})
    for col in by:
        counts[col] = counts[col].astype(str)
    return counts


def still_waiting_curve(time_days: np.ndarray, event: np.ndarray, horizon: int = 120) -> np.ndarray:
    """Kaplan-Meier estimate of the share of animals still at the shelter after
    each day 0..horizon.  Open stays are *censored* at the data's end date
    (they count as 'still waiting' only for the days we actually observed
    them), instead of being dropped."""
    t = np.asarray(time_days, dtype=float)
    e = np.asarray(event, dtype=bool)
    order = np.argsort(t, kind="mergesort")
    t, e = t[order], e[order]
    if e.sum() == 0:
        return np.ones(horizon + 1)
    event_times, first_idx = np.unique(t[e], return_index=True)
    deaths = np.bincount(np.searchsorted(event_times, t[e]))
    at_risk = len(t) - np.searchsorted(t, event_times, side="left")
    surv = np.cumprod(1.0 - deaths / at_risk)
    grid = np.arange(0, horizon + 1)
    pos = np.searchsorted(event_times, grid, side="right") - 1
    return np.where(pos >= 0, surv[np.clip(pos, 0, None)], 1.0)


def fmt_days(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:.0f} days" if x >= 10 else f"{x:.1f} days"


def species_color(n_species: int):
    return alt.Color("species:N", title="Species",
                     scale=alt.Scale(domain=SPECIES_ORDER, range=SPECIES_COLORS),
                     legend=alt.Legend(orient="top") if n_species > 1 else None)


# ----------------------------------------------------------------------------
# Chart builders (all take small, pre-aggregated frames)
# ----------------------------------------------------------------------------
def median_bar_chart(summary: pd.DataFrame, cat_col: str, cat_title: str, measure_label: str,
                     order: list[str] | None, n_species: int, height: int | None = None) -> alt.Chart:
    if order is None:
        order = (summary.groupby(cat_col)["median"].max().sort_values(ascending=False).index.tolist())
    else:
        order = [o for o in order if o in set(summary[cat_col])]
    height = height or max(160, 28 * len(order) * max(n_species, 1) + 40)
    base = alt.Chart(summary).encode(
        y=alt.Y(f"{cat_col}:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        yOffset=alt.YOffset("species:N", sort=SPECIES_ORDER) if n_species > 1 else alt.YOffset(),
        color=species_color(n_species),
        tooltip=[alt.Tooltip(f"{cat_col}:N", title=cat_title), alt.Tooltip("species:N", title="Species"),
                 alt.Tooltip("median:Q", title=f"Median {measure_label}", format=".1f"),
                 alt.Tooltip("q25:Q", title="25th percentile", format=".1f"),
                 alt.Tooltip("q75:Q", title="75th percentile", format=".1f"),
                 alt.Tooltip("n:Q", title="Stays", format=",")],
    )
    bars = base.mark_bar(cornerRadiusEnd=3, size=18 if n_species == 1 else 12).encode(
        x=alt.X("median:Q", title=f"Median {measure_label}"))
    iqr = base.mark_rule(strokeWidth=1.5, color="#52514e", opacity=0.55).encode(
        x=alt.X("q25:Q"), x2=alt.X2("q75:Q"))
    return (bars + iqr).properties(height=height).configure_view(strokeWidth=0)


def outcome_mix_chart(mix: pd.DataFrame, cat_col: str, cat_title: str, order: list[str] | None,
                      facet_species: bool) -> alt.Chart:
    if order is None:
        adopt = mix[mix["outcome_group"] == "Adoption"].groupby(cat_col)["share"].max()
        order = adopt.sort_values(ascending=False).index.tolist()
    else:
        order = [o for o in order if o in set(mix[cat_col])]
    chart = alt.Chart(mix).mark_bar().encode(
        y=alt.Y(f"{cat_col}:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("share:Q", stack="normalize", title="Share of closed stays", axis=alt.Axis(format="%")),
        color=alt.Color("outcome_group:N", title="Outcome",
                        scale=alt.Scale(domain=OUTCOME_ORDER, range=OUTCOME_COLORS),
                        legend=alt.Legend(orient="top", columns=3)),
        order=alt.Order("outcome_rank:Q"),
        tooltip=[alt.Tooltip(f"{cat_col}:N", title=cat_title), alt.Tooltip("species:N", title="Species"),
                 alt.Tooltip("outcome_group:N", title="Outcome"),
                 alt.Tooltip("share:Q", title="Share", format=".1%"), alt.Tooltip("n:Q", title="Stays", format=",")],
    ).properties(height=max(140, 26 * len(order) + 30))
    if facet_species:
        chart = chart.facet(column=alt.Column("species:N", title=None, sort=SPECIES_ORDER))
    return chart.configure_view(strokeWidth=0)


def waiting_curve_chart(curves: pd.DataFrame, group_title: str, facet_species: bool) -> alt.Chart:
    groups = curves["group"].drop_duplicates().tolist()
    chart = alt.Chart(curves).mark_line(strokeWidth=2).encode(
        x=alt.X("day:Q", title="Days since intake"),
        y=alt.Y("share:Q", title="Share still at the shelter", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("group:N", title=group_title,
                        scale=alt.Scale(domain=groups, range=LEVEL_COLORS[:len(groups)]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("group:N", title=group_title), alt.Tooltip("species:N", title="Species"),
                 alt.Tooltip("day:Q", title="Day"), alt.Tooltip("share:Q", title="Still waiting", format=".1%")],
    ).properties(height=260)
    if facet_species:
        chart = chart.facet(column=alt.Column("species:N", title=None, sort=SPECIES_ORDER))
    return chart.configure_view(strokeWidth=0)


# ----------------------------------------------------------------------------
# Sidebar: one filter row that scopes every chart
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🐾 Who Waits Longest?")
    st.caption("Length of stay at Austin Animal Center, Oct 2013 - May 2025")

    species_sel = st.pills("Species", SPECIES_ORDER, selection_mode="multi", default=SPECIES_ORDER,
                           help="Dogs are always blue and cats always orange in every chart.")
    yr_min, yr_max = int(stays["intake_year"].min()), int(stays["intake_year"].max())
    year_range = st.slider("Intake years", yr_min, yr_max, (yr_min, yr_max),
                           help="Filters by the year the animal *arrived*.")
    intake_types = sorted(stays["intake_type"].unique().tolist())
    intake_sel = st.multiselect("Intake type", intake_types, default=intake_types,
                                help="Stray = found loose; Owner surrender = given up by owner; "
                                     "Public assist = animal control / welfare cases.")
    measure_label_long = st.radio("Measure", list(MEASURES), index=0,
                                  help="'Any outcome' mixes in animals returned to their owner in a day or two "
                                       "and transfers to rescue partners, so it says little about how long an "
                                       "animal waits for a *new home*.")
    measure = MEASURES[measure_label_long]
    min_n = st.slider("Minimum group size", 20, 500, 100, step=10,
                      help="Groups with fewer stays than this are hidden: their medians jump around too much "
                           "to compare fairly.")

    st.divider()
    st.caption("Built by Braeden Waller for CMSE 830 (MSU, Fall 2026). "
               "Data: City of Austin Open Data Portal. "
               "See the **Data & decisions** tab for how the two raw files became one table of stays.")

# Apply the global filters once.
if not species_sel:
    species_sel = SPECIES_ORDER
mask = (stays["species"].isin(species_sel)
        & stays["intake_year"].between(year_range[0], year_range[1])
        & stays["intake_type"].isin(intake_sel if intake_sel else intake_types))
view = stays[mask]
if measure == "adoption":
    measured = view[view["adopted"]]
    measure_label = "days to adoption"
else:
    measured = view[view["has_outcome"]]
    measure_label = "days until leaving"
n_species = len(species_sel)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("Who waits longest at the animal shelter?")
st.markdown(
    "Austin Animal Center is the largest no-kill shelter in the United States, and the city publishes a "
    "record of **every animal that comes in and every animal that leaves**. This app matches those two "
    "records to answer one question for shelter staff, foster volunteers and would-be adopters: "
    "**which dogs and cats wait longest for a home, and what moves the wait?**"
)
if len(view) == 0:
    st.warning("No stays match the current filters. Widen the filters in the sidebar.")
    st.stop()

tab_overview, tab_explore, tab_trends, tab_data, tab_findings = st.tabs(
    ["Overview", "Explore a factor", "Trends over time", "Data & decisions", "Findings & limits"])

# ============================================================================
# TAB 1  -  Overview
# ============================================================================
with tab_overview:
    st.subheader("How to use this app")
    st.markdown(
        "- **Filters on the left** apply to every chart. Start with both species and all years.\n"
        "- **Measure**: the default is *days until adoption* for animals that were adopted. "
        "Switch to *any outcome* to include animals returned to owners, transferred to rescues, or euthanised.\n"
        "- **Explore a factor** lets you compare waits by age, breed, colour, condition and more. "
        "**Trends** shows how the shelter has changed since 2013. "
        "**Data & decisions** explains every cleaning step; **Findings & limits** is the summary."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Shelter stays in view", f"{len(view):,}",
              help="One stay = one intake matched to the outcome that followed it.")
    k2.metric(f"Median {measure_label}", fmt_days(measured["los_days"].median()),
              help="Half of the animals left sooner than this, half later. Medians, because a few very long stays pull the mean up.")
    closed = view[view["has_outcome"]]
    k3.metric("Share adopted", f"{closed['adopted'].mean():.0%}" if len(closed) else "n/a",
              help="Among stays with a recorded outcome.")
    k4.metric("No outcome recorded yet", f"{(~view['has_outcome']).mean():.1%}",
              help=f"Animals still in care on {AS_OF:%b %d, %Y} (the last day in the data), or whose outcome record is missing.")

    st.subheader(f"Median {measure_label}, by age at intake")
    known_age = measured[measured["age_group"] != "Unknown"]
    n_unknown = int((view["age_group"] == "Unknown").sum())
    age_summary = summarise(known_age, ["age_group", "species"])
    age_summary = age_summary[age_summary["n"] >= min_n]
    if len(age_summary):
        st.altair_chart(median_bar_chart(age_summary, "age_group", "Age at intake", measure_label,
                                         AGE_ORDER, n_species), width="stretch")
        longest = age_summary.loc[age_summary["median"].idxmax()]
        shortest = age_summary.loc[age_summary["median"].idxmin()]
        st.markdown(
            f"**In the current selection**, the longest median wait is **{longest['species'].lower()}s aged "
            f"{longest['age_group'].lower()}** ({fmt_days(longest['median'])}, {int(longest['n']):,} stays) and the "
            f"shortest is **{shortest['species'].lower()}s aged {shortest['age_group'].lower()}** "
            f"({fmt_days(shortest['median'])}, {int(shortest['n']):,} stays). The thin grey line on each bar is the "
            f"middle 50% of stays (25th to 75th percentile)."
        )
        if measure == "adoption" and "Cat" in species_sel:
            st.info("Kittens under 6 months look like they wait a long time for adoption, but most arrive only days "
                    "or weeks old and legally cannot be adopted until they are about 8 weeks old and 2 lb. "
                    "Their 'wait' is mostly growing up in foster care. See **Findings & limits**.", icon="ℹ️")
    else:
        st.info("No age group has enough stays under the current filters. Lower the minimum group size.")

    st.subheader("What happens to them")
    st.caption("Share of closed stays ending in each outcome, by age at intake. A short stay is not always good news: "
               "'Return to owner' and 'Euthanasia' are both fast.")
    mix = outcome_mix(view[view["age_group"] != "Unknown"], ["age_group", "species"])
    st.altair_chart(outcome_mix_chart(mix, "age_group", "Age at intake", AGE_ORDER, n_species > 1), width="stretch")
    st.caption(f"{n_unknown:,} stays with an unknown age at intake are left out of these two charts; "
               "choose *Age at intake* in the Explore tab to see them as their own group.")
    with st.expander("Table view of the two charts above"):
        st.dataframe(age_summary.round(1), hide_index=True)
        st.dataframe(mix.pivot_table(index=["species", "age_group"], columns="outcome_group", values="share",
                                     observed=True).reindex(columns=OUTCOME_ORDER).style.format("{:.1%}", na_rep="-"))

# ============================================================================
# TAB 2  -  Explore a factor
# ============================================================================
with tab_explore:
    st.subheader("Compare the wait across any factor")
    st.markdown("Pick a factor. The first chart shows the **median wait** per group, the second shows **how quickly "
                "each group leaves**, and the third shows **where they go**. Groups smaller than the minimum "
                "group size in the sidebar are hidden.")

    c1, c2 = st.columns([2, 1])
    factor_label = c1.selectbox("Factor", list(FACTORS), index=0)
    factor = FACTORS[factor_label]
    top_n = c2.slider("Show top N groups (by number of stays)", 5, 25, 10,
                      disabled=factor not in ("primary_breed", "primary_color"))

    expl = measured.copy()
    if factor == "primary_breed":
        expl = expl[expl["species"] == "Dog"]
        if "Dog" not in species_sel:
            st.warning("Breed comparison is shown for dogs only (cats are almost all 'Domestic Shorthair'). "
                       "Add dogs in the sidebar to see it.")
    order = FACTOR_ORDER.get(factor)
    if factor in ("primary_breed", "primary_color"):
        keep_levels = expl[factor].value_counts().head(top_n).index.tolist()
        expl = expl[expl[factor].isin(keep_levels)]

    fac_summary = summarise(expl, [factor, "species"])
    fac_summary = fac_summary[fac_summary["n"] >= min_n]
    n_sp_here = fac_summary["species"].nunique()

    if len(fac_summary) == 0:
        st.info("No group is large enough under the current filters. Lower the minimum group size or widen the filters.")
    else:
        st.markdown(f"#### Median {measure_label} by {factor_label.lower()}")
        st.altair_chart(median_bar_chart(fac_summary, factor, factor_label, measure_label, order, n_sp_here),
                        width="stretch")
        hi = fac_summary.loc[fac_summary["median"].idxmax()]
        lo = fac_summary.loc[fac_summary["median"].idxmin()]
        st.markdown(f"Longest: **{hi[factor]}** ({hi['species'].lower()}s, median {fmt_days(hi['median'])}, "
                    f"n = {int(hi['n']):,}). Shortest: **{lo[factor]}** ({lo['species'].lower()}s, median "
                    f"{fmt_days(lo['median'])}, n = {int(lo['n']):,}).")

        # --- Still-waiting curves -------------------------------------------
        st.markdown(f"#### How quickly do they leave? Share still at the shelter, day by day")
        st.caption("This curve uses **every stay** (not just adopted animals) and counts animals with no recorded "
                   "outcome as 'still waiting' only for the days we actually observed them, rather than dropping them. "
                   "Pick up to four groups to compare.")
        if order is None:   # no natural order: offer the biggest groups first
            by_size = fac_summary.groupby(factor)["n"].sum().sort_values(ascending=False).index.tolist()
            levels_available = by_size
        else:
            levels_available = [l for l in order if l in set(fac_summary[factor])]
        default_levels = levels_available[:3]
        chosen = st.multiselect("Groups to compare", levels_available, default=default_levels, max_selections=4)
        curve_rows = []
        curve_src = view[view["species"] == "Dog"] if factor == "primary_breed" else view
        for sp in [s for s in SPECIES_ORDER if s in species_sel]:
            for lvl in chosen:
                g = curve_src[(curve_src["species"] == sp) & (curve_src[factor].astype(str) == lvl)]
                if len(g) < min_n:
                    continue
                share = still_waiting_curve(g["los_lower_bound_days"].values, g["has_outcome"].values)
                curve_rows.append(pd.DataFrame({"day": np.arange(len(share)), "share": share,
                                                "group": lvl, "species": sp}))
        if curve_rows:
            curves = pd.concat(curve_rows, ignore_index=True)
            st.altair_chart(waiting_curve_chart(curves, factor_label, n_sp_here > 1), width="stretch")
            at30 = curves[curves["day"] == 30].sort_values("share", ascending=False)
            if len(at30):
                top = at30.iloc[0]
                st.markdown(f"After **30 days**, **{top['share']:.0%}** of *{top['group']}* {top['species'].lower()}s "
                            f"were still at the shelter, the highest of the groups shown.")
        else:
            st.info("Choose at least one group with enough stays to draw a curve.")

        # --- Outcome mix ----------------------------------------------------
        st.markdown(f"#### Where do they go? Outcome mix by {factor_label.lower()}")
        mix_src = view[view["species"] == "Dog"] if factor == "primary_breed" else view
        if factor in ("primary_breed", "primary_color"):
            mix_src = mix_src[mix_src[factor].isin(keep_levels)]
        fac_mix = outcome_mix(mix_src, [factor, "species"])
        sizes = fac_mix.groupby([factor, "species"])["n"].sum().reset_index()
        big = sizes[sizes["n"] >= min_n][[factor, "species"]]
        fac_mix = fac_mix.merge(big, on=[factor, "species"])
        if len(fac_mix):
            st.altair_chart(outcome_mix_chart(fac_mix, factor, factor_label, order, fac_mix["species"].nunique() > 1),
                            width="stretch")
        with st.expander("Table view"):
            st.dataframe(fac_summary.round(1), hide_index=True)
            if len(fac_mix):
                st.dataframe(fac_mix.pivot_table(index=["species", factor], columns="outcome_group", values="share",
                                                 observed=True).reindex(columns=OUTCOME_ORDER)
                             .style.format("{:.1%}", na_rep="-"))

# ============================================================================
# TAB 3  -  Trends over time
# ============================================================================
with tab_trends:
    st.subheader("How the shelter has changed, 2013 - 2025")

    st.markdown("#### Animals arriving each month")
    monthly = view.groupby(["intake_month_start", "species"], observed=True).size().rename("intakes").reset_index()
    monthly["species"] = monthly["species"].astype(str)
    covid = pd.DataFrame({"date": [pd.Timestamp("2020-03-15")], "label": ["COVID-19 shutdown"]})
    line = alt.Chart(monthly).mark_line(strokeWidth=2).encode(
        x=alt.X("intake_month_start:T", title=None),
        y=alt.Y("intakes:Q", title="Intakes per month"),
        color=species_color(n_species),
        tooltip=[alt.Tooltip("intake_month_start:T", title="Month", format="%b %Y"),
                 alt.Tooltip("species:N", title="Species"), alt.Tooltip("intakes:Q", title="Intakes", format=",")],
    )
    rule = alt.Chart(covid).mark_rule(color="#898781", strokeDash=[4, 3]).encode(x="date:T")
    text = alt.Chart(covid).mark_text(align="left", dx=4, dy=-110, color="#52514e").encode(x="date:T", text="label:N")
    st.altair_chart((line + rule + text).properties(height=280).configure_view(strokeWidth=0), width="stretch")
    st.caption("Cat intakes spike every May - July ('kitten season'). Intakes fell by roughly half when the shelter "
               "closed to walk-in strays in March 2020 and have not returned to pre-pandemic levels.")

    st.markdown(f"#### Median {measure_label}, by year of intake")
    hide_recent = st.toggle("Exclude intakes with less than 180 days of follow-up", value=False,
                            help="Animals that arrived in the last 6 months of the data have not all left yet. "
                                 "Only the fast ones have, so the most recent year looks artificially quick.")
    trend_src = measured
    if hide_recent:
        trend_src = trend_src[trend_src["intake_dt"] <= AS_OF - pd.Timedelta(days=180)]
    yearly = summarise(trend_src, ["intake_year", "species"])
    yearly = yearly[yearly["n"] >= min_n]
    yearly["intake_year"] = yearly["intake_year"].astype(int)
    yline = alt.Chart(yearly).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=50)).encode(
        x=alt.X("intake_year:O", title="Year of intake"),
        y=alt.Y("median:Q", title=f"Median {measure_label}"),
        color=species_color(n_species),
        tooltip=[alt.Tooltip("intake_year:O", title="Year"), alt.Tooltip("species:N", title="Species"),
                 alt.Tooltip("median:Q", title="Median days", format=".1f"), alt.Tooltip("n:Q", title="Stays", format=",")],
    ).properties(height=280).configure_view(strokeWidth=0)
    st.altair_chart(yline, width="stretch")
    if not hide_recent and year_range[1] >= 2025:
        st.warning("The drop in 2025 is **not** real: the data ends on May 5, 2025, so only animals that left quickly "
                   "have an outcome yet. Flip the toggle above to see the fair comparison.", icon="⚠️")
    st.caption("Dogs: the median wait for adoption roughly tripled between 2019 and 2023, matching the nationwide "
               "post-pandemic capacity crisis in dog shelters. Cats did not show the same jump.")

    st.markdown("#### Live-release rate by year")
    st.caption("Share of closed stays that ended alive (adoption, return to owner, transfer). "
               "Austin has been above the 90% 'no-kill' threshold for the whole period.")
    closed_v = view[view["has_outcome"]].copy()
    closed_v["live"] = closed_v["outcome_group"].isin(["Adoption", "Return to owner", "Transfer to partner"])
    lr = closed_v.groupby(["intake_year", "species"], observed=True)["live"].agg(rate="mean", n="size").reset_index()
    lr = lr[lr["n"] >= min_n]
    lr["species"] = lr["species"].astype(str)
    lr_chart = alt.Chart(lr).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=50)).encode(
        x=alt.X("intake_year:O", title="Year of intake"),
        y=alt.Y("rate:Q", title="Live-release rate", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0.8, 1.0])),
        color=species_color(n_species),
        tooltip=[alt.Tooltip("intake_year:O", title="Year"), alt.Tooltip("species:N", title="Species"),
                 alt.Tooltip("rate:Q", title="Live release", format=".1%"), alt.Tooltip("n:Q", title="Stays", format=",")],
    ).properties(height=240).configure_view(strokeWidth=0)
    st.altair_chart(lr_chart, width="stretch")
    st.caption("Note the y-axis starts at 80%, not 0%, to make year-to-year differences visible.")
    with st.expander("Table view"):
        st.dataframe(yearly.round(1), hide_index=True)
        st.dataframe(lr.round(3), hide_index=True)

# ============================================================================
# TAB 4  -  Data & decisions
# ============================================================================
with tab_data:
    r = report
    st.subheader("Where the data comes from")
    st.markdown(
        f"Two public tables from the [City of Austin Open Data Portal](https://data.austintexas.gov): "
        f"**Intakes** ({r['raw_rows']['intakes']:,} rows) and **Outcomes** ({r['raw_rows']['outcomes']:,} rows), "
        f"covering {r['date_range']['intakes'][0][:10]} to {r['date_range']['outcomes'][1][:10]}. "
        f"Each raw row is one *event*. The unit of analysis in this app is a **stay**: one intake matched to the "
        f"outcome that ended it. The processed table has **{r['stays']['total']:,} stays** for "
        f"**{r['stays']['animals']:,} individual dogs and cats**."
    )

    st.subheader("From two raw files to one table of stays")
    steps = pd.DataFrame([
        ["Raw rows", f"{r['raw_rows']['intakes']:,} intakes / {r['raw_rows']['outcomes']:,} outcomes", "-"],
        ["Exact duplicate rows removed", f"{r['exact_duplicate_rows']['intakes']} / {r['exact_duplicate_rows']['outcomes']}",
         "Identical copies of another row."],
        ["Same animal, same minute removed", f"{r['same_animal_same_time_duplicates']['intakes']} / "
         f"{r['same_animal_same_time_duplicates']['outcomes']}", "Double entries where one field was retyped."],
        ["Not a dog or cat removed", f"{r['rows_dropped_not_dog_or_cat']:,} intakes",
         "Mostly wildlife brought in for rabies testing, plus a few birds and livestock."],
        ["Intakes matched to an outcome", f"{r['stays']['with_outcome']:,}", "Walking each animal's history in time order."],
        ["Intakes with no outcome yet", f"{r['stays']['open']:,}", "Kept, flagged, and treated as censored (see below)."],
        ["Outcomes with no earlier intake", f"{r['matching']['outcomes_before_first_intake']:,}",
         "Animals already in the shelter before records begin in Oct 2013. Not used."],
    ], columns=["Step", "Count", "Why"])
    st.dataframe(steps, hide_index=True, width="stretch")

    with st.expander("1. Two files, two date formats (and a stray time zone)", expanded=False):
        st.markdown(
            f"Intakes store dates as `05/05/2025 07:22:00 AM`; outcomes store them as ISO `2025-05-05T07:22:00`. "
            f"**{r['outcome_rows_with_utc_offset']:,} outcome rows** additionally carry a UTC offset such as `-05:00`, "
            f"and *every one of them* is stamped exactly midnight, which means the time of day was never recorded. "
            f"Decision: parse each file with its own explicit format, strip the offset, and treat every timestamp as "
            f"Austin local time (which the intake file already is). Zero dates failed to parse. "
            f"{r['outcomes_stamped_midnight']:,} outcomes have a date but no time, so individual stays can be off by "
            f"up to a day; medians over thousands of stays are unaffected."
        )

    with st.expander("2. Ages are text, and some are impossible"):
        st.markdown(
            "Age arrives as strings like `2 years`, `3 weeks`, `1 month`, with 55 distinct values. I converted each "
            "to days (year = 365.25, month = 30.44, week = 7), then into four age bands that matter for adoption "
            "(under 6 months, 6 months - 2 years, 2 - 7 years, 7+ years)."
        )
        st.markdown(
            f"- **`0 years`** appears {r['age_zero_years_rows']:,} times. For 80% of them the recorded date of birth "
            f"equals the intake date, i.e. staff did not estimate an age. Treated as **unknown**, not as newborn.\n"
            f"- **Negative ages** (`-1 years` to `-4 years`) appear {r['age_negative_rows']} times: the date of birth was "
            f"typed in after the intake date. Treated as **unknown**.\n"
            f"- {r['age_unknown_after_parse']:,} stays end up with an unknown age. They stay in the data as their own "
            f"'Unknown' group rather than being imputed: an animal whose age staff could not guess is a different "
            f"kind of animal (often a neonate), and guessing would hide that."
        )
        st.dataframe(pd.DataFrame({"Raw text": ["1 week", "1 month", "1 year", "7 years", "0 years", "-1 years"],
                                   "Days": ["7", "30.4", "365.3", "2556.8", "unknown", "unknown"],
                                   "Age band": ["Under 6 months", "Under 6 months", "6 months - 2 years", "7+ years",
                                                "Unknown", "Unknown"]}), hide_index=True)

    with st.expander("3. Matching each intake to the right outcome (the step that matters most)"):
        st.markdown(
            f"**{r['stays']['animals_with_repeat_stays']:,} animals came in more than once**; one dog has "
            f"{r['stays']['max_stays_one_animal']} intakes. Joining the two files on Animal ID alone pairs every intake "
            f"with every outcome: it produces **{r['naive_join']['rows']:,} rows** instead of {r['stays']['total']:,}, "
            f"and **{r['naive_join']['pairs_with_outcome_before_intake']:,} of those pairs have the outcome *before* the "
            f"intake**. Lengths of stay computed that way are nonsense.\n\n"
            "Decision: list every event for each animal in time order. Each intake opens a new stay, and the first "
            "outcome after it closes that stay. Any further outcome before the next intake "
            f"({r['matching']['extra_outcomes_in_same_stay']} cases) is a duplicate record and is ignored. "
            "With this rule **no stay has a negative length**."
        )
        three = stays.groupby("animal_id")["has_outcome"].agg(["size", "all"])
        example_id = three[(three["size"] == 3) & three["all"]].index.min()
        ex = stays[stays["animal_id"] == example_id][["animal_id", "stay_number", "intake_dt", "intake_type",
                                                       "outcome_dt", "outcome_group", "los_days"]]
        n_out = int(ex["has_outcome"].sum()) if "has_outcome" in ex else len(ex)
        st.markdown(f"Example: animal **{example_id}** has three stays and three outcomes. The naive join would give "
                    f"it 3 x {n_out} = {3 * n_out} rows, six of them pairing an intake with the wrong outcome.")
        st.dataframe(ex.assign(los_days=ex["los_days"].round(1)), hide_index=True)

    with st.expander("4. Stays with no outcome recorded (right-censoring)"):
        open_by_year = pd.DataFrame({"Intake year": list(r["stays"]["open_by_intake_year"].keys()),
                                     "Open stays": list(r["stays"]["open_by_intake_year"].values())})
        c1, c2 = st.columns([1, 1])
        c1.markdown(
            f"**{r['stays']['open']:,} intakes ({r['stays']['open'] / r['stays']['total']:.1%})** have no outcome "
            f"after them. Most arrived in 2024 - 25 and were simply still in care when the data ends on "
            f"{AS_OF:%B %d, %Y}; a few hundred older ones are missing records.\n\n"
            "Dropping them is the easy mistake: it silently removes exactly the animals that wait longest. "
            "Decisions:\n"
            "- Median waits use only closed stays, but the app always reports the share with no outcome.\n"
            "- The 'still waiting' curves use every stay and treat open ones as *censored*: they count as waiting "
            "only for the days actually observed.\n"
            "- The **Trends** tab has a toggle that excludes intakes with under 180 days of follow-up, because the "
            "last few months otherwise look artificially fast."
        )
        c2.altair_chart(alt.Chart(open_by_year).mark_bar(color="#2a78d6", cornerRadiusEnd=3).encode(
            x=alt.X("Intake year:O", title="Intake year"), y=alt.Y("Open stays:Q", title="Stays with no outcome"),
            tooltip=["Intake year:O", "Open stays:Q"]).properties(height=220).configure_view(strokeWidth=0),
            width="stretch")

        st.markdown("**Demonstration.** Median days to adoption by month of intake, with and without the follow-up rule:")
        recent = stays[(stays["adopted"]) & (stays["intake_dt"] >= "2023-01-01")]
        a = recent.groupby("intake_month_start")["los_days"].median().rename("All adopted animals")
        b = (recent[recent["intake_dt"] <= AS_OF - pd.Timedelta(days=180)]
             .groupby("intake_month_start")["los_days"].median().rename("180+ days of follow-up only"))
        demo = pd.concat([a, b], axis=1).reset_index().melt("intake_month_start", var_name="rule", value_name="median").dropna()
        demo_chart = alt.Chart(demo).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40)).encode(
            x=alt.X("intake_month_start:T", title="Month of intake"),
            y=alt.Y("median:Q", title="Median days to adoption"),
            color=alt.Color("rule:N", title=None, scale=alt.Scale(range=["#eb6834", "#2a78d6"]),
                            legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("intake_month_start:T", format="%b %Y", title="Month"), "rule:N",
                     alt.Tooltip("median:Q", format=".1f")],
        ).properties(height=240).configure_view(strokeWidth=0)
        st.altair_chart(demo_chart, width="stretch")
        st.caption("The orange line collapses toward zero in the last months only because slow adoptions have not "
                   "happened yet. The blue line stops where the data can no longer be trusted.")

    with st.expander("5. Missing values: informative, structural, and genuinely missing"):
        st.markdown(
            "Missingness here is **not random**, so nothing was imputed. Three different kinds:"
        )
        nm = pd.DataFrame({"Intake type": list(r["missing_name_share_by_intake_type"].keys()),
                           "Share with no name": list(r["missing_name_share_by_intake_type"].values())})
        sub = pd.DataFrame({"Outcome type": list(r["missing_subtype_share_by_outcome"].keys()),
                            "Share with no subtype": list(r["missing_subtype_share_by_outcome"].values())})
        c1, c2 = st.columns(2)
        c1.markdown(f"**Informative.** `Name` is missing for {r['raw_missing_intakes']['Name'] / r['raw_rows']['intakes']:.0%} "
                    "of intakes, but far more often for strays than for owner surrenders. A missing name means "
                    "'nobody knew this animal', so I kept it as a feature (*Arrived with a name*) instead of filling it.")
        c1.dataframe(nm.style.format({"Share with no name": "{:.0%}"}), hide_index=True)
        c2.markdown(f"**Structural.** `Outcome Subtype` is empty for {r['raw_missing_outcomes']['Outcome Subtype'] / r['raw_rows']['outcomes']:.0%} "
                    "of outcomes because most outcome types simply have no subtype: transfers always name a partner, "
                    "returns to owner never do. Not a data problem; left as is.")
        c2.dataframe(sub.style.format({"Share with no subtype": "{:.0%}"}), hide_index=True)
        st.markdown(f"**Genuinely missing.** {r['raw_missing_outcomes']['Outcome Type']} outcomes have no type at all. "
                    "They are grouped as 'Other / unknown' and excluded from adoption rates.")

    with st.expander("6. Consolidating categories that changed over the years"):
        cg = pd.Series(r["condition_groups"]).rename("Grouped as").rename_axis("Raw intake condition").reset_index()
        og = pd.Series(r["outcome_groups"]).rename("Grouped as").rename_axis("Raw outcome type").reset_index()
        st.markdown("The shelter added intake-condition labels over time (`Med Attn`, `Med Urgent`, `Parvo`, "
                    "`Panleuk`, `Space`...), several with under 20 rows. Comparing 20 labels is noise, so they were "
                    "grouped by what they mean for the animal. Outcomes were grouped the same way (e.g. `Rto-Adopt`, an "
                    "owner adopting back their own surrendered pet, counts as *Return to owner*).")
        c1, c2 = st.columns(2)
        c1.dataframe(cg, hide_index=True, height=300)
        c2.dataframe(og, hide_index=True, height=300)
        st.markdown("Breeds (3,000 distinct strings like `Labrador Retriever/Pit Bull Mix`) were reduced to the "
                    "first listed breed plus a mixed-breed flag; colours (660 strings) to the first listed colour.")

    with st.expander("7. What I assessed and chose *not* to do"):
        st.markdown(
            "- **No imputation.** Every missing value above is informative or structural; filling it would invent data.\n"
            "- **No outlier removal.** "
            f"{r['stays']['los_over_365']} stays lasted over a year. They are real (long-term fosters, court holds), "
            "so instead of deleting them every summary uses **medians and quartiles**, which they cannot distort.\n"
            "- **No scaling or encoding.** There is no model yet; the midterm is descriptive. Both will be needed "
            "for the final project's survival model and are noted there.\n"
            "- **No second data source.** I considered joining daily weather or holidays to intake dates but "
            "could not state a question it would answer better than the shelter's own records, so I left it out "
            "rather than integrate for its own sake.\n"
            "- **Raw files not committed.** They are 45 MB and change daily on the portal; "
            "`src/download_data.py` fetches them and `src/prepare_data.py` rebuilds everything above in two seconds."
        )

# ============================================================================
# TAB 5  -  Findings & limits
# ============================================================================
with tab_findings:
    st.subheader("What the data says (all years, both species)")
    full = stays
    ad = full[full["adopted"]]
    dogs_ad, cats_ad = ad[ad["species"] == "Dog"], ad[ad["species"] == "Cat"]
    pit = dogs_ad["breed_raw"].str.contains("Pit Bull|Staffordshire|American Bulldog", regex=True)
    kitten = cats_ad["age_group"] == "Under 6 months"
    senior_dogs = full[(full["species"] == "Dog") & (full["age_group"] == "7+ years") & full["has_outcome"]]
    d19 = dogs_ad[dogs_ad["intake_year"] == 2019]["los_days"].median()
    d23 = dogs_ad[dogs_ad["intake_year"] == 2023]["los_days"].median()
    black_dog = dogs_ad[dogs_ad["primary_color"] == "Black"]["los_days"].median()
    other_dog = dogs_ad[dogs_ad["primary_color"] != "Black"]["los_days"].median()
    black_cat = cats_ad[cats_ad["primary_color"] == "Black"]["los_days"].median()
    other_cat = cats_ad[cats_ad["primary_color"] != "Black"]["los_days"].median()
    intact_cat = cats_ad[cats_ad["fixed_at_intake"] == "Intact"]["los_days"].median()
    fixed_cat = cats_ad[cats_ad["fixed_at_intake"] == "Fixed"]["los_days"].median()

    st.markdown(
        f"1. **The typical stay is short, but the typical *adoption* is not.** Half of all dogs and cats leave within "
        f"{full['los_days'].median():.0f} days, mostly returned to owners or transferred. Among animals that are "
        f"adopted, the median wait is {dogs_ad['los_days'].median():.0f} days for dogs and "
        f"{cats_ad['los_days'].median():.0f} days for cats.\n"
        f"2. **Breed is the biggest single factor for dogs.** Adopted pit-bull-type dogs wait a median of "
        f"**{dogs_ad[pit]['los_days'].median():.0f} days** versus **{dogs_ad[~pit]['los_days'].median():.0f} days** "
        f"for every other dog, and the gap holds within every age band.\n"
        f"3. **Senior dogs are the group most at risk.** Only {senior_dogs['adopted'].mean():.0%} of dogs aged 7+ are "
        f"adopted ({senior_dogs['outcome_group'].eq('Return to owner').mean():.0%} go home to an owner), their "
        f"euthanasia share ({senior_dogs['outcome_group'].eq('Euthanasia').mean():.1%}) is the highest of any dog age "
        f"band, and those that are adopted wait {dogs_ad[dogs_ad['age_group'] == '7+ years']['los_days'].median():.0f} days.\n"
        f"4. **Kittens 'wait' {cats_ad[kitten]['los_days'].median():.0f} days, but mostly to grow up.** Cats aged 6 months - "
        f"2 years wait {cats_ad[cats_ad['age_group'] == '6 months - 2 years']['los_days'].median():.0f}. Intact cats wait "
        f"{intact_cat:.0f} days versus {fixed_cat:.0f} for cats already spayed or neutered: the surgery queue is part of the wait.\n"
        f"5. **Dog waits tripled after the pandemic.** Median days to adoption for dogs went from {d19:.0f} (2019 intakes) "
        f"to {d23:.0f} (2023 intakes). Cats did not show the same rise.\n"
        f"6. **'Black dog syndrome' does not show up here.** Black dogs wait {black_dog:.0f} days versus {other_dog:.0f} "
        f"for other colours. Black cats do wait a little longer ({black_cat:.0f} vs {other_cat:.0f} days)."
    )

    st.subheader("Limitations and alternative explanations")
    st.markdown(
        "- **One shelter.** Austin is a large, well-funded no-kill shelter with a 96% live-release rate. "
        "Waits and outcomes elsewhere will differ.\n"
        "- **Days to adoption is only observed for animals that were adopted.** A group with a short median might be "
        "one where the slow animals are transferred or euthanised instead. Always read the wait alongside the outcome mix.\n"
        "- **Transfers hide outcomes.** 29% of stays end in a transfer, mostly to Austin Pets Alive!, and what "
        "happened next is not in this data.\n"
        "- **Confounding.** *Arrived with a name* roughly doubles the wait, but named animals are mostly owner "
        "surrenders, which are a different population from strays. None of the gaps here are causal estimates.\n"
        "- **Foster care counts as 'in care'.** An animal in a foster home for two months counts as a 60-day stay even "
        "though it never sat in a kennel.\n"
        "- **Ages are staff estimates**, usually rounded to whole years for adults; breeds are visual guesses.\n"
        "- **Date-only timestamps** for 3,788 outcomes and a data cut-off of May 2025 add small, known errors "
        "described in Data & decisions."
    )

    st.subheader("Next steps (final project)")
    st.markdown(
        "- Fit a **survival model** (Cox proportional hazards or an accelerated-failure-time model) that handles the "
        "censoring properly and estimates each factor's effect *holding the others fixed*, which the comparisons here cannot do.\n"
        "- Turn it into a **long-stay risk flag** at intake so staff can start marketing and foster outreach on day one.\n"
        "- Add shelter **capacity** (animals in care each day, derivable from these same two files) as a predictor.\n"
        "- Compare Austin with other cities that publish the same two tables (Dallas, Bloomington, Sonoma County)."
    )
