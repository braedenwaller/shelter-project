# 7-minute presentation plan

**Before you start:** open a terminal in the project folder and run `streamlit run app.py`. Set the browser zoom to
110-125% so the back row can read it. Collapse nothing. Have the **Overview** tab showing with both species, all years.

## 0:00 - 1:00  Problem, audience, data

- "Austin Animal Center is the biggest no-kill shelter in the country, and the city publishes every intake and every
  outcome, 174,000 rows each, since 2013."
- "My question: **which dogs and cats wait longest for a home, and what moves the wait?** The users are shelter
  staff and foster volunteers deciding where to put their effort, plus adopters who want to know who's overlooked."
- "The unit of analysis is a **stay**: one intake matched to the outcome that ended it. Getting to that table is most
  of the work."

## 1:00 - 3:00  Data-prep and EDA decisions (open the *Data & decisions* tab)

Pick **three** of these; don't try to do all seven.

1. **Matching (the big one, section 3).** "13,000 animals came in more than once, one dog 33 times. A plain join on
   Animal ID gives 211,000 rows instead of 164,000, and 25,000 of those pairs have the outcome *before* the intake.
   So I sort every event per animal in time order: each intake opens a stay, the next outcome closes it. Zero
   negative stays." Show the three-stay example table.
2. **Censoring (section 4).** "1,500 intakes have no outcome yet. If you drop them you silently remove the animals
   that wait longest." Show the demo chart: orange line collapses in the last months. "So medians use closed stays,
   the curves use a Kaplan-Meier estimate, and the trend tab has a follow-up toggle."
3. **Ages (section 2).** "Ages are text. `0 years` isn't a newborn: for 80% of them the date of birth equals the
   intake date. Ten animals have negative ages. Both become 'unknown', which I keep as its own group instead of imputing."
4. (If time) **Two date formats + time zones**, or **informative missingness** (missing name = stray).

Also say once: "Mean stay is 22 days, median is 6: the distribution is very skewed, so every chart uses medians and quartiles."

## 3:00 - 6:00  Live demo and key finding

1. **Overview tab.** Point at the metrics. "Half of all animals leave in 6 days, but that's mostly returns to owner
   and transfers. For animals that get *adopted*, the median is 9 days for dogs and 24 for cats."
   Headline chart: "Senior dogs and senior cats wait longest; kittens look slow but they're growing up to 8 weeks."
   Outcome mix: "Short isn't always good: 46% of senior dogs go back to an owner, but their euthanasia share is the
   highest of any dog age band."
2. **Explore tab -> Primary breed (dogs).** "Pit bull types: 28 days vs 8 for everyone else. Biggest single factor."
   Scroll to the still-waiting curve with Pit Bull vs Chihuahua vs Labrador selected: "After 30 days a third of pit
   bulls are still here."
3. **Explore tab -> Primary colour.** "Black dog syndrome? Not here: 9 days vs 9. Black cats, a little: 29 vs 24."
4. **Trends tab.** Median wait by year: "Dogs tripled from 2019 to 2023, the national post-pandemic shelter crisis."
   Then: "Notice 2025 looks great. It isn't. The data ends May 2025, so only fast adoptions have happened."
   Flip the toggle. "That's the censoring decision in action."

## 6:00 - 7:00  One limitation and one next step

- **Limitation:** "Days to adoption is only measured for animals that were adopted. A group with a short median
  might be one where the slow animals got transferred or euthanised instead. That's why every wait chart has the
  outcome mix underneath it."
- **Next step:** "For the final: a Cox survival model that handles the censoring properly and estimates each factor's
  effect holding the others fixed, then a long-stay risk flag at intake."

## Likely questions and short answers

- *Why medians?* Stays are heavily right-skewed (mean 22, median 6, max 1,900 days). Medians with quartile bars
  show the typical animal without letting a few multi-year fosters dominate.
- *Why dogs and cats only?* "Other" is mostly wildlife (bats, raccoons) euthanised for rabies testing. They would
  distort both the wait and the outcome mix and aren't part of the adoption question.
- *How does the matching code work?* Concatenate intakes and outcomes into one event table, sort by animal and
  time (intakes before outcomes on ties), cumulative-sum the intake flags per animal to number the stays, then
  take the first outcome in each stay. `pandas` `groupby().cumsum()` and `cumcount()`.
- *What is the "still waiting" curve?* A Kaplan-Meier estimate: at each day, the share of animals still present
  among those who could still be observed. Open stays count as present only for the days actually observed.
- *Did you integrate a second source?* I assessed weather and holidays and could not state a question they would
  answer better than the shelter's own records, so no, and I say so in the app.
- *Why not impute age?* The missing ages are informative (a neonate staff couldn't age) and only 1,200 rows;
  imputing would hide that and the app treats "Unknown" as its own group.
- *Is kittens' long wait a problem?* Mostly no: kittens can't be adopted until about 8 weeks and 2 lb, so most of
  their "wait" is in foster growing up. The intact-vs-fixed gap (30 vs 15 days) points the same way.
- *Foster?* Time in foster counts as time "in care" in this data; I can't separate kennel time from foster time.
- *Why is color blue for dogs everywhere?* Color follows the entity in every chart so a reader learns it once.
