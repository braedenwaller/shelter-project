# Generative AI use disclosure

<!-- Braeden: read this carefully and edit anything that does not match what you actually did.
     The course requires this to be accurate. Delete this comment before submitting. -->

**Tools used**

- **Claude (Anthropic), claude.ai chat** - project scoping and planning: explaining the rubric, brainstorming ten
  candidate topics, ranking them against the rubric, and drafting a day-by-day work plan with step-by-step setup
  instructions.
- **Claude Code (Anthropic, model Claude Fable 5.1), desktop app** - implementation: downloading the data, exploratory
  analysis of the raw files, writing `src/prepare_data.py`, `app.py`, the EDA notebook, this README and the
  presentation outline, and running / visually checking the Streamlit app.

**Representative prompts**

- "Explain the scope of the midterm, what's expected, what I need to turn in, and what the final product looks like."
- "Give me 10 directions I could go in that are genuinely useful, fit the rubric well, and would get good feedback from peers."
- "Please make those steps much more detailed, I don't know how to do this stuff."
- "Create this app for me for my class. Here's the conversation I've had about it and the project materials."
- Follow-ups during the build such as checking what the `0 years` ages mean, why a plain join on Animal ID produces
  outcomes dated before intakes, and how to treat stays with no recorded outcome.

**How I checked the output**

- Every count quoted in the app and README is computed by `src/prepare_data.py` and written to
  `data/processed/quality_report.json`; I re-ran the pipeline and compared the printed counts with the raw files
  (row counts, duplicate counts, date ranges) in the notebook.
- I checked the intake-to-outcome matching on individual animals with several visits (e.g. the three-stay example
  shown in the app) to confirm that each intake pairs with the next outcome and that no stay has a negative length.
- I read through `app.py` and `src/prepare_data.py` line by line so that I can explain each step, and tested each
  sidebar filter and tab in the running app.
- I cross-checked the headline numbers against the tables in the notebook (medians by age band, breed, year).

**My own contribution**

- Choosing the topic (an animal-shelter length-of-stay story) and the framing of the question and audience after
  rejecting earlier ideas (Apple Watch running data; flight delays; H-1B salaries).
- Deciding the analytic choices with the AI's input: dogs and cats only, the "days to adoption" measure as the
  default, treating `0 years` as unknown rather than newborn, grouping the intake conditions, keeping unfinished
  stays as censored rather than dropping them, and not imputing informative missingness.
- Interpreting the results: the kitten finding (they wait to reach adoption age, not because they are unwanted), the
  senior-dog risk, the post-pandemic rise in dog waits, and the limitations listed in the app.
- The presentation and answering questions about it.

I understand that I am responsible for every line of code and every conclusion in this submission.
