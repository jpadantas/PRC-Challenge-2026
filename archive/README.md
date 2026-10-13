# Archive: scripts behind the leaderboard submissions

These are the training, blending and submission scripts written during the competition, kept
because the eligibility rules ask for all produced source code and because they document how the
earlier leaderboard submissions were built, in particular `v104`, the anchor of the final
submission. `SUBMISSIONS_LOG.md` records what each submission contained and what it scored;
read its header first.

These scripts do **not** run from a fresh clone:

- many read trained models or intermediate predictions from `models/` and `scratch/`, which are not
  versioned (about 12 GB);
- they import modules as `src.<name>`, and those modules now live in this folder;
- several of them upload to the team's OpenSky bucket when run.

Their comments use the names given during the competition ("champion" or "record" meaning the
best public score so far), and some interpretations written at the time were later revised.
The paper in `joas_paper/main.pdf` is the authoritative description.

The runnable pipeline, including the steps from `v104` and `v109` to the final file, is in `src/`.

`v104` was built in `auto_sequential_submit_midnight_v101_to_v105.py` (slot 4) with the score-space
quadratic program of Eq. (7) of the paper: weights summing to one and bounded to [-0.2, 1.2] over
`v98`, `v99`, `v100`, `v101`, `v102` and `v103`, which gave
`1.0044 v99 + 0.5008 v103 + 0.0948 v100 - 0.20 v101 - 0.20 v102 - 0.20 v98`
(2026-10-02 entry of `SUBMISSIONS_LOG.md`). The 36 departures with `v60 > 15,000 s` keep the `v60`
prediction.
