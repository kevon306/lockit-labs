# LOCK IT Labs: project notes for Claude Code

NBA player-prop research app. Goal: calibrated over/under probabilities that beat the market,
proven with a timestamped prediction ledger. It is a research tool; it never places bets.

## Roadmap
Read `ROADMAP.md` for the launch plan and follow it in order, one week/prompt at a time -
finish and test each step before starting the next, unless the user says otherwise.

## The person you're working with
- Beginner with code. Explain each step plainly, one command at a time, and say what output to expect.
- Mac (Python 3.9 from Apple's tools, in .venv). Avoid Python 3.10+ syntax (no `X | Y` types, no `match`).
- Run commands with the venv: `source .venv/bin/activate` first, or use `.venv/bin/python3`.

## Files
- `pull_props.py`: pulls prop odds from The Odds API (key in `.env`, never print or commit it).
  Free plan: 500 credits/month. Live event odds cost markets x regions; historical costs 10x.
- `db.py`: SQLite at `data/lockit.db`. Tables: odds_snapshots, box_scores, predictions (LOCKED by
  triggers, never edit or delete), entries, entry_legs, corrections (audit log for entry fixes).
- `box_scores.py`: NBA box scores via nba_api (blocked from cloud IPs; works from this Mac).
  Only finished games (WL set) are saved. `grade_lines()` grades saved lines.
- `auto.py` + `setup_automation.sh`: launchd jobs. Morning job 9 AM (box scores + grading),
  snapshot job every 15 min (closing lines ~10-20 min before tip). Logs in `data/logs/`.
- `slip.py`: research slip (data/slip.json) and entry tracker (records entries placed elsewhere).
- `lockit_app.py`: Streamlit dashboard, `streamlit run lockit_app.py`. Theme in `.streamlit/config.toml`.
  Streamlit 1.50: `st.altair_chart` takes `use_container_width`, not `width`.

## Rules
- Never fabricate stats, odds, probabilities, or results. Label market-implied numbers as the market's,
  not LOCK IT predictions. Show missing or stale data as missing or stale.
- No future information in backtests. Predictions are written once with their inputs and never changed.
- Prediction accuracy and money results are reported separately. Show sample sizes.
- Neutral language about betting volume; never encourage more betting or chasing losses.
- Don't touch `data/` or `.env` unless asked. Test changes before saying they work.

## Design (dashboard)
Dark navy (#0E1525 bg, #161F33 panels, #26324D borders), text #E8EDF7 / muted #8A96B0,
blue #1E90FF primary, green #19C37D over/gain, red #F0565B under/loss, gold #E0A458 marks "the line".
Fonts: Bricolage Grotesque (headings), Figtree (body).

## Next up
First model: project minutes, then points per minute, combine with uncertainty into a full
distribution, probabilities for any line. Validate calibration on past box scores (2023-24 onward)
before buying historical odds; then test against no-vig closing lines and CLV.
