# LOCK IT Labs — Launch Roadmap

Work through this one week/prompt at a time. Let each step finish and get tested before starting the next. Claude Code: read this file at the start of a session when asked to continue the roadmap, and follow it in order unless the user says otherwise.

## This week (Oct 26 – Nov 1): foundation

- Delete the exposed Anthropic key and make a new one. Set the GitHub repo to private. Set spending limits in the Anthropic console and The Odds API dashboard.
- Health check: odds key, credits, logs, app.
- Save this plan in the project as `ROADMAP.md` so Claude Code always knows the plan.

## Nov 2 – 8: fix NBA picks

- Top Picks: only positive-edge sides, flag edges over 8 points, confidence tied to beating the closing line.
- Injury feed from official NBA injury reports, repricing the model when a player is out.

## Nov 9 – 15: NBA markets + business paperwork

- Add threes, steals, blocks, turnovers, PRA and combos. Save the extra box score columns and re-pull past seasons.
- File the Georgia LLC (~$100). Book a lawyer consult. Email 2–3 data providers for six-sport licensing quotes. Ask The Odds API about commercial use.

## Nov 16 – 22: multi-sport setup

- Add a sport selector to the app.
- Hook up stats for NHL, college basketball, and NFL.
- Add those sports to Tonight, Lines, and line shopping.

## Nov 23 – Dec 6: the big backtest

- Upgrade to the 5M Odds API plan for one month (~$119).
- Pull historical lines for NBA, NHL, college basketball, and NFL.
- Run the NBA prop and game-line backtests. Schedule the downgrade.

## Dec 7 – 20: more models + database

- Build and backtest game-line models for NHL, college basketball, and NFL.
- Move the database to Supabase.

## Dec 21 – Jan 3: polish + private beta

- Speed improvements, design polish, first-time walkthrough.
- Private beta with 5–10 people.
- Terms of service, privacy policy, and age check, based on the lawyer's advice.
- Start the MLB model.

## Jan 4 – 17: public launch

- Fix beta feedback. Set up a custom domain.
- Upgrade the Odds API to the 100K plan (~$59/mo).
- Launch the free public beta in mid-January, before the NFL divisional round. Picks only for sports that passed.

## Jan 18 – Feb 7: paid features

- Accounts, Stripe payments, alerts.
- Keep growing the live ledger.
- Continue the MLB model.

## Feb 8 – 28: paid launch

- Turn on the paid tier for sports with a passed backtest, licensed data, and a few weeks of live ledger results.
- Backtest MLB.

## March: MLB at opening day

## Costs through February

- Odds API: $30/mo now, ~$59/mo from January
- Backtest month: ~$119 once
- LLC: ~$100, plus ~$50–60/yr
- Lawyer: a few hundred to ~$1,000
- Domain: ~$15/yr
- Supabase: free to $25/mo
- Licensed data: depends on quotes (the big unknown)
