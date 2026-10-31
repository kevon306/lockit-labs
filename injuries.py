"""
injuries.py  -  NBA player availability, from ESPN's public (unofficial) injuries feed.

  python3 injuries.py pull   pull the whole league's current injury list, save a snapshot

This is NOT an official NBA or ESPN API - there's no published documentation, no
terms of service, no rate-limit guarantee, and no uptime SLA. It's the same kind of
reverse-engineered endpoint box_scores.py already depends on for box scores, just a
different provider. It could change shape or get blocked without notice.

Verified live before building this: real player names, real dates, real sourcing
(e.g. "Brad Rowland of the Locked On Podcast Network"), checked across many teams
including some with zero current injuries (not just always-populated noise). One
request covers the entire league - no need to hit a separate endpoint per team.

Status vocabulary is whatever ESPN itself reports, not invented. In practice this has
been "Out" and "Day-To-Day" - NBA injury reporting doesn't use the NFL-style
Questionable/Doubtful/Probable tiers, so this app doesn't invent that granularity
either. A team or player missing from a pull means "nothing reported," never
"confirmed healthy" - see ai_tools.injury_status(), which says that explicitly every
time it's used.

Pulled at most a few times a day by auto.py (same morning/afternoon clock windows
already used for odds freshness) - there's no reason to call this more often than
injury news actually changes, and no reason to hammer an unofficial endpoint.
"""

import sys
from datetime import datetime, timezone

import requests

import box_scores
import db

URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch():
    resp = requests.get(URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def flatten(data, pulled_at, team_abbrs):
    """ESPN's response to flat rows, one per reported injury. Teams with no current
    injuries are just absent from the response - not an error, not reported as rows."""
    rows = []
    unmapped_teams = set()
    for team in data.get("injuries", []):
        abbr = team_abbrs.get(team["displayName"])
        if abbr is None:
            unmapped_teams.add(team["displayName"])
            continue
        for inj in team.get("injuries", []):
            athlete = inj.get("athlete") or {}
            details = inj.get("details") or {}
            rows.append({
                "pulled_at": pulled_at,
                "team": abbr,
                "player_name": athlete.get("displayName"),
                "status": inj.get("status"),
                "status_code": (inj.get("type") or {}).get("name"),
                "comment": inj.get("shortComment"),
                "injury_type": details.get("type"),
                "report_date": inj.get("date"),
                "return_date": details.get("returnDate"),
                "espn_injury_id": str(inj.get("id")),
            })
    if unmapped_teams:
        print(f"Note: {len(unmapped_teams)} team name(s) from ESPN didn't match this "
              f"app's own team list, skipped: {', '.join(sorted(unmapped_teams))}")
    return rows


def cmd_pull():
    try:
        data = fetch()
    except requests.RequestException as e:
        sys.exit(f"Couldn't reach ESPN's injuries feed: {e}")

    pulled_at = iso(datetime.now(timezone.utc))
    team_abbrs = box_scores.team_abbreviations()
    rows = flatten(data, pulled_at, team_abbrs)

    if not rows:
        print("No injuries currently reported league-wide (or the feed's shape changed - "
              "worth a manual check if this persists).")
        return

    new_in_db = db.save_injury_reports(rows)
    by_status = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    print(f"Pulled {len(rows)} reported injuries across the league.")
    print(f"  database: {new_in_db} new rows added to the database")
    print(f"  by status: {', '.join(f'{k}: {v}' for k, v in sorted(by_status.items()))}")
    print("\nSample:")
    for r in rows[:8]:
        print(f"  {r['team']:<4} {r['player_name']:<22} {r['status']:<12} {r['comment'] or ''}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] != "pull":
        sys.exit("Use: python3 injuries.py pull")
    cmd_pull()
