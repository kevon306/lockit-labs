"""
news.py  -  NBA role-change news for tonight's teams, via Claude's web_search tool.

  python3 news.py pull   search tonight's teams for recent role-change news, save it

injuries.py already tracks basic health status (Out/Day-To-Day) from ESPN once a
day. This covers something that feed doesn't: a trade, a new starting lineup, a
coach benching or promoting a player, a suspension, or a player's role changing
after returning from injury - the kind of thing that changes who's actually on the
floor, not just who's healthy.

This is fundamentally different from every other data source in this app: it's an
LLM-written SUMMARY of real, cited web search results, not a number pulled from a
structured feed. Every place this is shown (the Tonight page, the AI Analyst) labels
it that way - a lead to check, never presented with the confidence of a verified
box score or injury status. The search itself is real and cited (Claude's hosted
web_search tool, restricted to known sports-news domains - see ALLOWED_DOMAINS); the
summary wrapped around it can still be wrong or incomplete the way any summary can.

Uses the same ANTHROPIC_API_KEY as the AI Analyst (ai_analyst.py). If that isn't
configured, this is skipped - not faked, not estimated.

Tonight's teams come from The Odds API's free /events endpoint (pull_props.py
already uses it the same way) - no odds credits spent just to get a team list.
"""

import json
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

import db
import pull_props as pp

load_dotenv(override=True)
ET = ZoneInfo("America/New_York")

MODEL = os.getenv("AI_ANALYST_MODEL", "claude-haiku-4-5-20251001")  # same cost-
                         # conscious default as the AI Analyst - this is a daily
                         # background job, not a user waiting on a reply
MAX_SEARCHES = 8  # one run covers every team playing tonight in a single call, so
                  # this caps total searches across all of them, not per team
ALLOWED_DOMAINS = ["espn.com", "nba.com", "theathletic.com", "cbssports.com",
                   "nbcsports.com", "yahoo.com", "si.com", "rotowire.com",
                   "basketballnews.com", "hoopshype.com", "clutchpoints.com"]

SEARCH_SYSTEM = """You are a sports news researcher for an NBA analytics tool. Search for \
real, recent NBA news about ROLE CHANGES ONLY: trades, a new starting lineup, a coach \
benching or promoting a player, a suspension, or a player's role changing after returning \
from injury. Do NOT report basic injury/health status (out, day-to-day, questionable) - \
that is already tracked separately from a different source. Only report news from the last \
3 days. If you find nothing relevant for a team, leave it out entirely - never invent or \
guess at news, and never pad the results with routine season recaps or game previews."""


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def tonights_teams():
    """Full team names (e.g. 'Golden State Warriors') playing today, ET calendar
    date - the free /events endpoint, same source pull_props.py's own 'today'
    listing uses, so this costs zero odds-API credits."""
    events = pp.call_api(f"/sports/{pp.SPORT}/events", {})
    today = datetime.now(ET).strftime("%Y-%m-%d")
    teams = set()
    for e in events:
        tip = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        if tip.astimezone(ET).strftime("%Y-%m-%d") == today:
            teams.add(e["home_team"])
            teams.add(e["away_team"])
    return sorted(teams)


def search_role_news(team_names):
    """Calls Claude with the web_search tool to find role-change news for the given
    teams, returns a list of {team, headline, summary, source_name, source_url}
    dicts. Returns [] (never raises) if the API isn't configured, the call fails, or
    the response can't be parsed as the expected JSON - a missing or broken news
    search should never take down the rest of the morning job, and a parse failure
    should never be guessed at instead of just reporting nothing."""
    if not team_names:
        return []
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set - skipping role-news search (same key the "
              "AI Analyst uses).")
        return []

    import anthropic
    client = anthropic.Anthropic(timeout=60)
    user_prompt = (
        f"Search for recent NBA role-change news for these teams: {', '.join(team_names)}.\n\n"
        "Respond with ONLY a JSON array (no markdown code fences, no other text before or "
        "after it). Each item:\n"
        '{"team": "<exact team name as given above>", "headline": "<short headline>", '
        '"summary": "<1-2 sentence factual summary, nothing beyond what the source says>", '
        '"source_name": "<publication name>", "source_url": "<the article URL>"}\n\n'
        "Respond with an empty array [] if nothing relevant turned up for any team.")
    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=2000,
            system=SEARCH_SYSTEM,
            tools=[{"type": "web_search_20260318", "name": "web_search",
                   "max_uses": MAX_SEARCHES, "allowed_domains": ALLOWED_DOMAINS}],
            messages=[{"role": "user", "content": user_prompt}])
    except Exception as e:
        print(f"Role-news search failed (non-fatal): {e}")
        return []

    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    # Strip a markdown fence if the model added one anyway, despite being told not to -
    # cheap defensive parsing, not a silent guess at malformed content.
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        print("Role-news search returned something that wasn't valid JSON - skipping "
              f"this run rather than guessing. Raw response:\n{text[:500]}")
        return []
    if not isinstance(items, list):
        print(f"Role-news search returned JSON that wasn't a list - skipping. Got: {items!r}")
        return []
    return [it for it in items if isinstance(it, dict) and it.get("team") and it.get("headline")]


def cmd_pull():
    try:
        teams = tonights_teams()
    except Exception as e:
        sys.exit(f"Couldn't fetch tonight's schedule: {e}")

    if not teams:
        print("No games scheduled today - nothing to search for.")
        return

    print(f"Searching role-change news for {len(teams)} team(s): {', '.join(teams)}")
    items = search_role_news(teams)
    if not items:
        print("No role-change news found (or the search was skipped - see above).")
        return

    pulled_at = iso(datetime.now(timezone.utc))
    rows = [{**it, "pulled_at": pulled_at} for it in items]
    new_in_db = db.save_role_news(rows)
    print(f"Found {len(items)} item(s), {new_in_db} new in the database.")
    for it in items:
        print(f"  [{it['team']}] {it['headline']} ({it.get('source_name', 'unknown source')})")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] != "pull":
        sys.exit("Use: python3 news.py pull")
    cmd_pull()
