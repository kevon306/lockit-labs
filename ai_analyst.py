"""
ai_analyst.py  -  the AI Analyst's LLM orchestration.

Turns a user message plus conversation history into a reply, using ai_tools.py as its
only way to touch real data. The model never gets fed information except what a tool
call returns or what the user typed - no fabricated stats, lines, injuries, or
probabilities. See ai_tools.py for what's actually available and what explicitly isn't
(individual defenders, coverage schemes, injury/lineup data).

Configuration (.env):
  ANTHROPIC_API_KEY=        required - get one at https://console.anthropic.com/
  AI_ANALYST_ENABLED=on     optional, "off" disables the feature even with a key set
  AI_ANALYST_MODEL=         optional, defaults to claude-sonnet-5
  AI_ANALYST_DAILY_LIMIT=   optional, defaults to 200 requests/day - a soft usage
                            guard against runaway cost, not a real business rule

If ANTHROPIC_API_KEY isn't set, is_configured() returns False and the UI shows a clear
"not connected" state (see page_ai_analyst() in lockit_app.py) instead of attempting
calls or faking a response.
"""

import json
import os
import time

from dotenv import load_dotenv

import ai_tools
import db

load_dotenv(override=True)  # .env always wins - see the comment on pull_props.get_api_key()

MODEL = os.getenv("AI_ANALYST_MODEL", "claude-haiku-4-5-20251001")  # cost-conscious
                         # default - still genuinely capable at tool-calling and
                         # research-style reasoning, at roughly a quarter of Sonnet's
                         # price. Set AI_ANALYST_MODEL=claude-sonnet-5 for more depth
                         # on harder questions, at a real cost increase - see the
                         # pricing table below for the actual numbers.
DAILY_LIMIT = int(os.getenv("AI_ANALYST_DAILY_LIMIT", "200"))
MAX_TOOL_ROUNDS = 6      # a single reply can chain at most this many tool calls before
                         # being forced to answer, so one odd request can't loop forever
REQUEST_TIMEOUT = 30     # seconds per Anthropic API call
MAX_RETRIES = 3

# Anthropic's per-million-token list prices for the configured model, used only to
# show a rough running cost in the UI - not billed anywhere, and goes stale if the
# model above changes without updating this table (kept visible on purpose so that's
# an easy thing to notice and fix, not a silent wrong number).
PRICING_PER_MTOK = {"claude-sonnet-5": (3.00, 15.00), "claude-opus-4-8": (15.00, 75.00),
                    "claude-haiku-4-5-20251001": (0.80, 4.00)}


class AnalystUnavailable(Exception):
    """AI Analyst isn't configured (no API key) or is disabled via feature flag."""


class AnalystRateLimited(Exception):
    """This app's own daily usage guard was hit - not Anthropic's rate limit."""


class AnalystError(Exception):
    """The Anthropic API call itself failed after retries."""


def is_configured():
    return bool(os.getenv("ANTHROPIC_API_KEY")) and os.getenv("AI_ANALYST_ENABLED", "on").lower() != "off"


SYSTEM_PROMPT = """You are the AI Analyst inside LOCK IT Labs, an NBA player-prop research \
tool. You help the user think through player props, spreads, totals, and parlays using this \
app's own real data and model projections - retrieved only through the tools you're given, \
never from memory or general knowledge of current rosters, injuries, or odds.

Ground rules, no exceptions:
- Never state a statistic, line, odds price, projection, or probability without having \
retrieved it through a tool call first. If a tool returns an error or says something isn't \
available, say so plainly - do not fill the gap with a plausible-sounding guess.
- Distinguish, explicitly when it matters: verified data (from a tool), this app's model \
projection, a simplified statistical estimate (like the parlay independence baseline), \
general basketball reasoning, and a claim the user made that you haven't verified (like a \
reported injury - call injury_status, and if it returns unavailable, treat the user's claim \
as an unverified assumption in your answer, not a fact).
- Do not automatically agree with the user. If they propose a pick or a piece of reasoning, \
investigate it with the tools, then say plainly which parts the data supports, which \
assumptions look shaky, and what the strongest counterargument is. If their case is strong, \
say why. If it's weak, say why - specifically and respectfully, not just "disagree."
- You're allowed to disagree with this app's own Top Picks or model projection when the \
retrieved evidence supports a different read. Explain the disagreement clearly and say what \
in the data drove it.
- Never claim a parlay is "safe," never invent a correlation coefficient, and never present \
the product of independent probabilities as a parlay's true combined probability - \
analyze_parlay's independence_baseline is explicitly a simplified floor, not a real joint \
probability, and you must characterize it that way every time you use it.
- Never promise a win or a guaranteed return. This is research, not a guarantee.
- No individual-defender assignments, defensive coverage schemes (drop coverage, switching, \
traps), or coaching-adjustment claims - this app has no data source for any of that. If asked, \
say plainly it isn't available rather than reasoning your way to a plausible-sounding guess.

Response style:
- For a detailed question about a specific pick, or any parlay review, use this structure: \
**My assessment** (concise overall case), **Why it could hit**, **What concerns me**, \
**Matchup breakdown**, **What the numbers say** (line, model projection, probabilities, \
confidence - from tools, always), **My conclusion** (favorable / mixed / unfavorable / too \
uncertain to assess), **What could change the assessment**.
- For a simple, narrow question, just answer it directly - don't force the full structure on \
everything.
- End a substantive answer with 2-3 short, specific follow-up questions the user could ask \
next (not generic ones - tailored to what you just discussed).
- Call tools freely and in parallel when it helps (e.g. compare_props for two players at \
once). search_player first whenever a name might be ambiguous, before calling other player \
tools with it.
- Keep a conversational tone even when using the structured format - this is a knowledgeable \
analyst talking to the user, not a form being filled out."""


TOOLS = [
    {"name": "search_player",
     "description": "Find a player by name (handles partial names, nicknames, typos "
                    "loosely). Call this first whenever a user names a player, to "
                    "resolve to the exact name on file before calling other player tools.",
     "input_schema": {"type": "object",
                      "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "player_profile",
     "description": "Real season averages, last-10 and last-5 form, position, and "
                    "team for one player.",
     "input_schema": {"type": "object",
                      "properties": {"player_name": {"type": "string"}},
                      "required": ["player_name"]}},
    {"name": "player_projection",
     "description": "This app's own model projection for a player's stat tonight, "
                    "including the model's probability vs. the market's if there's an "
                    "active betting line for it.",
     "input_schema": {"type": "object", "properties": {
         "player_name": {"type": "string"},
         "stat": {"type": "string", "enum": ["points", "rebounds", "assists", "fg3m",
                                             "steals", "blocks", "turnovers"]}},
         "required": ["player_name", "stat"]}},
    {"name": "matchup_stats",
     "description": "Real season-to-date defensive numbers for an opponent team "
                    "(points/rebounds/assists/threes allowed, pace, rim and perimeter "
                    "defense from real shot-location data) and this player's real "
                    "head-to-head history against that opponent.",
     "input_schema": {"type": "object", "properties": {
         "player_name": {"type": "string"},
         "opponent_abbr": {"type": "string",
                           "description": "3-letter team abbreviation, e.g. DAL, LAL"}},
         "required": ["player_name", "opponent_abbr"]}},
    {"name": "game_markets",
     "description": "Tonight's real spread/total/moneyline odds and this app's team-"
                    "model projection for a team's game.",
     "input_schema": {"type": "object",
                      "properties": {"team_query": {"type": "string"}},
                      "required": ["team_query"]}},
    {"name": "top_picks",
     "description": "This app's current Top Picks (positive-edge, EV-ranked player "
                    "props and game lines).",
     "input_schema": {"type": "object", "properties": {
         "market": {"type": "string",
                   "description": "Optional filter: points, rebounds, assists, spread, "
                                  "total, or moneyline"}}, "required": []}},
    {"name": "compare_props",
     "description": "Real side-by-side projection and market data for two player props.",
     "input_schema": {"type": "object", "properties": {
         "player_a": {"type": "string"}, "stat_a": {"type": "string"},
         "player_b": {"type": "string"}, "stat_b": {"type": "string"}},
         "required": ["player_a", "stat_a", "player_b", "stat_b"]}},
    {"name": "analyze_parlay",
     "description": "Pulls real projection/market data for every leg of a proposed "
                    "parlay and flags legs that share a game. Each leg must be "
                    "{'player': name, 'stat': stat} or {'team': name, 'market': "
                    "'spreads'|'totals'|'h2h'}.",
     "input_schema": {"type": "object",
                      "properties": {"legs": {"type": "array", "items": {"type": "object"}}},
                      "required": ["legs"]}},
    {"name": "injury_status",
     "description": "Checks for injury/lineup information on a player. This app has "
                    "no injury data source connected, so this always returns "
                    "unavailable - call it anyway so that's explicit, not assumed.",
     "input_schema": {"type": "object",
                      "properties": {"player_name": {"type": "string"}},
                      "required": ["player_name"]}},
    {"name": "prediction_track_record",
     "description": "This app's real, locked prediction ledger and its actual graded "
                    "hit rate so far.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "saved_picks",
     "description": "The user's own tracked entries (bets they recorded as placed "
                    "elsewhere).",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
]

# Prompt caching: the system prompt and tool definitions above are large, static, and
# sent on every single API call (every tool-loop round, every turn of every
# conversation). Marking the end of each with cache_control lets Anthropic cache that
# whole prefix for ~5 minutes - a cached read costs 10% of normal input price instead
# of 100%, which matters a lot here since most of a typical call's input tokens are
# this fixed prefix, not the actual conversation. See _estimate_cost() for the real
# cost math, including the one-time write cost (1.25x) the first call pays.
SYSTEM_BLOCKS = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
CACHED_TOOLS = [dict(t) for t in TOOLS]
CACHED_TOOLS[-1] = {**CACHED_TOOLS[-1], "cache_control": {"type": "ephemeral"}}

TOOL_FUNCS = {
    "search_player": lambda **kw: ai_tools.search_player(kw["query"]),
    "player_profile": lambda **kw: ai_tools.player_profile(kw["player_name"]),
    "player_projection": lambda **kw: ai_tools.player_projection(
        kw["player_name"], kw.get("stat", "points")),
    "matchup_stats": lambda **kw: ai_tools.matchup_stats(kw["player_name"], kw["opponent_abbr"]),
    "game_markets": lambda **kw: ai_tools.game_markets(kw["team_query"]),
    "top_picks": lambda **kw: ai_tools.top_picks(kw.get("market")),
    "compare_props": lambda **kw: ai_tools.compare_props(
        kw["player_a"], kw["stat_a"], kw["player_b"], kw["stat_b"]),
    "analyze_parlay": lambda **kw: ai_tools.analyze_parlay(kw["legs"]),
    "injury_status": lambda **kw: ai_tools.injury_status(kw["player_name"]),
    "prediction_track_record": lambda **kw: ai_tools.prediction_track_record(),
    "saved_picks": lambda **kw: ai_tools.saved_picks(),
}


def _estimate_cost(tokens_in, tokens_out, cache_write=0, cache_read=0):
    """Real blended cost, accounting for prompt caching: a cache write costs 1.25x
    normal input price (paid once, the first call), a cache read costs 0.1x (paid on
    every call after that within the ~5 min cache window)."""
    rates = PRICING_PER_MTOK.get(MODEL)
    if not rates:
        return None
    in_rate, out_rate = rates
    cost = (tokens_in / 1_000_000 * in_rate + tokens_out / 1_000_000 * out_rate
           + cache_write / 1_000_000 * in_rate * 1.25
           + cache_read / 1_000_000 * in_rate * 0.1)
    return round(cost, 6)


def _call_with_retry(client, convo):
    import anthropic
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return client.messages.create(
                model=MODEL, max_tokens=1500, system=SYSTEM_BLOCKS,
                tools=CACHED_TOOLS, messages=convo)
        except anthropic.APIStatusError as e:
            if e.status_code in (401, 403):
                raise  # a bad/missing credential - retrying changes nothing
            last_err = e
        except (anthropic.APIConnectionError, anthropic.RateLimitError,
                anthropic.APITimeoutError) as e:
            last_err = e
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)
    raise last_err


def call_analyst(messages, conversation_id=None, user_id="owner"):
    """messages: [{"role": "user"|"assistant", "content": str}, ...] - the full
    conversation so far, ending with the new user message. Returns
    (reply_text, tool_calls_made, tokens_in, tokens_out).

    Raises AnalystUnavailable, AnalystRateLimited, or AnalystError with a message
    that's safe to show the user directly."""
    if not is_configured():
        raise AnalystUnavailable(
            "AI Analyst isn't configured yet. Add ANTHROPIC_API_KEY to your .env file "
            "to turn it on (get one at console.anthropic.com).")

    usage = db.ai_usage_today(user_id)
    if usage["requests"] >= DAILY_LIMIT:
        raise AnalystRateLimited(
            f"Daily limit of {DAILY_LIMIT} AI Analyst requests reached. Resets at "
            "midnight UTC, or raise AI_ANALYST_DAILY_LIMIT in .env.")

    import anthropic
    client = anthropic.Anthropic(timeout=REQUEST_TIMEOUT)

    convo = [{"role": m["role"], "content": m["content"]} for m in messages]
    tool_calls_made = []
    total_in = total_out = total_cache_write = total_cache_read = 0

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            resp = _call_with_retry(client, convo)
        except Exception as e:
            db.log_ai_usage(user_id, conversation_id, MODEL, total_in, total_out, None, str(e))
            raise AnalystError(
                "The AI Analyst couldn't reach the model right now. Try again in a "
                f"moment. ({e})")

        total_in += resp.usage.input_tokens
        total_out += resp.usage.output_tokens
        total_cache_write += getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
        total_cache_read += getattr(resp.usage, "cache_read_input_tokens", 0) or 0

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            text = "".join(b.text for b in resp.content if b.type == "text")
            cost = _estimate_cost(total_in, total_out, total_cache_write, total_cache_read)
            db.log_ai_usage(user_id, conversation_id, MODEL,
                            total_in + total_cache_write + total_cache_read, total_out, cost)
            return text, tool_calls_made, total_in, total_out

        convo.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for call in tool_uses:
            func = TOOL_FUNCS.get(call.name)
            try:
                result = func(**call.input) if func else {"error": f"Unknown tool '{call.name}'"}
            except Exception as e:
                result = {"error": f"Tool failed: {e}"}
            tool_calls_made.append({"tool": call.name, "input": call.input, "result": result})
            tool_results.append({"type": "tool_result", "tool_use_id": call.id,
                                "content": json.dumps(result, default=str)})
        convo.append({"role": "user", "content": tool_results})

    # Hit MAX_TOOL_ROUNDS without a final answer - force one more call with no tools
    # available so the user gets an actual reply instead of silence.
    try:
        resp = client.messages.create(model=MODEL, max_tokens=1500, system=SYSTEM_BLOCKS,
                                      messages=convo)
        total_in += resp.usage.input_tokens
        total_out += resp.usage.output_tokens
        total_cache_write += getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
        total_cache_read += getattr(resp.usage, "cache_read_input_tokens", 0) or 0
        text = "".join(b.text for b in resp.content if b.type == "text")
    except Exception:
        text = ("I gathered a lot of information on this but ran out of room to finish "
               "the analysis cleanly - try asking a narrower follow-up question.")
    cost = _estimate_cost(total_in, total_out, total_cache_write, total_cache_read)
    db.log_ai_usage(user_id, conversation_id, MODEL,
                    total_in + total_cache_write + total_cache_read, total_out, cost)
    return text, tool_calls_made, total_in, total_out


def start_conversation(mode="ask", title=None, user_id="owner"):
    return db.new_conversation(mode=mode, title=title, user_id=user_id)


def ask(conversation_id, user_message, user_id="owner"):
    """The UI's main entry point: saves the user's message, builds the full message
    history for this conversation, calls the model, saves the reply, and returns it.
    Raises the same exceptions as call_analyst() - the UI is expected to catch them
    and show a clear state, not crash the page."""
    db.save_message(conversation_id, "user", user_message)
    db.auto_title_conversation(conversation_id, user_message)
    history = db.conversation_messages(conversation_id, user_id=user_id)
    convo_messages = [{"role": r.role, "content": r.content} for r in history.itertuples()]
    reply, tool_calls, tin, tout = call_analyst(
        convo_messages, conversation_id=conversation_id, user_id=user_id)
    mid = db.save_message(conversation_id, "assistant", reply, tool_calls=tool_calls,
                          tokens_in=tin, tokens_out=tout)
    return reply, mid, tool_calls
