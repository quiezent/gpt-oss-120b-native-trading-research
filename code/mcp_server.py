"""The research tools and `submit`, served over MCP (stdio) so a Claude model trades exactly as the
gpt-oss models do: same tools, same descriptions, same results (tools.py), one submit per look.

Started by agent.claude_look through the Claude CLI. Environment:
  TRADER_SUBMIT_FILE  where the submit call is written (the live loop places it after the CLI exits)
  TRADER_LOG_FILE     where each research call is logged (JSON lines)
  TRADER_MAX_CALLS    the sleeve's research-call limit (tools.MAX_CALLS if absent)
"""

import json
import os
import sys
from pathlib import Path
from typing import Annotated

from pydantic import Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.mcpserver import MCPServer

import tools

server = MCPServer(name="trader")
SUBMIT = Path(os.environ["TRADER_SUBMIT_FILE"])
LOG = Path(os.environ["TRADER_LOG_FILE"])
_gw = None
_calls = 0


def gw():
    """Research only, on the read-only client the look was given (TRADER_MCP_CLIENT: trader-mcp-a = 12 or trader-mcp-b = 13,
    registered by the TWS Gateway Owner), so Claude looks can run while the live loop holds client 10."""
    global _gw
    if _gw is None:
        from tws import Gateway
        _gw = Gateway(os.environ.get("TRADER_MCP_CLIENT", "trader-mcp-a"), timeout=30)
    return _gw


ML_TOOL = os.environ.get("TRADER_ML_TOOL") or None     # the sleeve's forecast tool version (ml_tools.py)
TEXT = "A" if os.environ.get("TRADER_TEXT") == "A" else "B"   # the sleeve's text (trader.text_of): the submit description


def _sleeve_value():
    """The sleeve's value in its state (TRADER_SLEEVE_VALUE), the forecast tool's sleeve% base; None if absent or not a number."""
    try:
        return float(os.environ.get("TRADER_SLEEVE_VALUE") or "")
    except ValueError:
        return None


SLEEVE_VALUE = _sleeve_value()


def _max_calls():
    """The sleeve's research-call limit (TRADER_MAX_CALLS, from trader.max_calls_of through agent.claude_look); tools.MAX_CALLS
    if absent or not a whole number from 1 to tools.MAX_CALLS_TOP."""
    try:
        v = int(os.environ.get("TRADER_MAX_CALLS") or "")
    except ValueError:
        return tools.MAX_CALLS
    return v if 1 <= v <= tools.MAX_CALLS_TOP else tools.MAX_CALLS


MAX_CALLS = _max_calls()


def research(name, args):
    global _calls
    if SUBMIT.exists():
        return "you have already submitted; this look is over"
    if _calls >= MAX_CALLS:
        out = f"research limit reached ({MAX_CALLS} calls): call submit now"
    else:
        try:
            g = gw()
        except Exception as e:              # e.g. client 11 in use elsewhere (326): no call spent; the next call retries
            out = f"error: research connection failed ({type(e).__name__}: {str(e)[:200]}); this call did not count"
        else:
            _calls += 1
            # one MCP server process is one look, so "mcp" keys the forecast legend for this look alone
            out = (tools.run(g, name, args, ml_tool=ML_TOOL, sleeve_value=SLEEVE_VALUE, look="mcp")
                   + f"\n(research call {_calls} of {MAX_CALLS})")   # as agent.gptoss_look
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"tool": name, "args": args, "result": out}) + "\n")
    return out


DESC = {s["name"]: s["description"] for s in tools.SPECS}


@server.tool(description=DESC["quote"])
def quote(symbols: list[str]) -> str:
    return research("quote", {"symbols": symbols})


@server.tool(description=DESC["bars"])
def bars(symbol: str, bar_size: str, duration: str) -> str:
    return research("bars", {"symbol": symbol, "bar_size": bar_size, "duration": duration})


if os.environ.get("TRADER_HAS_NEWS") == "1":
    @server.tool(description=DESC["news"])
    def news(symbol: str, limit: int = 10) -> str:
        return research("news", {"symbol": symbol, "limit": limit})

    @server.tool(description=DESC["tape"])
    def tape() -> str:
        return research("tape", {})

    @server.tool(description=DESC["article"])
    def article(provider: str, article_id: str) -> str:
        return research("article", {"provider": provider, "article_id": article_id})

if os.environ.get("TRADER_HAS_STATS") == "1":
    @server.tool(description=DESC["stats"])
    def stats(symbols: list[str]) -> str:
        return research("stats", {"symbols": symbols})

if os.environ.get("TRADER_HAS_FORECAST") == "1":
    import ml_tools
    _forecast = ml_tools.module(ML_TOOL)

    _fp = _forecast.SPEC["parameters"]["properties"]

    if "track_record" in _fp:                   # ML4: the record on request (forecast4.py); ML2 and ML3 take the block below
        @server.tool(description=_forecast.SPEC["description"])
        def forecast(symbols: Annotated[list[str], Field(description=_fp["symbols"]["description"])],
                     notional_usd: Annotated[float | None, Field(description=_fp["notional_usd"]["description"])] = None,
                     sleeve_usd: Annotated[float | None, Field(description=_fp["sleeve_usd"]["description"])] = None,
                     track_record: Annotated[bool | None, Field(description=_fp["track_record"]["description"])] = None) -> str:
            return research("forecast", {"symbols": symbols, "notional_usd": notional_usd, "sleeve_usd": sleeve_usd,
                                         "track_record": track_record})
    else:
        @server.tool(description=_forecast.SPEC["description"])         # gpt-oss's SPEC text, so both read the same (F12)
        def forecast(symbols: Annotated[list[str], Field(description=_fp["symbols"]["description"])],
                     notional_usd: Annotated[float | None, Field(description=_fp["notional_usd"]["description"])] = None,
                     sleeve_usd: Annotated[float | None, Field(description=_fp["sleeve_usd"]["description"])] = None) -> str:
            return research("forecast", {"symbols": symbols, "notional_usd": notional_usd, "sleeve_usd": sleeve_usd})

if os.environ.get("TRADER_HAS_SCANNER") == "1":
    @server.tool(description=DESC["scanner"])
    def scanner(scan: str, rows: int = 15, industry: str | None = None, types: str = "stocks", price_above: float = 5,
                avg_volume_above: float = 1_000_000, market_cap_above: float | None = None) -> str:
        return research("scanner", {"scan": scan, "rows": rows, "industry": industry, "types": types, "price_above": price_above,
                                    "avg_volume_above": avg_volume_above, "market_cap_above": market_cap_above})


_checks, _link, _tries = 0, None, 0
CHECK_FILE = os.environ.get("TRADER_CHECK_FILE") or ""   # the sleeve's state for check_orders (agent.claude_look writes it)


def order_check(args):
    """Text B's check_orders (checks.py), up to checks.MAX_CHECKS a look outside the research calls (Clayton, 14 Nov). One
    MCP process is one look, so the count lives here. The what-ifs run on this process's own research client (12 to 15: the
    Gateway Owner's correction 6), never on 17, which live.py holds; the sleeve's state comes from TRADER_CHECK_FILE."""
    global _checks, _link, _tries
    import checks
    raw = None
    _tries += 1                             # every call, served or not, as agent.gptoss_look counts them (review, 15 Nov)
    if SUBMIT.exists():
        out = "you have already submitted; this look is over"
    elif _checks >= checks.MAX_CHECKS or _tries > checks.MAX_TRIES:
        out = checks.cap_text(_checks)
    else:
        try:
            g = gw()
        except Exception as e:              # noqa: BLE001 - no check spent; the next call retries
            out = f"error: the what-if connection failed ({type(e).__name__}: {str(e)[:200]}); this check did not count."
        else:
            try:
                ctx = json.loads(Path(CHECK_FILE).read_text(encoding="utf-8"))
            except Exception:               # noqa: BLE001 - checks.check then says the check could not run, uncounted
                ctx = None
            if _link is None or _link.gw is not g:
                _link = checks.Link(g)
            out, counted, raw = checks.check(_link, args, ctx)
            if counted:
                _checks += 1
                out += f"\n(order check {_checks} of {checks.MAX_CHECKS})"
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"tool": "check_orders", "args": args, "result": out, "ibkr": raw}, default=str) + "\n")
    return out


if os.environ.get("TRADER_HAS_CHECK") == "1" and CHECK_FILE:      # text B only (agent.claude_look sets it)
    import checks as _c

    @server.tool(description=_c.CHECK_DESC)
    def check_orders(place: Annotated[list[dict], Field(description=_c.PLACE_DESC)]) -> str:
        return order_check({"place": place})


SUBMIT_HEAD = ("Submit this look's decision: orders to place, open orders to cancel, and notes for your next look. "
               "Call it exactly once; it ends the look. "
               "Each order: {symbol, action BUY|SELL, quantity (whole shares), type MKT|MOO|LMT|STP|STP LMT|TRAIL|MOC|LOC, "
               "limit, stop, trail_percent (as needed), tif DAY|GTC")
SUBMIT_TAIL = "notes: " + tools.NOTES_DESC + " Up to 8,000 characters are kept."
# Text A (the control) keeps the description of fd95e2a, with p_fill and p_win; text B, from 16 Nov 2026, drops them and
# states the trailing-stop unit after the order fields (Clayton, 14 Nov)
SUBMIT_DESC = {"A": SUBMIT_HEAD + ", p_fill, p_win}. Empty lists mean no change. " + tools.PROBS_DESC + " " + SUBMIT_TAIL,
               "B": SUBMIT_HEAD + ", close_only true|false}. " + tools.TRAIL_DESC + " " + tools.CLOSE_DESC
                    + " Empty lists mean no change. " + SUBMIT_TAIL}   # close_only: Clayton, 17 Nov (text B only)


_reasked = False        # the one re-ask a look has for a submit whose order lacks its price field (one process = one look)
_pending = False        # the re-asked submit is written as the look's decision until a second submit replaces it, once


@server.tool(description=SUBMIT_DESC[TEXT])
def submit(place: list[dict], cancel: list[str], notes: str = "") -> str:
    global _reasked, _pending
    if SUBMIT.exists() and not _pending:
        return "already submitted; only the first submit counts"
    lack = tools.missing_prices(place)
    if lack and not _reasked:           # as agent.gptoss_look (Clayton, 16 Nov: "Yes, build it"): asked once. The submit is
        _reasked = _pending = True      # written now, so it is the decision if the CLI ends or times out before a second one
        SUBMIT.write_text(json.dumps({"place": place, "cancel": cancel, "notes": notes}), encoding="utf-8")   # (review)
        out = tools.missing_text(lack)
        with LOG.open("a", encoding="utf-8") as f:         # "reask", not "tool": no research call in the look's record
            f.write(json.dumps({"reask": "submit", "args": {"place": place, "cancel": cancel, "notes": notes}, "result": out,
                                "missing": [{"order": i, "fields": k} for i, _, k, _ in lack]}) + "\n")
        return out
    _pending = False
    SUBMIT.write_text(json.dumps({"place": place, "cancel": cancel, "notes": notes}), encoding="utf-8")
    g = os.environ.get("TRADER_MAX_GROSS", "2")      # the sleeve's gross term (trader.max_gross_of); "none" = no software limit
    if g == "none":
        co = any(isinstance(o, dict) and (o.get("close_only") is True or str(o.get("close_only")).strip().lower() == "true")
                 for o in place or [])
        return ("submitted: your orders go to the broker as written when this look ends; one that IBKR refuses is "
                "reported at your next look" + ("; a close_only order goes only if your position covers it then"
                                                if co else ""))
    return ("submitted: your orders go to the broker as written when this look ends; one that would take gross exposure "
            f"past {g}x the sleeve is rejected (you'll see it at your next look)")


if __name__ == "__main__":
    server.run("stdio")
