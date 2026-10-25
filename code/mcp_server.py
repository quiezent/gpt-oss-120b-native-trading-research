"""The research tools and `submit`, served over MCP (stdio) so a Claude model trades exactly as the
gpt-oss models do: same tools, same descriptions, same results (tools.py), one submit per look.

Started by agent.claude_look through the Claude CLI. Environment:
  TRADER_SUBMIT_FILE  where the submit call is written (the live loop places it after the CLI exits)
  TRADER_LOG_FILE     where each research call is logged (JSON lines)
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


def research(name, args):
    global _calls
    if SUBMIT.exists():
        return "you have already submitted; this look is over"
    if _calls >= tools.MAX_CALLS:
        out = f"research limit reached ({tools.MAX_CALLS} calls): call submit now"
    else:
        try:
            g = gw()
        except Exception as e:              # e.g. client 11 in use elsewhere (326): no call spent; the next call retries
            out = f"error: research connection failed ({type(e).__name__}: {str(e)[:200]}); this call did not count"
        else:
            _calls += 1
            out = tools.run(g, name, args)
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
    import forecast as _forecast

    _fp = _forecast.SPEC["parameters"]["properties"]

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


@server.tool(description="Submit this look's decision: orders to place, open orders to cancel, and notes for your next look. "
                         "Call it exactly once; it ends the look. "
                         "Each order: {symbol, action BUY|SELL, quantity (whole shares), type MKT|MOO|LMT|STP|STP LMT|TRAIL|MOC|LOC, "
                         "limit, stop, trail_percent (as needed), tif DAY|GTC, p_fill, p_win}. Empty lists mean no change. "
                         + tools.PROBS_DESC + " notes: " + tools.NOTES_DESC + " Up to 8,000 characters are kept.")
def submit(place: list[dict], cancel: list[str], notes: str = "") -> str:
    if SUBMIT.exists():
        return "already submitted; only the first submit counts"
    SUBMIT.write_text(json.dumps({"place": place, "cancel": cancel, "notes": notes}), encoding="utf-8")
    return ("submitted: your orders go to the broker as written when this look ends; one that would take gross exposure "
            "past 2x the sleeve is rejected (you'll see it at your next look)")


if __name__ == "__main__":
    server.run("stdio")
