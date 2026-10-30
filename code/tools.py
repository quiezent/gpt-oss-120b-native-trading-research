"""Research tools every model gets before it submits orders: the same names, schemas and results for
gpt-oss (native tool calls) and for Claude (through mcp_server.py).

Each look, a model may call up to MAX_CALLS research tools, then must call `submit` once. Results are
compact text so they fit the context. News and scanners appear only if the TWS framework offers them.

news, article and tape: live, the gateway's rows plus the company's SEC 8-Ks and our macro calendar rows
(ml_research/livenews.py); in training, ml_research/simnews.py serves the same sources point-in-time.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
LISTS = Path(__file__).resolve().parent / "lists"
MAX_CALLS = 12
MAX_SYMBOLS = 20
MAX_STATS = 10
NOTES_DESC = ("Notes to yourself for your next look: your thesis, plan and exit levels. Nothing else you think or "
              "research carries over; these notes are shown to you at your next look.")
# Per-order probabilities (shared with mcp_server.py): logged and scored by calib_report.py and reward.calibration_bonus,
# never used to handle an order.
P_FILL_DESC = "Your probability (0-1) that this order fills before it expires or is cancelled. Give it for every order."
P_WIN_DESC = ("Your probability (0-1) that at the close 5 sessions after the session it fills, the price has moved in this "
              "order's favour by more than 0.2% of the fill price (up for a BUY, down for a SELL). Give it for every order.")
PROBS_NOTE = "Each order's p_fill and p_win are recorded and scored against what happens; they never change how the order is handled."
# in submit's own description too: the gpt-oss (Harmony) renderer drops descriptions of an array's item fields
PROBS_DESC = f"p_fill: {P_FILL_DESC} p_win: {P_WIN_DESC} {PROBS_NOTE}"

# IBKR scan codes that work on this account (checked 15 Oct 2026; the Refinitiv fundamental scans are not allowed).
SCANS = {
    "today's move": {"TOP_PERC_GAIN": "biggest % gain today", "TOP_PERC_LOSE": "biggest % loss today",
                     "TOP_OPEN_PERC_GAIN": "biggest % gain since today's open", "TOP_OPEN_PERC_LOSE": "biggest % loss since today's open",
                     "HIGH_OPEN_GAP": "biggest gap up at today's open", "LOW_OPEN_GAP": "biggest gap down at today's open"},
    "activity": {"MOST_ACTIVE": "most shares traded today", "MOST_ACTIVE_USD": "most dollars traded today",
                 "HOT_BY_VOLUME": "volume far above its usual"},
    "range": {"HIGH_VS_52W_HL": "at or near the 52-week high", "LOW_VS_52W_HL": "at or near the 52-week low",
              "HIGH_VS_13W_HL": "near the 13-week high", "LOW_VS_13W_HL": "near the 13-week low"},
    "trend": {"BULLISH_LAST_VS_EMA50": "price moving above its 50-day average", "BEARISH_LAST_VS_EMA50": "price moving below its 50-day average",
              "BULLISH_EMA_DIFF_VS_LAST": "20-day average rising against the 200-day", "BEARISH_EMA_DIFF_VS_LAST": "20-day average falling against the 200-day",
              "SCAN_histPerfRatio1m_DESC": "best over the past month", "SCAN_histPerfRatio1m_ASC": "worst over the past month",
              "SCAN_histPerfRatio3m_DESC": "best over three months", "SCAN_histPerfRatio3m_ASC": "worst over three months"},
    "options and positioning": {"HIGH_OPT_IMP_VOLAT": "highest option implied volatility (big moves expected)",
                                "HIGH_OPT_IMP_VOLAT_OVER_HIST": "implied volatility far above realized (an event is expected)",
                                "HIGH_OPT_VOLUME_PUT_CALL_RATIO": "heavy put buying today", "LOW_OPT_VOLUME_PUT_CALL_RATIO": "heavy call buying today",
                                "SCAN_utilization_DESC": "most heavily shorted (share-lending utilization)",
                                "HIGH_DIVIDEND_YIELD_IB": "highest dividend yield"},
}
SCAN_CODES = [c for group in SCANS.values() for c in group]
INDUSTRIES = ["Basic Materials", "Communications", "Consumer, Cyclical", "Consumer, Non-cyclical", "Energy", "Financial",
              "Industrial", "Technology", "Utilities"]   # IBKR's industry classes (health care is in Consumer, Non-cyclical)

TAPE_DESC = ("Recent market-wide headlines. First our macro rows, newest first: US economic releases of the past week with "
             "first-release values and the Kalshi pre-release expectation, FOMC decisions, and today's release calendar (ET) "
             "(a release's values reach our data the evening after it). Then {sources} headlines, newest first: markets, "
             "sectors, Fed.")
LIVE_SOURCES = "Briefing.com and Dow Jones (Dow Jones rows are headlines only, with no article text)"

SPECS = [
    {"name": "quote", "description": "Delayed (about 15 minutes) snapshot quotes, one labelled line per symbol: last with its % change vs the previous close, bid, ask, the session's open/high/low, previous close, volume, and the security's name (with ETF or its industry).",
     "parameters": {"type": "object", "properties": {"symbols": {"type": "array", "items": {"type": "string"},
                                                                 "description": f"Up to {MAX_SYMBOLS} US stock or ETF symbols."}},
                    "required": ["symbols"]}},
    {"name": "bars", "description": "Price history for one symbol (regular hours). bar_size: '5 mins', '15 mins', '1 hour' or '1 day'. duration: e.g. '1 D', '5 D', '1 M', '6 M', '1 Y'. At most the last 60 bars are returned.",
     "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "bar_size": {"type": "string"},
                                                     "duration": {"type": "string"}}, "required": ["symbol", "bar_size", "duration"]}},
    {"name": "news", "description": "Headlines for one symbol, newest first, with article ids: Briefing.com columns and analyst "
                                    "actions, and the company's SEC 8-K filings (provider EDGAR; acceptance time; its article is the "
                                    "press release or filing text). A symbol's newest headline can be months or even years old; "
                                    "each row carries its date. ETFs and funds usually have none here (no 8-Ks, few Briefing.com "
                                    "items); market and sector news is on tape.",
     "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "limit": {"type": "integer", "description": "Up to 10."}},
                    "required": ["symbol"]}},
    {"name": "tape", "description": TAPE_DESC.format(sources=LIVE_SOURCES),
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "article", "description": "The text of one news article, by provider and article id from `news` or `tape`.",
     "parameters": {"type": "object", "properties": {"provider": {"type": "string"}, "article_id": {"type": "string"}},
                    "required": ["provider", "article_id"]}},
    {"name": "stats", "description": f"Volatility, ranges, liquidity and short availability for up to {MAX_STATS} US stocks or ETFs: "
                                     "30-day historical and implied volatility (annualized %; implied far above historical often means "
                                     "an event such as earnings is expected; no implied figure if the stock has no listed options), "
                                     "13- and 52-week low and high, average daily volume in shares, and whether shares can be borrowed "
                                     "to sell short (available; locate = only if IBKR can find shares, a short may be rejected; "
                                     "unavailable) with the number of shares IBKR can lend.",
     "parameters": {"type": "object", "properties": {"symbols": {"type": "array", "items": {"type": "string"},
                                                                 "description": f"Up to {MAX_STATS} symbols."}},
                    "required": ["symbols"]}},
    {"name": "scanner", "description": "IBKR market scanner for US stocks on the major exchanges: the top-ranked stocks for one scan, "
                                       "with delayed last price, % change today and name. Pick the scan (and industry) that tests your hypothesis. Scans by kind: "
                                       + " ".join(f"[{g}] " + "; ".join(f"{k} = {v}" for k, v in s.items()) + "." for g, s in SCANS.items()) +
                                       " industry (optional) limits the scan to one of IBKR's industry classes: " + ", ".join(INDUSTRIES) +
                                       " (health care is in 'Consumer, Non-cyclical'). Unless you set them, price_above is 5 and avg_volume_above is "
                                       "1,000,000 shares a day on average (this leaves out penny stocks) and types is 'stocks' (common stocks and "
                                       "ADRs; 'all' adds ETFs and ETNs, which include many leveraged products); market_cap_above is in $ millions (10000 means $10 billion). "
                                       "rows: up to 25 (default 15).",
     "parameters": {"type": "object", "properties": {"scan": {"type": "string", "enum": SCAN_CODES}, "rows": {"type": "integer"},
                                                     "industry": {"type": "string", "enum": INDUSTRIES},
                                                     "types": {"type": "string", "enum": ["stocks", "all"]},
                                                     "price_above": {"type": "number"}, "avg_volume_above": {"type": "number"},
                                                     "market_cap_above": {"type": "number"}},
                    "required": ["scan"]}},
]


def available(gw):
    """The research tools this framework version supports (news/scanner only once the framework has them)."""
    have = {"quote", "bars"}
    if hasattr(gw, "news"):
        have.add("news")
    if hasattr(gw, "article"):
        have.add("article")
    if hasattr(gw, "tape"):
        have.add("tape")
    if hasattr(gw, "scan"):
        have.add("scanner")
    if hasattr(gw, "stats"):
        have.add("stats")
    src = getattr(gw, "tape_sources", None)          # the simulator names only the sources it has
    return [dict(s, description=TAPE_DESC.format(sources=src)) if src and s["name"] == "tape" else s
            for s in SPECS if s["name"] in have]


def _live(gw):
    """The real gateway (the simulator's news gateway serves EDGAR and macro rows itself)."""
    return not hasattr(gw, "news_archive")


def _f(x, p=2):
    return "na" if x is None else f"{x:.{p}f}" if isinstance(x, float) else str(x)


_names = None
_no_name, _names_off = set(), [0.0]
NAME_LOOKUPS, NAME_SECONDS = 10, 6


def names(gw, syms):
    """{symbol: "LONG NAME (ETF | industry)"}: lists/industries.json, then, live, the gateway's contract details for
    any other symbol, kept in lists/names.json (week 2: 18 gpt-oss looks ordered a symbol they had only guessed at,
    e.g. 'NANO is a biotech?'). Never raises: a symbol without a name gets none."""
    global _names
    try:
        if _names is None:
            _names = {}
            for f in (LISTS / "industries.json", LISTS / "names.json"):
                if f.exists():
                    _names.update(json.loads(f.read_text(encoding="utf-8")))
        miss = [s for s in dict.fromkeys(syms) if s not in _names and s not in _no_name]
        if miss and _live(gw) and hasattr(gw, "details") and time.monotonic() >= _names_off[0]:
            got, t0 = {}, time.monotonic()
            for s in miss[:NAME_LOOKUPS]:           # one at a time, at most 10 and none started after 6 s, so a
                if time.monotonic() - t0 > NAME_SECONDS:    # scanner of unknown names never holds up the look
                    break
                try:
                    r = gw.details(s)
                except Exception as e:              # noqa: BLE001
                    if any(k in str(e) for k in ("200", "no contract", "ambiguous")):
                        _no_name.add(s)             # IBKR has no single such contract: don't ask again
                        continue
                    _names_off[0] = time.monotonic() + 600      # a stall or disconnect: no lookups for 10 minutes
                    break
                if r.get("long_name"):
                    got[r["symbol"]] = {k: r.get(k) for k in ("industry", "stock_type", "long_name")}
            if got:
                _names.update(got)
                f = LISTS / "names.json"
                old = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
                tmp = f.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_text(json.dumps(old | got, indent=1), encoding="utf-8")
                os.replace(tmp, f)
    except Exception:                               # noqa: BLE001 - names are a courtesy; the quote goes out regardless
        pass
    out = {}
    for s in syms:
        d = (_names or {}).get(s)
        if d and d.get("long_name"):
            kind = "ETF" if d.get("stock_type") == "ETF" else (d.get("industry") or d.get("stock_type") or "")
            out[s] = d["long_name"] + (f" ({kind})" if kind else "")
    return out


def _headline(r):
    """'YYYY-MM-DD HH:MM ET [provider article_id] headline' (IBKR news times are UTC)."""
    t = r.get("time") or ""
    try:
        t = datetime.fromisoformat(str(t).replace("Z", "+00:00")).astimezone(ET).strftime("%Y-%m-%d %H:%M ET")
    except ValueError:
        pass
    return f"{t} [{r.get('provider')} {r.get('article_id')}] {r.get('headline')}"


def run(gw, name, args):
    """Execute one research tool; returns text. Errors come back as text, verbatim."""
    try:
        if not isinstance(args, dict):
            return "error: arguments must be a JSON object"
        if name == "quote":
            syms = [str(s).upper().strip() for s in (args.get("symbols") or [])][:MAX_SYMBOLS]
            if not syms:
                return "error: give at least one symbol"
            lines = []
            qs = gw.quotes(syms)
            nm = names(gw, [q["symbol"] for q in qs if not q.get("error")])   # a stall is names()' own back-off
            for q in qs:
                if q.get("error"):
                    lines.append(f"{q['symbol']} error: {q['error']}")
                    continue
                t = q.get("last_time")
                if t:
                    try:
                        t = datetime.fromisoformat(t.replace("Z", "+00:00")).astimezone(ET).strftime("%H:%M")
                    except ValueError:
                        pass
                last, prev = q.get("last"), q.get("close")
                chg = f" ({(last / prev - 1) * 100:+.2f}% vs prev_close)" if last and prev else ""
                lines.append(f"{q['symbol']}: last {_f(last)}{chg} | bid {_f(q.get('bid'))} ask {_f(q.get('ask'))} | session "
                             f"open {_f(q.get('open'))} high {_f(q.get('high'))} low {_f(q.get('low'))} | prev_close {_f(prev)} "
                             f"| volume {_f(q.get('volume'), 0)} | last trade {f'{t} ET' if t else 'na'}"
                             + (f" | {nm[q['symbol']]}" if q["symbol"] in nm else ""))
            return "\n".join(lines)
        if name == "bars":
            sym = str(args.get("symbol", "")).upper().strip()
            rows = gw.bars(sym, duration=str(args.get("duration", "5 D")), bar_size=str(args.get("bar_size", "1 hour")),
                           what="TRADES", rth=True)
            n = len(rows)
            rows = rows[-60:]
            return "\n".join([f"{sym} time open high low close volume"] +
                             [f"{r['date']} {_f(r['open'])} {_f(r['high'])} {_f(r['low'])} {_f(r['close'])} {_f(r['volume'], 0)}" for r in rows]
                             + ([f"(the last 60 of {n} bars)"] if n > 60 else []))
        if name == "news":
            sym, lim = str(args.get("symbol", "")).upper().strip(), min(int(args.get("limit") or 10), 10)
            if _live(gw):
                from ml_research import livenews
                rows, note = livenews.news(gw, sym, lim)
            else:
                rows, note = gw.news(sym, limit=lim), ""
            return ("\n".join([_headline(r) for r in rows] + ([note] if note else []))
                    or f"no headlines for {sym} (ETFs and funds usually have none here; market and sector news is on tape)")
        if name == "tape":
            if _live(gw):
                from ml_research import livenews
                rows, note = livenews.tape(gw)
            else:
                rows, note = gw.tape(seconds=5), ""
            return "\n".join([_headline(r) for r in rows[:25]] + ([note] if note else [])) or "no headlines"
        if name == "article":
            if _live(gw):
                from ml_research import livenews
                text = livenews.article(gw, str(args.get("provider")), str(args.get("article_id")))
            else:
                text = gw.article(str(args.get("provider")), str(args.get("article_id")))
            text = text.get("text", "") if isinstance(text, dict) else str(text)
            if len(text) <= 4000:
                return text
            edgar = str(args.get("provider")).upper() == "EDGAR"     # 8-K text is stored only up to 6,000 characters
            return text[:4000] + (f"\n[cut at 4,000 characters; our copy of the filing stops at {len(text):,}, and the "
                                  "filing may be longer]" if edgar else f"\n[cut at 4,000 of {len(text):,} characters]")
        if name == "forecast":                     # the ML arm's tool (forecast.py); only -ml sleeves are offered it
            import forecast
            return forecast.run(gw, args)
        if name == "stats":
            syms = [str(s).upper().strip() for s in (args.get("symbols") or [])][:MAX_STATS]
            if not syms:
                return "error: give at least one symbol"
            pct = lambda v: "na" if v is None else f"{100 * v:.1f}"
            lines = ["sym hist_vol% implied_vol% low_13w high_13w low_52w high_52w avg_volume borrow shortable_shares"]
            for r in gw.stats(syms):
                if r.get("error"):
                    lines.append(f"{r.get('symbol')} error: {r['error']}")
                    continue
                sh = r.get("shortable_shares")
                lines.append(" ".join([r["symbol"], pct(r.get("hist_vol")), pct(r.get("implied_vol")), _f(r.get("low_13w")),
                                       _f(r.get("high_13w")), _f(r.get("low_52w")), _f(r.get("high_52w")),
                                       "na" if r.get("avg_volume") is None else f"{r['avg_volume']:,.0f}", str(r.get("borrow") or "na"),
                                       "na" if sh is None else f"{sh:,.0f}"]))
            return "\n".join(lines)
        if name == "scanner":
            n = max(1, min(int(args.get("rows") or 15), 25))
            filters = {"priceAbove": args.get("price_above", 5), "avgVolumeAbove": args.get("avg_volume_above", 1_000_000),
                       "marketCapAbove1e6": args.get("market_cap_above"),
                       "stkTypes": None if args.get("types") == "all" else "inc:CORP,ADR", "industryLike": args.get("industry")}
            hits = gw.scan(str(args.get("scan", "")).upper().strip(), rows=n, **{k: v for k, v in filters.items() if v is not None})
            if not hits:
                return "no results"
            q = {x["symbol"]: x for x in gw.quotes([r["symbol"] for r in hits]) if not x.get("error")}
            nm = names(gw, [r["symbol"] for r in hits])
            lines = ["rank sym last today% name"]
            for r in hits:
                x = q.get(r["symbol"], {})
                last, prev = x.get("last"), x.get("close")
                lines.append(f"{r['rank']} {r['symbol']} {_f(last)} " + (f"{100 * (last / prev - 1):+.1f}" if last and prev else "na")
                             + (f" {nm[r['symbol']]}" if r["symbol"] in nm else ""))
            return "\n".join(lines)
        return f"error: unknown tool {name!r}"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"[:600]


def symbols_in(name, args):
    """Symbols a research call looked at (for the research log)."""
    if not isinstance(args, dict):
        return []
    if name == "quote":
        return [str(s).upper().strip() for s in (args.get("symbols") or [])]
    if name in ("bars", "news"):
        return [str(args.get("symbol", "")).upper().strip()]
    if name in ("stats", "forecast"):
        return [str(s).upper().strip() for s in (args.get("symbols") or [])]
    return []
