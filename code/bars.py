"""Daily bars from IBKR through the framework, cached in data/bars/<SYMBOL>.json.

    python bars.py SPY AAPL ...        # fetch or refresh
    python bars.py --file universe.txt

A cache file holds completed sessions only (today's partial bar is dropped while the market is open).
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

DATA = Path(__file__).resolve().parent / "data" / "bars"
ET = ZoneInfo("America/New_York")


def path(symbol):
    return DATA / f"{symbol.upper().replace(' ', '_')}.json"


def load(symbol):
    p = path(symbol)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def completed(rows):
    """Drop today's bar unless the session has closed (16:00 ET)."""
    now = datetime.now(ET)
    today = now.strftime("%Y%m%d")
    if now.hour < 16:
        rows = [r for r in rows if r["date"] != today]
    return rows


def fetch(gw, symbol, duration="2 Y", what="ADJUSTED_LAST"):
    rows = gw.bars(symbol, duration=duration, bar_size="1 day", what=what, rth=True)
    rows = completed([{"date": str(r["date"])[:8], "open": r["open"], "high": r["high"], "low": r["low"],
                       "close": r["close"], "volume": r["volume"]} for r in rows])
    DATA.mkdir(parents=True, exist_ok=True)
    doc = {"symbol": symbol.upper(), "what": what, "fetched": datetime.now(ET).isoformat(timespec="seconds"),
           "bars": rows}
    path(symbol).write_text(json.dumps(doc), encoding="utf-8")
    return doc


def refresh(gw, symbol, duration="1 M", what="ADJUSTED_LAST"):
    """Fetch recent bars and merge them into the cache (new dates added, overlapping dates replaced).

    ADJUSTED_LAST history is re-based on every dividend or split, so if the overlap disagrees with the
    cache by more than 0.05% the whole history is refetched instead of merged."""
    old = load(symbol)
    if not old or old.get("what") != what:
        return fetch(gw, symbol, "3 Y", what)
    rows = gw.bars(symbol, duration=duration, bar_size="1 day", what=what, rth=True)
    new = completed([{"date": str(r["date"])[:8], "open": r["open"], "high": r["high"], "low": r["low"],
                      "close": r["close"], "volume": r["volume"]} for r in rows])
    have = {b["date"]: b for b in old["bars"]}
    for b in new:
        o = have.get(b["date"])
        if o and abs(o["close"] / b["close"] - 1) > 0.0005:   # a dividend re-bases history: refetch it all
            return fetch(gw, symbol, "3 Y", what)
    for b in new:
        have[b["date"]] = b
    old["bars"] = [have[d] for d in sorted(have)]
    old["fetched"] = datetime.now(ET).isoformat(timespec="seconds")
    path(symbol).write_text(json.dumps(old), encoding="utf-8")
    return old


def load_market(symbols):
    """{symbol: {date: bar}} and the sorted calendar of all dates seen."""
    bars, dates = {}, set()
    for s in symbols:
        doc = load(s)
        if not doc:
            continue
        bars[doc["symbol"]] = {b["date"]: {k: b[k] for k in ("open", "high", "low", "close", "volume")}
                               for b in doc["bars"]}
        dates.update(bars[doc["symbol"]])
    return bars, sorted(dates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--file")
    ap.add_argument("--duration", default="2 Y")
    ap.add_argument("--what", default="ADJUSTED_LAST")
    a = ap.parse_args()
    syms = list(a.symbols)
    if a.file:
        syms += [l.split("#")[0].strip() for l in Path(a.file).read_text().splitlines() if l.split("#")[0].strip()]
    from tws import Gateway, GatewayError
    with Gateway("trader-training", timeout=30) as gw:
        for s in syms:
            for attempt in range(3):
                try:
                    doc = fetch(gw, s, a.duration, a.what)
                    print(s, len(doc["bars"]), doc["bars"][0]["date"] if doc["bars"] else "-", flush=True)
                    break
                except GatewayError as e:
                    if e.code in (162, 420) and attempt < 2:   # pacing: wait and try again
                        time.sleep(15)
                        continue
                    print(s, "ERROR", e, file=sys.stderr, flush=True)
                    break


if __name__ == "__main__":
    main()
