"""Fetch 1 year (2 years for the first few symbols) of regular-hours 30-minute TRADES bars for the universe into data/bars30/<SYM>.json, for the
ML arm's research (read-only client "trader-research"). Resumable: symbols already fetched are skipped.

    python fetch_intraday.py
"""

import json
import time
from pathlib import Path

import trader
from tws import Gateway

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "bars30"
SPACING = 10.5   # IBKR: at most 60 historical-data requests in any 10 minutes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    syms = sorted(set(trader.universe()))
    todo = [s for s in syms if not (OUT / f"{s.replace(' ', '_')}.json").exists()]
    print(f"{len(todo)} of {len(syms)} to fetch", flush=True)
    gw = None
    for i, s in enumerate(todo):
        for attempt in range(6):                  # the Gateway restarts daily (18:00 UTC+8): reconnect and retry
            t0 = time.time()
            try:
                if gw is None:
                    gw = Gateway("trader-research", timeout=120).__enter__()
                rows = gw.bars(s, duration="1 Y", bar_size="30 mins", what="TRADES", rth=True)
                (OUT / f"{s.replace(' ', '_')}.json").write_text(json.dumps({"symbol": s, "bars": rows}), encoding="utf-8")
                msg = f"{len(rows)} bars"
                break
            except Exception as e:
                msg = f"ERROR {type(e).__name__}: {str(e)[:120]}"
                if "504" in msg or "connect" in msg.lower():
                    try:
                        gw.__exit__(None, None, None)
                    except Exception:
                        pass
                    gw = None
                    time.sleep(60)
                    continue
                break
        print(f"{i + 1}/{len(todo)} {s}: {msg} ({time.time() - t0:.1f}s)", flush=True)
        time.sleep(max(0.0, SPACING - (time.time() - t0)))


if __name__ == "__main__":
    main()
