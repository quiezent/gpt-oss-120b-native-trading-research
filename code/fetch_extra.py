"""Daily implied volatility (OPTION_IMPLIED_VOLATILITY) and bid/ask (BID_ASK: open = time-average bid, close =
time-average ask) bars, 2 years, for the universe -> data/iv/<SYM>.json and data/bidask/<SYM>.json. For the ML tools
(vol/range blending, regime, cost model). Read-only client "trader-research"; resumable; reconnects after the
Gateway's daily restart. SPY, QQQ and IWM go first (the regime model needs them).

    python fetch_extra.py
"""

import json
import time
from pathlib import Path

import trader
from tws import Gateway

ROOT = Path(__file__).resolve().parent
KINDS = {"iv": "OPTION_IMPLIED_VOLATILITY", "bidask": "BID_ASK"}
SPACING = 1.5      # daily bars: IBKR's strict pacing applies only to bars of 30 seconds or less


def main():
    syms = sorted(set(trader.universe()))
    syms = ["SPY", "QQQ", "IWM"] + [s for s in syms if s not in ("SPY", "QQQ", "IWM")]
    jobs = [(k, s) for s in syms for k in KINDS if not (ROOT / "data" / k / f"{s.replace(' ', '_')}.json").exists()]
    print(f"{len(jobs)} requests", flush=True)
    gw = None
    for i, (k, s) in enumerate(jobs):
        for attempt in range(6):
            t0 = time.time()
            try:
                if gw is None:
                    gw = Gateway("trader-research", timeout=120).__enter__()
                rows = gw.bars(s, duration="2 Y", bar_size="1 day", what=KINDS[k], rth=True)
                p = ROOT / "data" / k / f"{s.replace(' ', '_')}.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"symbol": s, "what": KINDS[k], "bars": rows}), encoding="utf-8")
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
        print(f"{i + 1}/{len(jobs)} {k} {s}: {msg} ({time.time() - t0:.1f}s)", flush=True)
        time.sleep(max(0.0, SPACING - (time.time() - t0)))


def update(gw, symbols, kind, duration="1 M"):
    """Merge the last month of `kind` bars into data/<kind>/<SYM>.json (the daily refresh). Returns failures."""
    bad = []
    for s in symbols:
        p = ROOT / "data" / kind / f"{s.replace(' ', '_')}.json"
        try:
            if kind == "bars30":                    # the day's 30-minute TRADES bars (after the close)
                new = gw.bars(s, duration=duration, bar_size="30 mins", what="TRADES", rth=True)
            else:
                new = gw.bars(s, duration=duration, bar_size="1 day", what=KINDS[kind], rth=True)
            old = json.loads(p.read_text(encoding="utf-8"))["bars"] if p.exists() else []
            merged = {str(r["date"]): r for r in old}
            merged.update({str(r["date"]): r for r in new})
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"symbol": s, "what": KINDS.get(kind, "TRADES"), "bars": [merged[k] for k in sorted(merged)]}), encoding="utf-8")
        except Exception as e:                     # noqa: BLE001
            bad.append(f"{s}: {str(e)[:80]}")
        time.sleep(0.5)
    return bad


if __name__ == "__main__":
    main()
