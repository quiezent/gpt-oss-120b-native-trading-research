"""Daily implied volatility (OPTION_IMPLIED_VOLATILITY) and bid/ask (BID_ASK: open = time-average bid, close =
time-average ask) bars, 2 years, for the universe -> data/iv/<SYM>.json and data/bidask/<SYM>.json. For the ML tools
(vol/range blending, regime, cost model). Read-only client "trader-research"; resumable; reconnects after the
Gateway's daily restart. SPY, QQQ and IWM go first (the regime model needs them).

    python fetch_extra.py
    python fetch_extra.py --bars30 "2 M" [stats.json]     # re-fetch the universe's 30-minute bars (merge_bars30)

update() is also the daily refresh: iv and bidask at 07:30, bars30 (the last 5 sessions) at the 16:15 record.
"""

import json
import os
import sys
import time
from pathlib import Path
from statistics import median

import trader
from tws import Gateway

ROOT = Path(__file__).resolve().parent
KINDS = {"iv": "OPTION_IMPLIED_VOLATILITY", "bidask": "BID_ASK"}
SPACING = 1.5      # daily bars: IBKR's strict pacing applies only to bars of 30 seconds or less
SCALE_TOL = 0.02   # a session whose bars sit this far from the stored ones (median close ratio) is on another price scale:
                   # a split since, which IBKR's history adjusts for. IBKR's overnight revisions are cents: p99 7 bp, the
                   # largest 40 bp, over 19 Oct - 19 Nov (the Researcher's lens D-5, 21 Nov)


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


def _slot(b):
    return str(b["date"])[:17]                      # '20261120 09:30:00' (the zone label varies: US/Eastern, America/New_York)


def _values(b):
    return (_slot(b),) + tuple(float(b[k] or 0) for k in ("open", "high", "low", "close", "volume"))


def merge_bars30(old, new, split_dates=()):
    """(bars, stats): IBKR's 30-minute bars `new` merged into the stored `old`, one session at a time.

    The stored file is as traded: since the bulk fetch of 17 Oct each session was added on the day it traded, and
    carry_record's day factor k (09:30 bar open over the adjusted daily open) needs the price as traded. IBKR revises its
    history overnight (a session fetched after the close differs from the next day's by about a cent in 28% of bars,
    19 Oct - 19 Nov), and its history is adjusted for any split since. So a session of `new`:
    - on the stored scale (median close ratio within SCALE_TOL) replaces the stored session ("revised" if any bar changed,
      else "same"), unless IBKR now has fewer bars of it ("short": the stored one is kept);
    - on another scale (a split since) leaves the stored session as it is ("rescaled");
    - not stored is added ("added"), so a session a failed fetch missed is filled the next time it is in the window,
      unless IBKR's price for it is split-adjusted and so not as traded: it falls before a rescaled session, or before a
      split of `split_dates` (the split ledger's ex-dates, YYYYMMDD) inside the reply ("before_split"); or it falls
      between the last rescaled session and the next session on the stored scale, where the split's ex-date lies, and
      the ledger does not say which side it is on ("ambiguous").
    With no stored session in the reply's window and no ledger row, a split inside the window can't be seen: such
    sessions are added at IBKR's adjusted prices (no universe name is in that state: all 523 files are kept current).
    A stored session IBKR no longer serves is kept."""
    by_old, by_new = {}, {}
    for b in old:
        by_old.setdefault(_slot(b)[:8], []).append(b)
    for b in new:
        by_new.setdefault(_slot(b)[:8], []).append(b)
    st = {"revised": [], "same": 0, "added": [], "rescaled": [], "before_split": [], "ambiguous": [], "short": []}
    scale_ok = {}
    for d in by_new.keys() & by_old.keys():
        o = {_slot(b): b["close"] for b in by_old[d]}
        rs = [b["close"] / o[_slot(b)] for b in by_new[d] if (o.get(_slot(b)) or 0) > 0 and (b["close"] or 0) > 0]
        if not rs:                                  # no common slot: the sessions' median closes
            oc, nc = [b["close"] for b in by_old[d] if (b["close"] or 0) > 0], [b["close"] for b in by_new[d] if (b["close"] or 0) > 0]
            rs = [median(nc) / median(oc)] if oc and nc else [1.0]
        scale_ok[d] = abs(median(rs) - 1) <= SCALE_TOL
    last_rescaled = max((d for d, ok in scale_ok.items() if not ok), default=None)
    reply_last = max(by_new, default=None)
    edge = None                                     # the ex-date of the split IBKR applied is in (last_rescaled, edge]
    if last_rescaled:
        edge = min((d for d in by_old if d > last_rescaled), default=reply_last)
    exs = sorted(str(e) for e in split_dates)
    out = dict(by_old)
    for d, bars in by_new.items():
        if d in by_old:
            if not scale_ok[d]:
                st["rescaled"].append(d)
            elif len(bars) < len(by_old[d]):
                st["short"].append(d)
            elif sorted(map(_values, bars)) != sorted(map(_values, by_old[d])):
                out[d] = bars
                st["revised"].append(d)
            else:
                st["same"] += 1
        elif (last_rescaled and d < last_rescaled) or any(d < e <= reply_last for e in exs):
            st["before_split"].append(d)
        elif last_rescaled and d < edge and not any(last_rescaled < e <= d for e in exs):
            st["ambiguous"].append(d)
        else:
            out[d] = bars
            st["added"].append(d)
    return sorted((b for d in out for b in out[d]), key=lambda b: (_slot(b), str(b["date"]))), st


def _write(p, doc):
    """Write via a temp file and os.replace, so no reader (a look's forecast call, warm_outside, an MCP server) and no
    later merge sees part of a file. A reader holding the file open can make the replace fail on Windows: retried."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(doc), encoding="utf-8")
        for i in range(5):
            try:
                os.replace(tmp, p)
                return
            except PermissionError:
                if i == 4:
                    raise
                time.sleep(0.2 * (i + 1))
    finally:
        tmp.unlink(missing_ok=True)


def update(gw, symbols, kind, duration="1 M", stats=None):
    """Merge the last `duration` of `kind` bars into data/<kind>/<SYM>.json (the daily refresh). Returns failures.

    bars30 merges one session at a time (merge_bars30, with the split ledger's ex-dates); with `stats` (a dict) each
    symbol written gets its session lists as stats[symbol]. Other kinds replace stored dates with the fetched ones."""
    bad = []
    exs = {}
    if kind == "bars30":
        try:                                        # record() books today's splits (book_dividends) before this runs
            import paper
            for r in paper.splits():
                exs.setdefault(r["symbol"], []).append(str(r["ex_date"]))
        except Exception as e:                     # noqa: BLE001 - without the ledger a split is still seen from the bars
            bad.append(f"split ledger not read: {type(e).__name__}: {str(e)[:60]}")
    for s in symbols:
        p = ROOT / "data" / kind / f"{s.replace(' ', '_')}.json"
        try:
            if kind == "bars30":                    # the day's 30-minute TRADES bars (after the close)
                new = gw.bars(s, duration=duration, bar_size="30 mins", what="TRADES", rth=True)
            else:
                new = gw.bars(s, duration=duration, bar_size="1 day", what=KINDS[kind], rth=True)
            old = json.loads(p.read_text(encoding="utf-8"))["bars"] if p.exists() else []
            st = None
            if kind == "bars30":
                bars, st = merge_bars30(old, new, exs.get(s, ()))
            else:
                merged = {str(r["date"]): r for r in old}
                merged.update({str(r["date"]): r for r in new})
                bars = [merged[k] for k in sorted(merged)]
            _write(p, {"symbol": s, "what": KINDS.get(kind, "TRADES"), "bars": bars})
            if st is not None and stats is not None:
                stats[s] = st
        except Exception as e:                     # noqa: BLE001
            bad.append(f"{s}: {str(e)[:80]}")
        time.sleep(0.5)
    return bad


def bars30_summary(stats, bad, today):
    """One log line from update(..., "bars30", stats=...); today is 'YYYYMMDD'."""
    tot = {k: sum(len(v[k]) if isinstance(v[k], list) else v[k] for v in stats.values()) for k in
           ("revised", "same", "added", "rescaled", "before_split", "ambiguous", "short")}
    filled = sorted(f"{s} {d}" for s, v in stats.items() for d in v["added"] if d != today)
    kept = sorted({s for s, v in stats.items() if v["rescaled"]})
    unsure = sorted(f"{s} {d}" for s, v in stats.items() for d in v["before_split"] + v["ambiguous"])
    return (f"30-minute bars: {tot['added'] - len(filled)} sessions of {today} added; {tot['revised']} earlier sessions "
            f"revised by IBKR since fetched, {tot['same']} unchanged; {len(filled)} missed sessions filled {filled[:5]}; "
            f"{tot['rescaled']} sessions kept as traded after a split {kept[:5]}; {len(unsure)} missed sessions not filled "
            f"(IBKR's price is split-adjusted, or may be) {unsure[:5]}; {tot['short']} kept (IBKR now has fewer bars); "
            f"{len(bad)} failed {bad[:3]}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--bars30"]:   # one-off re-fetch of the universe's bars30 window (client 11, read-only):
        from datetime import datetime   #   python fetch_extra.py --bars30 "2 M" [stats.json]
        from zoneinfo import ZoneInfo
        syms = sorted(set(trader.universe()))
        stats = {}
        with Gateway("trader-research", timeout=120) as gw:
            bad = update(gw, syms, "bars30", duration=sys.argv[2] if len(sys.argv) > 2 else "5 D", stats=stats)
        print(bars30_summary(stats, bad, datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")))
        if len(sys.argv) > 3:
            Path(sys.argv[3]).write_text(json.dumps({"stats": stats, "bad": bad}, indent=1), encoding="utf-8")
    else:
        main()
