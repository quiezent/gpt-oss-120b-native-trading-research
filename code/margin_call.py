"""Margin cut: bring a sleeve's gross exposure down to a target multiple of its value by cutting its largest positions
first, at market, as a broker would, booked to the sleeve. Only Clayton orders one (CLAUDE.md).

- 16 Oct 2026: Clayton chose this for base and rl1 under the 2x limit of the time (target 1.95x).
- Tue 17 Nov 2026: with no software exposure limit since 16 Nov, the shared paper account fell below IBKR's margin again
  (SMA negative from 10:00 ET; on Mon 16 Nov IBKR liquidated at 15:51 ET). Clayton, ~11:50 ET: change the warning's
  wording, and "If the change in wording don't work, try order a margin cut on Base B Candidate 2 and Base A". The target
  is the account's maximum his wording states, the account's gross over its net liquidation at IBKR's read (--target).

    python margin_call.py base-90k-b cand2 base-90k --target 2.9            # the plan, read-only on client 11
    python margin_call.py base-90k-b cand2 base-90k --target 2.9 --place    # place it on client 10, between looks only

Each cut is a MKT DAY order through paper.place (the sleeve's own ref, so the fill is booked to it). The sleeve's model gets
one framework note: what was cut and why, and its orders still resting in those names (placed before the cut). It refuses
to place while a look runs (runs/live/LOOKING*) or from 3 minutes before a look (:00 and :30, 09:00-15:30 ET) to 12 after.
"""

import argparse
import math
import sys
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import paper
import live  # noqa: F401 - the TWS framework's path
from tws import Gateway

ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
TARGET = 1.95   # the 16 Oct default: a little under 2x, so the next price move doesn't put the sleeve straight back over


def plan(book, px, target=TARGET):
    pos = {s: p["quantity"] for s, p in book["positions"].items()}
    mark = {s: px.get(s) or p["avg_cost"] for s, p in book["positions"].items()}
    nav = book["cash"] + sum(q * mark[s] for s, q in pos.items())
    gross = sum(abs(q) * mark[s] for s, q in pos.items())
    cuts = []
    for s in sorted(pos, key=lambda s: -abs(pos[s]) * mark[s]):
        excess = gross - target * nav
        if excess <= 0:
            break
        n = min(abs(pos[s]), math.ceil(excess / mark[s]))
        cuts.append((s, "SELL" if pos[s] > 0 else "BUY", int(n), mark[s]))
        gross -= n * mark[s]
    return nav, sum(abs(q) * mark[s] for s, q in pos.items()), gross, cuts


def window_ok(now):
    """Not while a look runs, nor from 3 minutes before a look (:00 and :30, 09:00-15:30 ET) to 12 minutes after."""
    if list((ROOT / "runs" / "live").glob("LOOKING*")):
        return False, "a look is running (runs/live/LOOKING*)"
    t = now.hour * 60 + now.minute
    for h in range(9, 16):
        for m in (0, 30):
            k = h * 60 + m
            if k - 3 <= t < k + 12:
                return False, f"inside the {h:02d}:{m:02d} look's window ({h:02d}:{m:02d} -3/+12 minutes)"
    return True, ""


def note_for(name, g0x, target, cuts, resting, figs):
    text = (f"framework note (margin cut, {datetime.now(ET):%a %d %b %H:%M} ET): the whole paper account was below IBKR's margin "
            f"({figs}), and Clayton ordered a margin cut on your sleeve: your gross exposure was {g0x:.2f}x its value, above "
            f"your account's maximum of {target:g}x, so these were placed for you at market, largest positions first, to bring "
            "it to about that: " + "; ".join(f"{act} {n} {s}" for s, act, n, _ in cuts) + ".")
    names = {s for s, _, _, _ in cuts}
    mine = [o for o in resting if o.get("symbol") in names]
    if mine:
        text += (" Your orders still resting in those names were placed before the cut: "
                 + "; ".join(f"{o.get('action')} {o.get('quantity', 0):g} {o.get('symbol')} {o.get('order_type')}"
                             + (f" limit {o['limit']:g}" if o.get("limit") else "") + (f" stop {o['aux']:g}" if o.get("aux") else "")
                             + f" {o.get('tif') or ''} [{o.get('ref')}]" for o in mine) + ".")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sleeves", nargs="+")
    ap.add_argument("--target", type=float, default=TARGET, help="gross as a multiple of the sleeve's value to cut to")
    ap.add_argument("--place", action="store_true")
    a = ap.parse_args()
    unknown = [s for s in a.sleeves if s not in paper.sleeves()]
    if unknown:
        sys.exit(f"no such sleeve: {unknown}")
    if a.place:
        ok, why = window_ok(datetime.now(ET))
        if not ok:
            sys.exit(f"not placing: {why}")
    with Gateway("trader-training" if a.place else "trader-research", timeout=30) as gw:
        if a.place:
            paper.sync_fills(gw)
        acct = gw.account()
        figs = "; ".join(f"{n} ${float(acct[k]):,.0f}" for k, n in (("SMA", "SMA"), ("ExcessLiquidity", "excess liquidity"),
                                                                     ("NetLiquidation", "net liquidation"))
                         if acct.get(k) not in (None, ""))
        print(f"account: {figs}")
        rest_all = gw.all_open_orders()
        for name in a.sleeves:
            b0 = paper.book(name)
            px = {q["symbol"]: paper.mark(q) for q in gw.quotes(sorted(b0["positions"])) if not q.get("error")}
            nav, g0, g1, cuts = plan(paper.book(name, px), px, a.target)
            print(f"{name}: value ${nav:,.0f}, gross ${g0:,.0f} = {g0 / nav:.2f}x -> {g1 / nav:.2f}x; "
                  f"cuts {[(s, act, n, round(n * p)) for s, act, n, p in cuts]}")
            if a.place and cuts:
                for s, act, n, p in cuts:
                    r = paper.place(gw, name, s, act, n, "MKT", None, None, "DAY")
                    st = (r.get("result") or {}).get("status", r.get("error"))
                    print(f"  {act} {n} {s} MKT: {st}")
                resting = [o for o in rest_all if paper.sleeve_of(o.get("ref")) == name]
                paper.add_notes(name, [note_for(name, g0 / nav, a.target, cuts, resting, figs)])


if __name__ == "__main__":
    main()
