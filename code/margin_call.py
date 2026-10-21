"""Margin call: bring an over-limit sleeve's gross exposure back under its 2x limit by cutting its largest
positions first, at market, as a broker would. Clayton chose this on 16 Oct 2026 for base and rl1, after an
unenforced catch-up look let them borrow the shared paper account's margin.

    python margin_call.py base rl1            # show the plan
    python margin_call.py base rl1 --place    # place it (client 10; not while a look is running)
"""

import argparse
import math

import paper
from tws import Gateway

TARGET = 1.95   # a little under 2x, so the next price move doesn't put the sleeve straight back over


def plan(book, px):
    pos = {s: p["quantity"] for s, p in book["positions"].items()}
    mark = {s: px.get(s) or p["avg_cost"] for s, p in book["positions"].items()}
    nav = book["cash"] + sum(q * mark[s] for s, q in pos.items())
    gross = sum(abs(q) * mark[s] for s, q in pos.items())
    cuts = []
    for s in sorted(pos, key=lambda s: -abs(pos[s]) * mark[s]):
        excess = gross - TARGET * nav
        if excess <= 0:
            break
        n = min(abs(pos[s]), math.ceil(excess / mark[s]))
        cuts.append((s, "SELL" if pos[s] > 0 else "BUY", int(n), mark[s]))
        gross -= n * mark[s]
    return nav, sum(abs(q) * mark[s] for s, q in pos.items()), gross, cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sleeves", nargs="+")
    ap.add_argument("--place", action="store_true")
    a = ap.parse_args()
    with Gateway("trader-training", timeout=30) as gw:
        paper.sync_fills(gw)
        for name in a.sleeves:
            b0 = paper.book(name)
            px = {q["symbol"]: paper.mark(q) for q in gw.quotes(sorted(b0["positions"])) if not q.get("error")}
            nav, g0, g1, cuts = plan(paper.book(name, px), px)
            print(f"{name}: value ${nav:,.0f}, gross {g0 / nav:.2f}x -> {g1 / nav:.2f}x; cuts {[(s, act, n) for s, act, n, _ in cuts]}")
            if a.place and cuts:
                notes = []
                for s, act, n, p in cuts:
                    r = paper.place(gw, name, s, act, n, "MKT", None, None, "DAY")
                    st = (r.get("result") or {}).get("status", r.get("error"))
                    print(f"  {act} {n} {s} MKT: {st}")
                    notes.append(f"margin call: your gross exposure was {g0 / nav:.2f}x, above the 2x limit, so {act} {n} {s} "
                                 f"was placed at market for you (largest positions first) to bring it to about {TARGET:g}x")
                paper.add_notes(name, notes)


if __name__ == "__main__":
    main()
