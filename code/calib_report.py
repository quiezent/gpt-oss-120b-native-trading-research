"""Calibration of the models' per-order probabilities on the paper account (p_fill and p_win in trader.TOOL).

    python calib_report.py [runs_dir]        # default runs/; reads runs_dir and runs_dir/archive/<week>/

Every orders.jsonl row that carries probs (paper.place logs them) is scored per sleeve, with the daily bars in data/bars:
  p_fill  y = 1 if the order's ref has any fill. Resolved once it filled, was cancelled, its DAY/OPG session has a
          bar, or its week is archived (the weekly reset cancels every open order); otherwise pending. Orders IBKR
          refused at placement weren't placed and aren't scored.
  p_win   for filled orders: y = 1 if the close 5 sessions after the session of the order's last fill is beyond its
          average fill price by more than 0.2% in the order's favour (up for a BUY, down for a SELL). Pending until
          that bar exists.
Brier reward 1 - (p - y)^2, a missing p scoring 0 (as reward.calibration_bonus); always 0.5 earns 0.75. data/bars is
ADJUSTED_LAST: a dividend after the window re-bases its closes, so run this soon after the outcomes resolve.
"""

import sys
from datetime import datetime
from pathlib import Path

import bars
import paper
import trader
from tools import ET

REFUSED = ("Cancelled", "Inactive", "ApiCancelled")
BINS = [0.2, 0.4, 0.6, 0.8, 1.0001]


def load(runs):
    dirs = [runs] + sorted(p for p in (runs / "archive").glob("*") if p.is_dir())
    orders, fills, cancels = [], {}, set()
    for i, d in enumerate(dirs):
        for r in paper.read(d / "orders.jsonl"):
            if "cancel" in r:
                if not r.get("error"):
                    cancels.add(r["cancel"])
            elif "probs" in r:
                orders.append({**r, "archived": i > 0})
        for f in paper.read(d / "fills.jsonl"):
            fills[f["exec_id"]] = f
    by_ref = {}
    for f in fills.values():
        by_ref.setdefault(f.get("ref"), []).append(f)
    return orders, by_ref, cancels


def outcomes(runs):
    orders, by_ref, cancels = load(runs)
    spy = [b["date"] for b in (bars.load("SPY") or {}).get("bars", [])]
    closes = {}
    out = []
    for r in orders:
        fs = by_ref.get(r["ref"], [])
        status = (r.get("result") or {}).get("status")
        if not fs and (r.get("error") or status in REFUSED):
            continue                                  # never placed
        day = datetime.fromisoformat(r["time"]).astimezone(ET).strftime("%Y%m%d")
        resolved = bool(fs) or r["archived"] or r["ref"] in cancels or (r["tif"] in ("DAY", "OPG") and any(d >= day for d in spy))
        p = r["probs"] or {}
        row = {"sleeve": r["sleeve"], "p_fill": trader.prob(p.get("p_fill")), "p_win": trader.prob(p.get("p_win")),
               "filled": float(bool(fs)) if resolved else None, "win": None}
        if fs:
            n = sum(f["shares"] for f in fs)
            px = sum(f["shares"] * f["price"] for f in fs) / n
            fd = max(paper.et_date(f["time"]) for f in fs)
            if r["symbol"] not in closes:
                closes[r["symbol"]] = {b["date"]: b["close"] for b in (bars.load(r["symbol"]) or {}).get("bars", [])}
            c = closes[r["symbol"]]
            dates = sorted(c)
            if fd in c and dates.index(fd) + 5 < len(dates):
                c5 = c[dates[dates.index(fd) + 5]]
                row["win"] = float(c5 > px * 1.002) if r["action"] == "BUY" else float(c5 < px * 0.998)
        out.append(row)
    return out


def score(pairs):
    """pairs: [(p or None, y)] -> n, share given, mean Brier reward, Brier score of the given, base rate, mean p."""
    n = len(pairs)
    given = [(p, y) for p, y in pairs if p is not None]
    rew = sum(1 - (p - y) ** 2 for p, y in given) / n
    f = lambda xs: f"{sum(xs) / len(xs):.3f}" if xs else "-"
    return [str(n), f"{len(given) / n:.0%}", f"{rew:.3f}", f([(p - y) ** 2 for p, y in given]), f([y for _, y in pairs]),
            f([p for p, _ in given])]


def main(runs):
    rows = outcomes(runs)
    if not rows:
        print("# Order-probability calibration\n\nNo placed orders with p_fill/p_win logged yet "
              f"(ledgers under {runs}).")
        return
    print("# Order-probability calibration\n")
    print("Brier reward 1 - (p - y)^2 (missing p = 0; always 0.5 = 0.750). brier = mean (p - y)^2 over the given p.\n")
    print("| sleeve | kind | n | pending | given | Brier reward | brier | observed | mean p |")
    print("|---|---|---|---|---|---|---|---|---|")
    rel = []
    for s in sorted({r["sleeve"] for r in rows}):
        mine = [r for r in rows if r["sleeve"] == s]
        for kind, y, todo in (("p_fill", "filled", lambda r: True), ("p_win", "win", lambda r: r["filled"] == 1.0)):
            due = [r for r in mine if todo(r)]
            pairs = [(r[kind], r[y]) for r in due if r[y] is not None]
            pending = len(due) - len(pairs)
            cells = score(pairs) if pairs else ["0", "-", "-", "-", "-", "-"]
            print(f"| {s} | {kind} | {cells[0]} | {pending} | " + " | ".join(cells[1:]) + " |")
            for lo, hi in zip([0.0] + BINS[:-1], BINS):
                b = [(p, v) for p, v in pairs if p is not None and lo <= p < hi]
                if b:
                    rel.append(f"| {s} | {kind} | {lo:.1f}-{min(hi, 1):.1f} | {len(b)} | {sum(p for p, _ in b) / len(b):.2f} | "
                               f"{sum(v for _, v in b) / len(b):.2f} |")
    if rel:
        print("\nReliability (given probabilities only):\n")
        print("| sleeve | kind | bin | n | mean p | observed |")
        print("|---|---|---|---|---|---|")
        print("\n".join(rel))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else paper.RUNS)
