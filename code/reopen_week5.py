"""Re-open Base + ML2.2's 17 carried positions after the $1M paper reset (Clayton, 9 Nov 2026).

    python reopen_week5.py            # show the plan and the checks; places nothing
    python reopen_week5.py --go       # place them (client 10), only after the TWS Gateway Owner's "go"

The 17 come from runs/live/owner_week5_reset.md (the block after "After the reset"), and must equal base-ml's book in
paper.book("base-ml"). Orders go on client 10 (trader-training) under refs reopen-20261109-<SYM>: paper.sleeve_of
maps those to no sleeve, so their fills don't double-book into base-ml's ledger, which already holds the positions at
its own cost basis (the slippage between the pre-market close and the open is the framework's). Before placing, it
checks that the account is flat, that client 10 has no open order (it sees only its own; the TWS Gateway Owner's go
covers the other clients), and that it shows about $1,000,000 cash; it refuses otherwise. Before 09:28 ET the orders are
MKT with tif OPG (the opening auction, as live sends MOO); after that MKT DAY. A failed symbol is reported and the rest
are still placed.

9 Nov, 19:10 UTC+8: IBKR's Portal took at most 720,000, so the reset set cash to about $723k; run with --min-cash 715000.
--allow-held SYM,SYM lets names that client 1's reset closes are still working on stay held; none may be one of the 17.
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
REF = "reopen-20261109-{}"


def plan():
    import paper
    text = (ROOT / "runs" / "live" / "owner_week5_reset.md").read_text(encoding="utf-8")
    part = text[text.index("## After the reset"):]
    want = json.loads(re.search(r"```json\n(\{.*?\})\n```", part, re.S).group(1))
    book = {s: p["quantity"] for s, p in paper.book("base-ml")["positions"].items()}
    if want != book:
        raise SystemExit(f"the list and base-ml's book differ: list {want} book {book}")
    return want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--min-cash", type=float, default=990000.0)
    ap.add_argument("--allow-held", default="", help="comma list of names client 1 is still closing; never one of the 17")
    a = ap.parse_args()
    want = plan()
    allow = {s.strip().upper() for s in a.allow_held.split(",") if s.strip()}
    if allow & set(want):
        raise SystemExit(f"--allow-held names one of the 17: {sorted(allow & set(want))}")
    import paper
    from tws import Gateway
    now = datetime.now(ET)
    tif = "OPG" if (now.hour, now.minute) < (9, 28) else "DAY"     # market-on-open is MKT with tif OPG, as live sends it
    with Gateway("trader-training", timeout=30) as gw:
        held = {p["symbol"]: p["quantity"] for p in gw.positions() if p.get("quantity")}
        live = [o for o in gw.orders() if o.get("status") not in ("Filled", "Cancelled", "ApiCancelled", "Inactive")]
        acct = gw.account() or {}
        cash = float(acct.get("TotalCashValue") or 0)
        print(f"account: cash {cash:,.2f}, {len(held)} positions, {len(live)} open orders on client 10; {now:%H:%M:%S} ET; MKT {tif}")
        print("plan:", {s: q for s, q in want.items()})
        bad = []
        closing = {s: q for s, q in held.items() if s in allow}
        if closing:
            print(f"still closing on client 1 (allowed): {closing}")
        held = {s: q for s, q in held.items() if s not in allow}
        if held:
            bad.append(f"the account isn't flat: {held}")
        if live:
            bad.append(f"open orders: {[(o.get('ref'), o.get('symbol')) for o in live]}")
        if cash < a.min_cash:
            bad.append(f"cash {cash:,.0f} is below {a.min_cash:,.0f}: the reset hasn't shown")
        if bad:
            print("NOT PLACING:", "; ".join(bad))
            return 1
        if not a.go:
            print("checks pass; rerun with --go to place")
            return 0
        failed = []
        for s, q in want.items():
            assert paper.sleeve_of(REF.format(s)) is None
            try:                                        # one symbol's error doesn't stop the rest
                r = gw.place(s, "BUY" if q > 0 else "SELL", abs(q), order_type="MKT", tif=tif, ref=REF.format(s))
                print("  placed", s, q, f"MKT {tif}", "->", (r or {}).get("status", r))
            except Exception as e:                      # noqa: BLE001
                failed.append(s)
                print("  FAILED", s, q, f"{type(e).__name__}: {str(e)[:150]}")
        if failed:
            print(f"{len(failed)} not placed: {failed}. Place those by hand under the same refs (reopen-20261109-<SYM>);"
                  " a rerun refuses while the others are open.")
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
