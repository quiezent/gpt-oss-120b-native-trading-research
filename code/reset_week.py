"""Start a new comparison week: flatten the paper account and restart every sleeve from cash.

Clayton, 17 Oct 2026: close all positions (the sleeves' and the unattributed leftovers), then start all six
sleeves (base, rl1, opus and their -ml arms) from $30,000 cash on Monday, with no positions, notes or history.

    python reset_week.py week1            # show the plan
    python reset_week.py week1 --go       # do it (client 10; not while a look is running)
    python reset_week.py week2 --rl rl4=tinker://.../sampler_weights/rl4-stepNNN --go    # week 3 (Clayton, 24 Oct)
    python reset_week.py week3 --carry base-ml --new runs/week4_sleeves.json --go           # week 4 (Clayton, 1 Nov)

Steps: cancel every open order of this client (the sleeves' GTC orders can rest over the weekend); move the week's
ledgers to runs/archive/<week>/; write the new sleeves. paper.sync_fills never re-books an archived execution.

Positions are not touched here: closing the account's holdings and the IBKR reset are the TWS Gateway Owner's
(Clayton, 26 Oct; the reset is on Monday before the open, the weekend being for analysis and training). Monday:
create runs/live/HOLD, run this after the Gateway's 06:00 ET restart, check that no open order is left, and tell the
Owner "reset" with Clayton's amount. The Owner closes every position pre-market from client 1 (refs
reset-YYYYMMDD-SYMBOL), resets, and says "go" once the Gateway shows the new cash and no positions; then remove HOLD.
No week-4 order goes in before the go.
"""

import argparse
import json
import shutil
from pathlib import Path

import paper
from tws import Gateway

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
LEDGERS = [RUNS / "sleeves.json", RUNS / "fills.jsonl", RUNS / "orders.jsonl", RUNS / "looks.json", RUNS / "notes.json",
           RUNS / "live" / "memo.json", RUNS / "live" / "daily.jsonl", RUNS / "live" / "research.jsonl",
           paper.DIVIDENDS]
START_CASH = 30000.0
NEW = {"base": "base", "opus": "claude-opus-5-5"}     # plus the RL sleeve from --rl name=sampler
EFFORT = {"base": "high", "opus": "max"}             # every sleeve at maximum effort (gpt-oss high, Opus max); RL: high


def carry_rows(f, carry):
    """Keep the carried sleeves' rows of one ledger in place (the whole file is already archived)."""
    if not f.exists() or f.name == "sleeves.json":
        return
    if f.suffix == ".jsonl":
        rows = [r for r in paper.read(f) if r.get("sleeve") in carry]
        f.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    else:                                   # looks.json, notes.json, memo.json: {sleeve: ...}
        d = json.loads(f.read_text(encoding="utf-8"))
        f.write_text(json.dumps({k: v for k, v in d.items() if k in carry}, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("week")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--rl", help="the RL sleeve as name=tinker sampler path, e.g. rl4=tinker://...; "
                    "name=hold starts it in cash with no looks until its model is set in sleeves.json (live.HOLD_MODEL)")
    ap.add_argument("--carry", nargs="*", default=[], help="sleeves that continue: their orders stay working and their "
                    "rows stay in the live ledgers (Clayton, 1 Nov: base-ml runs 4 weeks)")
    ap.add_argument("--new", help="the new sleeves as JSON {name: {start_cash, model, ml, effort, ml_tool, label}}, instead of --rl's "
                    "base/rl/opus pairs")
    a = ap.parse_args()
    if not (a.rl or a.new):
        ap.error("give --rl or --new")
    carry = set(a.carry)
    old = paper.sleeves()
    if carry - set(old):
        raise SystemExit(f"not a sleeve now: {sorted(carry - set(old))}")
    dest = paper.ARCHIVE / a.week
    if dest.exists():
        raise SystemExit(f"{dest} exists")
    with Gateway("trader-training", timeout=30) as gw:
        paper.sync_fills(gw)
        live = [o for o in gw.orders() if o.get("status") not in ("Filled", "Cancelled", "ApiCancelled", "Inactive")]
        kept = [o for o in live if paper.sleeve_of(o.get("ref")) in carry]
        live = [o for o in live if o not in kept]
        show = lambda os_: [(o.get("ref"), o.get("symbol"), o.get("action"), o.get("quantity"), o.get("order_type")) for o in os_]
        print(f"cancel {len(live)} open orders: {show(live)}")
        print(f"keep {len(kept)} open orders of {sorted(carry)}: {show(kept)}")
        held = {s: {k: p["quantity"] for k, p in paper.book(s)["positions"].items()} for s in old}
        print(f"sleeve positions at the reset (released by the TWS Gateway Owner, carried ones kept): {held}")
        if not a.go:
            return
        for o in live:
            r = gw.cancel(order_id=o["order_id"])
            print("  cancel", o.get("ref"), o.get("symbol"), "->", (r or {}).get("status", r))
    dest.mkdir(parents=True)
    for f in LEDGERS:
        if f.exists():
            if carry:                       # the week stays whole in the archive; the carried rows stay live too
                shutil.copy2(str(f), str(dest / f.name))
                carry_rows(f, carry)
            else:
                shutil.move(str(f), str(dest / f.name))
    sl = {k: old[k] for k in old if k in carry}
    if a.new:
        for name, m in json.loads(Path(a.new).read_text(encoding="utf-8")).items():
            sl[name] = {"ml": False, "effort": "high", **m, "start_cash": float(m["start_cash"]), "start": paper.now_iso()}
    else:
        rl_name, rl_model = a.rl.split("=", 1)
        for name, model in {"base": NEW["base"], rl_name: rl_model, "opus": NEW["opus"]}.items():   # last week's look order
            for arm in ("", "-ml"):
                sl[name + arm] = {"start_cash": START_CASH, "start": paper.now_iso(), "model": model, "ml": arm == "-ml",
                                  "effort": EFFORT.get(name, "high")}
    (RUNS / "sleeves.json").write_text(json.dumps(sl, indent=1), encoding="utf-8")
    print(f"archived to {dest}; sleeves now: {list(sl)}")


if __name__ == "__main__":
    main()
