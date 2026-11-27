"""The 16:15 record's daily.jsonl rows carry "marks": the price paper.book valued each held name at (None when IBKR gave
no mark, and nav is then None), so a day split reconciles to the cent (the Researcher, 20 Nov). The real live.record and
paper.book on temporary ledger files and a fake gateway; offline, nothing under runs/ is written.

    python record_marks_test.py
"""
import contextlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import live  # noqa: E402
import paper  # noqa: E402

ok = []


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail and not good else ""))


tmp = Path(tempfile.mkdtemp(prefix="record_marks_"))
for k, f in (("FILLS", "fills.jsonl"), ("ORDERS", "orders.jsonl"), ("SLEEVES", "sleeves.json"),
             ("DIVIDENDS", "dividends.jsonl"), ("SPLITS", "splits.jsonl")):
    setattr(paper, k, tmp / f)
live.LIVE = tmp / "live"
paper.SLEEVES.write_text(json.dumps({
    "a": {"start_cash": 30000.0, "start": "2026-11-23T12:00:00+00:00", "model": "m"},
    "b": {"start_cash": 30000.0, "start": "2026-11-23T12:00:00+00:00", "model": "m"},
    "c": {"start_cash": 30000.0, "start": "2026-11-23T12:00:00+00:00", "model": "m"}}), encoding="utf-8")
fill = lambda n, sl, sym, side, sh, px: {"exec_id": f"e{n}", "time": "20261123 10:00:00 US/Eastern", "symbol": sym,
                                         "side": side, "shares": sh, "price": px, "commission": 1.0, "sleeve": sl}
for r in (fill(1, "a", "AAA", "BOT", 10, 100.0), fill(2, "a", "BBB", "SLD", 5, 50.0), fill(3, "a", "CCC", "BOT", 3, 20.0),
          fill(4, "a", "CCC", "SLD", 3, 21.0), fill(5, "b", "AAA", "BOT", 2, 100.0), fill(6, "b", "ZZZ", "BOT", 4, 10.0)):
    paper.append(paper.FILLS, r)

QUOTES = {"AAA": {"last": 101.5}, "BBB": {"bid": 48.0, "ask": 48.5}, "SPY": {"last": 600.0},
          "ZZZ": {"error": "no market data"}}


class GW:
    def quotes(self, syms):
        return [{"symbol": s, **QUOTES.get(s, {"error": "unknown"})} for s in syms]

    def account(self):
        return {"NetLiquidation": 720000.0}


@contextlib.contextmanager
def fake_gateway():
    yield GW()


live.gateway = fake_gateway
paper.sync_fills = lambda gw: None
live.reconcile = lambda gw: None
live.book_dividends = lambda gw=None: None
live.macro_rebuild = lambda: None
live.log = lambda *a, **k: None
live.record()

rows = {r["sleeve"]: r for r in paper.read(live.LIVE / "daily.jsonl")}
a, b, c = rows["a"], rows["b"], rows["c"]
check("a row a sleeve", set(rows) == {"a", "b", "c"}, sorted(rows))
check("marks: one per held name, a closed name absent, last else mid", a["marks"] == {"AAA": 101.5, "BBB": 48.25}, a["marks"])
check("marks: the names of positions, and only those", set(a["marks"]) == set(a["positions"]), (a["marks"], a["positions"]))
check("marks reconcile nav to the cent: cash + sum(quantity x mark)",
      abs(a["nav"] - (a["cash"] + sum(q * a["marks"][s] for s, q in a["positions"].items()))) < 0.005, a)
check("an unmarked name is null, and nav is null with it", b["marks"] == {"AAA": 101.5, "ZZZ": None} and b["nav"] is None, b)
check("a sleeve in cash: marks empty", c["marks"] == {} and c["positions"] == {} and c["nav"] == 30000.0, c)
check("SPY stays its own field, not a mark unless held", a["spy"] == 600.0 and "SPY" not in a["marks"], a)
check("the row as before otherwise: the fields written before 23 Nov all kept",
      {"date", "sleeve", "model", "nav", "cash", "start_cash", "realized", "unrealized", "commissions", "fills",
       "missing_commissions", "positions", "spy", "account_net_liquidation", "ml_tool"} <= set(a), sorted(a))
check("a mark survives the JSON line as a number", json.loads(json.dumps(a))["marks"]["BBB"] == 48.25)

print(f"\nrecord_marks_test.py: {'ALL OK' if all(ok) else f'{ok.count(False)} FAILED'}")
sys.exit(0 if all(ok) else 1)
