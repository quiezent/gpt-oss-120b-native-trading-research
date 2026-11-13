"""The single-shock sleeve term (sleeves.json "shock_cap", off unless Clayton sets it): trader.shock_losses against an
independent pandas version of the Kelly audit's b2_books.py formulas, PaperWorld.act's rejections (an adding order past
the limit, never a shrinking one, never a hedge that lowers the worst shock, nothing when the term is off) and the state's
line. Offline: data/bars and a fake gateway and paper.

    python shock_cap_test.py
"""
import json
import sys

import numpy as np
import pandas as pd

import trader

ok = []
BEFORE = "20261106"


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail else ""))


def audit_betas(sym, day):                       # b2_books.betas, written again with pandas
    def rets(x):
        b = json.load(open(trader.BARS_DIR / f"{x}.json", encoding="utf-8"))["bars"]
        b = [r for r in b if r.get("close") and r.get("volume", 1)]
        c = np.array([r["close"] for r in b], float)
        return pd.Series(c[1:] / c[:-1] - 1, index=[str(r["date"])[:8] for r in b[1:]])
    df = pd.concat({"i": rets(sym), "m": rets("SPY"), "s": rets("SMH")}, axis=1, join="inner")
    df = df[df.index < day].dropna().tail(120)
    i, m, s = df["i"].values, df["m"].values, df["s"].values
    return np.cov(i, m)[0, 1] / np.var(m, ddof=1), np.cov(i, s)[0, 1] / np.var(s, ddof=1)


book = {"AMD": -20 * 404.40, "NVDA": -25 * 180.0, "TLT": 100 * 90.0}
nav = 30000.0
got = trader.shock_losses(book, nav, BEFORE)
bm = sum(v / nav * audit_betas(k, BEFORE)[0] for k, v in book.items())
bs = sum(v / nav * audit_betas(k, BEFORE)[1] for k, v in book.items())
want = {"SPY falls 10%": 0.10 * bm, "SPY rises 10%": -0.10 * bm, "SMH falls 11%": 0.11 * bs, "SMH rises 11%": -0.11 * bs,
        "its largest short rises 30%": 0.30 * 20 * 404.40 / nav, "its largest long falls 30%": 0.30 * 9000 / nav}
check("the six shocks match the audit's formulas", all(abs(got[k] - want[k]) < 1e-9 for k in want),
      {k: round(100 * v, 2) for k, v in got.items()})
check("a name with too little history takes beta 1", trader._betas("ZZZZ", BEFORE) == (1.0, 1.0))


class FakePaper:
    def __init__(self, pos):
        self.pos, self.placed, self.kept = pos, [], []

    def book(self, sleeve, marks=None):
        return {"positions": {s: {"quantity": q, "avg_cost": 100.0} for s, q in self.pos.items()}, "cash": self.cash}

    def mark(self, q):
        return q["last"]

    def place(self, gw, sleeve, symbol, action, quantity, *a, **k):
        self.placed.append((action, quantity, symbol))
        return {"result": {"status": "Submitted", "messages": []}}

    def add_notes(self, sleeve, notes):
        self.kept += notes

    def cancel(self, gw, sleeve, ref):
        return {}

    def et_today(self):
        return BEFORE


PX = {"AMD": 404.40, "SMH": 447.73, "SPY": 731.73, "TLT": 90.0, "NVDA": 180.0, "MU": 200.0}


class FakeGW:
    def quotes(self, syms):
        return [{"symbol": s, "last": PX[s], "bid": PX[s], "ask": PX[s]} for s in syms]


def world(pos, cap=0.15, nav=30000.0):
    w = trader.PaperWorld(FakeGW(), "test-sleeve", None)
    w.paper = FakePaper(dict(pos))
    w.paper.cash = nav - sum(q * PX[s] for s, q in pos.items())          # NAV at these prices
    w.SHOCK_CAP = cap
    return w


order = lambda act, q, s: {"symbol": s, "action": act, "quantity": q, "type": "MKT", "tif": "DAY"}
# TLT: its long is the binding shock (largest long falls 30%), since its betas are small; $90 a share, NAV $30,000
tl = lambda q: trader.shock_losses({"TLT": q * 90.0}, 30000.0, BEFORE)
check("TLT's binding shock is its own -30%", max(tl(200), key=tl(200).get) == "its largest long falls 30%", tl(200))
w = world({"TLT": 150})
notes = w.act({"place": [order("BUY", 10, "TLT")]})
check("an add within the limit is placed (160 TLT: 14.4%)", w.paper.placed == [("BUY", 10, "TLT")], notes)
w = world({"TLT": 150})
notes = w.act({"place": [order("BUY", 10, "TLT"), order("BUY", 10, "TLT")]})
check("the same submit's earlier order counts: the second add, to 170 TLT (15.3%), is rejected",
      w.paper.placed == [("BUY", 10, "TLT")] and "15.3% of the sleeve" in notes[0] and "largest long falls 30%" in notes[0], notes)
w = world({"AMD": -30})
notes = w.act({"place": [order("SELL", 10, "AMD")]})
name, worst = max(trader.shock_losses({"AMD": -40 * 404.40}, 30000.0, BEFORE).items(), key=lambda kv: kv[1])
check("an add past the limit is rejected, naming its worst shock", not w.paper.placed and f"if {name}," in notes[0]
      and f"{100 * worst:.1f}%" in notes[0], notes)
w = world({"AMD": -40})
notes = w.act({"place": [order("BUY", 10, "AMD")]})
check("a shrinking order is never rejected, even over the limit", w.paper.placed == [("BUY", 10, "AMD")], notes)
# the review of 74dc2b1: a hedge with a limit away from the mark ($100k sleeve, net short chips, SPY-rise the worst shock)
hb = {"AMD": -100, "NVDA": -150, "MU": -40, "SPY": 60}
lb = trader.shock_losses({k: q * PX[k] for k, q in hb.items()}, 100000.0, BEFORE)
check("the hedge case's worst shock is SPY rising 10%, above the cap", max(lb, key=lb.get) == "SPY rises 10%" and lb["SPY rises 10%"] > 0.15,
      round(100 * lb["SPY rises 10%"], 2))
for typ, lim in (("MKT", None), ("LMT", round(PX["SPY"] * 0.98, 2))):
    w = world(hb, nav=100000.0)
    notes = w.act({"place": [dict(order("BUY", 1, "SPY"), type=typ, limit=lim)]})
    check(f"a hedge that lowers the worst shock is placed ({typ}{' 2% under the mark' if lim else ''})",
          w.paper.placed == [("BUY", 1, "SPY")], notes)
# a flip through zero to a position no larger is checked: long 20 AMD beside short 60 NVDA, SELL 40 AMD -> short 20 AMD
fb = {"AMD": 20, "NVDA": -60}
pre = max(trader.shock_losses({k: q * PX[k] for k, q in fb.items()}, 30000.0, BEFORE).values())
post = max(trader.shock_losses({"AMD": -20 * PX["AMD"], "NVDA": -60 * PX["NVDA"]}, 30000.0, BEFORE).values())
check("the flip case raises the worst shock past the cap", post > 0.15 and post > pre, (round(100 * pre, 1), round(100 * post, 1)))
w = world(fb)
notes = w.act({"place": [order("SELL", 40, "AMD")]})
check("a flip through zero to a smaller or equal position is checked and rejected", not w.paper.placed and "single-shock limit" in notes[0], notes)
check("shock_cap must be a number in (0, 1)", [trader.shock_cap_of({"shock_cap": v}) for v in (0.15, "0.15", 15, True, 0, None)]
      == [0.15, None, None, None, None, None])
w = world({"AMD": -30}, cap=None)
notes = w.act({"place": [order("SELL", 10, "AMD")]})
check("with the term off nothing changes", w.paper.placed == [("SELL", 10, "AMD")] and not notes, notes)

st = {"value": 30000.0, "cash": 12000.0, "gross": 0.6, "net": 0.6, "beta_net": 0.0, "events": [], "orders": [],
      "positions": [{"symbol": "TLT", "quantity": 200, "avg_cost": 90.0, "close": 90.0, "unrealized": 0, "days": 1}]}
line = trader.shock_rule(0.15, st, BEFORE)
text = trader.render_account(st, {}, "H", None, 2.0, shock=line)
check("the state tells the rule and the worst shock now", "Single-shock limit (15% of the sleeve)" in text and
      "Now the worst is 18.0% of the sleeve, if its largest long falls 30%." in text, line[-80:])
check("the fallback line states the rule and no claim about the book",
      "no positions" not in trader.shock_rule(0.15, dict(st, value=0.0), "") and "Single-shock limit" in trader.shock_rule(0.15, dict(st, value=0.0), ""))
check("without the term the state is unchanged", "Single-shock" not in trader.render_account(st, {}, "H", None, 2.0))
print("ALL OK" if all(ok) else "FAILED")
sys.exit(0 if all(ok) else 1)
