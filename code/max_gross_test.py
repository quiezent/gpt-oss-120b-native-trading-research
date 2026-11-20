"""The gross-exposure sleeve term as a per-sleeve switch (sleeves.json "max_gross"; Clayton, 14 Nov 2026: "I prefer less
intervention from the software framework or a hard limit unless IBKR restricts it"). Offline: fakes for the gateway and
paper, and the committed code before the change for the texts.
- Without "max_gross" nothing changes: the system prompt and the state are byte-identical to d4fcbc5, and act() rejects
  an add past 2x as before.
- "max_gross": null: the 2x sentence gives way to the no-limit sentence, the state has no buying-power line, act()
  sends an add of any size to the broker, and a refusal by IBKR reaches the model's notes.
- A malformed value keeps 2x.

    python max_gross_test.py
"""
import importlib.util
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import trader

ok = []
ROOT = Path(__file__).resolve().parent


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail else ""))


old_src = subprocess.run(["git", "show", "d4fcbc5:trader.py"], capture_output=True, text=True, encoding="utf-8", cwd=ROOT).stdout
assert "def live_system" in old_src
tmp = Path(tempfile.mkdtemp()) / "trader_before.py"
tmp.write_text(old_src, encoding="utf-8")
spec = importlib.util.spec_from_file_location("trader_before", tmp)
before = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(ROOT))
spec.loader.exec_module(before)

day = date(2026, 11, 16)
cases = [(size, ml, news, kelly) for size in (30000, 90000) for ml in (False, True) for news in (False, True) for kelly in (False, True)]
check("default: the system prompt is byte-identical to d4fcbc5 (16 cases)",
      all(trader.live_system(s, day, ml=m, news=n, kelly=k) == before.live_system(s, day, ml=m, news=n, kelly=k) for s, m, n, k in cases))
check("max_gross 2.0 is the default text", trader.live_system(90000, day, ml=True, max_gross=2.0) == trader.live_system(90000, day, ml=True))

base = trader.live_system(90000, day, ml=True)
free = trader.live_system(90000, day, ml=True, max_gross=None)
check("null: only the 2x sentence changes", free == base.replace(trader.GROSS_RULE, trader.NO_GROSS_RULE)
      and base.count(trader.GROSS_RULE) == 1 and "2x" not in free, free[free.index("Short selling"):free.index("Short selling") + 260])
check("null: plain ASCII (the Claude CLI gets the system prompt as an argument)", all(ord(c) < 128 for c in free))
kfree = trader.live_system(90000, day, ml=True, kelly=True, max_gross=None)
check("null with the Kelly paragraph: its 2x bullet goes too", "2x gross" not in kfree and "Sizing by the Kelly rule" in kfree
      and "Use a quarter to a half" in kfree and "Positions that move together" in kfree)
three = trader.live_system(90000, day, max_gross=3)
check("a number changes the figure", "Gross exposure may not exceed 3x the sleeve." in three and "2x" not in three)

check("max_gross_of: absent 2.0, null None, a number itself, malformed 2.0",
      [trader.max_gross_of(m) for m in ({}, {"max_gross": None}, {"max_gross": 3}, {"max_gross": 1.5}, {"max_gross": "none"},
                                        {"max_gross": 0}, {"max_gross": True}, {"max_gross": -1})]
      == [2.0, None, 3.0, 1.5, 2.0, 2.0, 2.0, 2.0])
check("max_gross_bad flags only the malformed", [trader.max_gross_bad(m) for m in ({}, {"max_gross": None}, {"max_gross": 3},
                                                                                   {"max_gross": "none"}, {"max_gross": 0}, {"max_gross": True})]
      == [False, False, False, True, True, True])

PX = {"AMD": 400.0, "TLT": 90.0}


class FakePaper:
    def __init__(self, pos, cash, refuse=None):
        self.pos, self.cash, self.placed, self.kept, self.refuse = pos, cash, [], [], refuse

    def book(self, sleeve, marks=None):
        return {"positions": {s: {"quantity": q, "avg_cost": 100.0} for s, q in self.pos.items()}, "cash": self.cash}

    def mark(self, q):
        return q["last"]

    def place(self, gw, sleeve, symbol, action, quantity, *a, **k):
        self.placed.append((action, quantity, symbol))
        if self.refuse:
            return {"result": {"status": "Inactive", "messages": [self.refuse]}}
        return {"result": {"status": "Submitted", "messages": []}}

    def add_notes(self, sleeve, notes):
        self.kept += notes

    def cancel(self, gw, sleeve, ref):
        return {}

    def et_today(self):
        return "20261106"


class FakeGW:
    def quotes(self, syms):
        return [{"symbol": s, "last": PX[s], "bid": PX[s], "ask": PX[s]} for s in syms]


def world(pos, max_gross, nav=30000.0, refuse=None):
    w = trader.PaperWorld(FakeGW(), "test-sleeve", None)
    w.paper = FakePaper(dict(pos), nav - sum(q * PX[s] for s, q in pos.items()), refuse)
    w.SHOCK_CAP, w.MAX_GROSS = None, max_gross
    w.TEXT = "A"            # no order echo (16 Nov 2026, text B's), so the notes are the refusals alone; text_ab_test.py tests the echo
    return w


order = lambda act, q, s: {"symbol": s, "action": act, "quantity": q, "type": "MKT", "tif": "DAY"}
check("a new PaperWorld of a sleeve without the field has the 2x limit", trader.PaperWorld(FakeGW(), "test-sleeve", None).MAX_GROSS == 2.0)
w = world({"TLT": 600}, 2.0)                                   # $54,000 of $30,000: 1.8x
notes = w.act({"place": [order("BUY", 100, "TLT")]})           # to $63,000: 2.1x
check("2x: an add past 2x is rejected, as before", not w.paper.placed and "insufficient margin" in notes[0] and "2.10x" in notes[0], notes)
w = world({"TLT": 600}, None)
notes = w.act({"place": [order("BUY", 100, "TLT"), order("BUY", 3000, "TLT"), order("SELL", 500, "AMD")]})
check("null: adds of any size go to the broker, with no note", w.paper.placed == [("BUY", 100, "TLT"), ("BUY", 3000, "TLT"), ("SELL", 500, "AMD")]
      and not notes, notes)
w = world({"TLT": 600}, None, refuse="Order rejected - reason: insufficient margin test")
notes = w.act({"place": [order("BUY", 3000, "TLT")]})
check("null: IBKR's refusal reaches the model's notes", len(notes) == 1 and "BUY 3000 TLT MKT: Inactive Order rejected - reason" in notes[0]
      and w.paper.kept == notes, notes)
w = world({"TLT": 600}, 3.0)
notes = w.act({"place": [order("BUY", 300, "TLT"), order("BUY", 200, "TLT")]})    # 2.7x, then 3.3x
check("a number is the limit", w.paper.placed == [("BUY", 300, "TLT")] and "above the 3x limit" in notes[0], notes)

st = {"value": 30000.0, "cash": -24000.0, "gross": 1.8, "net": 1.8, "beta_net": 0.0, "events": [], "orders": [],
      "positions": [{"symbol": "TLT", "quantity": 600, "avg_cost": 90.0, "close": 90.0, "unrealized": 0, "days": 1}]}
check("default: the state is byte-identical to d4fcbc5", trader.render_account(st, {}, "H", None, 2.0) == before.render_account(st, {}, "H", None, 2.0))
text = trader.render_account(st, {}, "H", None, None)
check("null: no buying-power line and no 2x in the state", "Buying power" not in text and "2x" not in text and "margin" not in text
      and "Gross exposure 1.80x" in text, text[:200])
check("null: the state is the default one less its buying-power line",
      text == "\n".join(l for l in trader.render_account(st, {}, "H", None, 2.0).split("\n") if not l.startswith("Buying power")))


class W:                                                       # account_text passes the world's own term to the state
    MAX_GROSS = None

    def state(self, d, feats):
        return st


class D:
    def iso(self, d):
        return "2026-11-16"


acct, _ = trader.account_text(W(), D(), 0, {})
check("account_text of a no-limit world has no buying-power line", "Buying power" not in acct and "2x" not in acct)
print("ALL OK" if all(ok) else "FAILED")
sys.exit(0 if all(ok) else 1)
