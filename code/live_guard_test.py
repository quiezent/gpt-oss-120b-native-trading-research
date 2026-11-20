"""An IBKR refusal (321, status None) reaches the model's notes, a failed look record costs that record only, and a look
with no decision leaves what it was told for the next look. Offline: a fake gateway and paper, a temp folder for the look
files, memo and notes, and the log kept in a list.

    python live_guard_test.py
"""
import json
import tempfile
from datetime import datetime
from pathlib import Path

import live
import paper
import trader


class FakePaper:
    def __init__(self, status, messages=None, error=None):
        self.reply = {"result": {"status": status, "messages": messages}, "error": error}
        self.placed, self.kept = [], []

    def book(self, sleeve, marks=None):
        return {"positions": {}, "cash": 30000.0}

    def mark(self, q):
        return q["last"]

    def place(self, gw, sleeve, symbol, *a, **k):
        self.placed.append(symbol)
        return self.reply

    def add_notes(self, sleeve, notes):
        self.kept += notes


class FakeGW:
    def quotes(self, syms):
        return [{"symbol": s, "last": 100.0, "bid": 99.9, "ask": 100.1} for s in syms]


def check(name, got, want=(), absent=()):
    ok = all(w in got for w in want) and not any(a in got for a in absent)
    print(("ok   " if ok else "FAIL ") + name + ": " + got[:300])
    return ok


def act_notes(status, messages=None, error=None):
    world = object.__new__(trader.PaperWorld)
    world.paper, world.gw, world.sleeve, world.data, world.dry_run = FakePaper(status, messages, error), FakeGW(), "s", None, False
    world.TEXT = "A"        # the refusal notes alone: text B's order echo (16 Nov 2026) is tested in text_ab_test.py
    notes = world.act({"place": [{"symbol": "AAPL", "action": "BUY", "quantity": 5, "type": "LMT", "limit": 99.0, "tif": "DAY"}],
                       "cancel": []})
    assert world.paper.placed == ["AAPL"] and world.paper.kept == notes     # the order is sent as asked; the note goes to the next look
    return str(notes)


REFUSAL = "321: Error validating request.-'PlaceOrder(3)' : cause - Invalid order type was entered"
ok = [check("321 refusal is told", act_notes(None, [REFUSAL]), ["BUY 5 AAPL LMT: refused 321: Error validating"], ["None"]),
      check("working order says nothing", act_notes("PreSubmitted"), ["[]"]),
      check("no acknowledgement says nothing, as before", act_notes("Unknown", ["No acknowledgement within 30s"]), ["[]"]),
      check("Inactive still told", act_notes("Inactive"), ["BUY 5 AAPL LMT: Inactive"]),
      check("gateway error still told", act_notes(None, error="GatewayError: x"), ["BUY 5 AAPL LMT:", "GatewayError: x"])]

logged = []
live.log = logged.append
with tempfile.TemporaryDirectory() as d:
    live.LIVE = Path(d)
    now = datetime(2026, 11, 3, 10, 0, tzinfo=live.ET)
    st = {"value": 1.0, "events": []}
    call = {"place": [], "cancel": [], "notes": ""}
    meta = {"model": "base"}
    live.finish(now, "bad-symbols", meta, "text", st, call, [{"tool": "quote", "args": {"symbols": 5}}], [], [], True, secs=1.0)
    ok.append(check("{'symbols': 5} is recorded", json.dumps(json.loads((Path(d) / "2026-11-03" / "bad-symbols-1000.json").read_text())["research"]),
                    ['"research_calls": 1']))
    live.finish(now, "bad-record", meta, "text", st, call, [{"tool": "quote", "not_json": object()}], [], [], True, secs=1.0)
    ok.append(check("a failed record is logged and returns", logged[-1], ["bad-record: look record failed", "TypeError"]))

    live.MEMO, paper.NOTES = Path(d) / "memo.json", Path(d) / "notes.json"
    told = {"value": 1.0, "events": ["filled SELL 5 NVDA @ 222.52 [tt-x]", "BUY 300 T MMT: refused 321: bad type"]}
    live.finish(now, "no-decision", meta, "text", told, None, [], [], [], False, secs=1.0)
    ok.append(check("no decision: the next look hears it again", str(paper.take_notes("no-decision")), ["filled SELL 5 NVDA", "refused 321"]))
    live.finish(now, "decided", meta, "text", told, dict(call, notes="my plan"), [], [], [], False, secs=1.0)
    ok.append(check("a decision: nothing repeats", str(paper.take_notes("decided")), ["[]"]))
    live.finish(now, "dry", meta, "text", told, None, [], [], [], True, secs=1.0)
    ok.append(check("a dry run queues nothing", str(paper.take_notes("dry")), ["[]"]))
    live.finish(now, "bad-kept", meta, "text", st, dict(call, notes="still my plan"), [{"tool": "quote", "not_json": object()}], [], [], False, secs=1.0)
    ok.append(check("a failed record keeps the model's notes", str(live.load_memo("bad-kept")), ["still my plan"]))
    ok.append(check("and says so", logged[-1], ["bad-kept: look record failed"]))

# research_summary (16 Nov 2026): a call the loop did not run for lack of room and a dropped turn are no research and no check
tr = [{"step": 0, "tool": "tape", "args": {}, "result": "news"},
      {"step": 1, "tool": "quote", "args": {"symbols": ["NVDA", "AMD"]}, "result": "NVDA 1\nAMD 2\n(research call 2 of 12)"},
      {"step": 2, "tool": "check_orders", "args": {"place": []}, "result": "ok\n(order check 1 of 3)", "ibkr": {}},
      {"step": 3, "tool": "stats", "args": {"symbols": ["TSLA"]}, "result": "context nearly full (3000 tokens left); no research call is run"},
      {"step": 4, "tool": "check_orders", "args": {"place": []}, "result": "context nearly full (2900 tokens left); no check is run"},
      {"step": 5, "room": 2600, "left": 900, "dropped": "quote", "args": {"symbols": ["MSFT"]}, "tokens": 1700, "said": "x",
       "note": "turn left too little room for the decision: dropped, submit opened"},
      {"step": 6, "tool": "quote", "args": {"symbols": ["XLE"]}, "result": "research limit reached (12 calls): call submit now"},
      {"step": 7, "tool": "submit", "tokens": 50, "said": "y"}]
s = live.research_summary(tr, {"place": [{"symbol": "NVDA"}, {"symbol": "TSLA"}, {"symbol": "MSFT"}]}, {"NVDA": 1})
ok.append(check("research_summary: calls not run and a dropped turn are not research; a limit-told call counts as before",
                json.dumps(s, sort_keys=True),
                ['"research_calls": 3', '"order_checks": 1', '"calls_not_run": 2', '"turns_dropped": 1',
                 '"researched": ["AMD", "NVDA", "XLE"]', '"traded_researched": ["NVDA"]', '"quote": 2', '"tape": 1']))
ok.append(check("research_summary: nothing of the not-run or dropped calls", str("TSLA" in s["researched"] or "MSFT" in s["researched"]
                                                                               or "stats" in s["tools"]), ["False"]))
print("ALL OK" if all(ok) else "FAILED")
