"""Close-only orders (Clayton, Tue 17 Nov 2026: "Close-only orders now, before the 20 Nov plan: a flag the model sets on an
order, so a stop it means as protective can never open a new position."). The real trader.PaperWorld.act and paper.py on
temporary ledger files and a fake client-10 gateway; offline, nothing under runs/ is written.

    python close_only_test.py
"""
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import close_only  # noqa: E402
import paper  # noqa: E402
import tools  # noqa: E402
import trader  # noqa: E402

close_only.V2 = False   # this file pins the rule of 17-19 Nov (the switch's off side); close_only_v2_test.py tests v2.1

ok = []


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail and not good else ""))


# ================================================================ the flag and the forms
co = lambda v: trader.clean_order({"symbol": "ARM", "action": "BUY", "quantity": 1, "type": "STP", "stop": 1, "tif": "GTC",
                                   **({} if v is None else {"close_only": v})})[0]["close_only"]
check("clean_order: close_only only for an explicit true (true or \"true\"); missing, false, 1, \"yes\" are not",
      [co(v) for v in (True, "true", " True ", None, False, 1, "yes")] == [True, True, True, False, False, False, False])
f_b = trader.TOOL_LIVE["parameters"]["properties"]["place"]["items"]["properties"]
f_a = trader.TOOL_LIVE_PROBS["parameters"]["properties"]["place"]["items"]["properties"]
check("text B's submit form has close_only, in the field and in the tool's own description; text A's form does not",
      f_b.get("close_only", {}).get("type") == "boolean" and tools.CLOSE_DESC in trader.TOOL_LIVE["description"]
      and "close_only" not in f_a and "close_only" not in trader.TOOL_LIVE_PROBS["description"])
probe = ("import sys; sys.path.insert(0, sys.argv[1]); import mcp_server, tools, json; "
         "print(json.dumps([tools.CLOSE_DESC in mcp_server.SUBMIT_DESC['B'], 'close_only' in mcp_server.SUBMIT_DESC['A']]))")
d0 = Path(tempfile.mkdtemp())
p = subprocess.run([sys.executable, "-c", probe, str(ROOT)], capture_output=True, text=True, encoding="utf-8", cwd=d0,
                   env=dict(os.environ, TRADER_SUBMIT_FILE=str(d0 / "s.json"), TRADER_LOG_FILE=str(d0 / "l.jsonl")))
check("Fable's submit: text B's description names close_only and states the rule; text A's does not",
      p.returncode == 0 and json.loads(p.stdout.strip().splitlines()[-1]) == [True, False], p.stderr[-300:])

# ================================================================ a temporary ledger and a fake client 10
TMP = Path(tempfile.mkdtemp())
paper.RUNS, paper.FILLS, paper.ORDERS = TMP, TMP / "fills.jsonl", TMP / "orders.jsonl"
paper.SLEEVES, paper.LOOKS, paper.NOTES, paper.SPLITS = TMP / "sleeves.json", TMP / "looks.json", TMP / "notes.json", TMP / "splits.jsonl"
paper.ARCHIVE = TMP / "archive"
paper.SLEEVES.write_text(json.dumps({"cand2": {"start_cash": 90000.0, "start": "2026-11-16T11:00:00+00:00", "model": "x"}}))
N = [0]


def fill(sym, side, qty, px, ref, t="20261117 11:30:00 US/Eastern"):
    N[0] += 1
    return {"exec_id": f"e{N[0]}", "time": t, "symbol": sym, "side": side, "shares": qty, "price": px, "ref": ref,
            "commission": 1.0, "client_id": 10}


class GW:
    def __init__(self):
        self.execs, self.open, self.placed, self.cancelled, self.fail_execs = [], [], [], [], False

    def executions(self, days=7):
        if self.fail_execs:
            raise TimeoutError("executions timed out")
        return list(self.execs)

    def orders(self):
        return [dict(o) for o in self.open]

    def quotes(self, syms):
        return [{"symbol": s, "last": 100.0, "bid": 99.9, "ask": 100.1, "close": 100.0} for s in syms]

    def place(self, symbol, action, quantity, order_type, limit=None, aux=None, tif="DAY", ref=None, **f):
        self.placed.append({"symbol": symbol, "action": action, "quantity": quantity, "order_type": order_type, "ref": ref, **f})
        self.open.append({"ref": ref, "symbol": symbol, "action": action, "quantity": quantity, "order_type": order_type,
                          "aux": aux, "limit": limit, "tif": tif, "status": "PreSubmitted", "remaining": quantity})
        return {"status": "PreSubmitted", "messages": []}

    def cancel(self, ref=None):
        self.cancelled.append(ref)
        if not any(o["ref"] == ref for o in self.open):
            raise RuntimeError(f"Ref {ref!r} is not an open order")
        self.open = [o for o in self.open if o["ref"] != ref]
        return {"status": "PendingCancel"}


def world(gw):
    w = trader.PaperWorld.__new__(trader.PaperWorld)
    w.gw, w.sleeve, w.paper, w.dry_run, w.log = gw, "cand2", paper, False, None
    w.MAX_GROSS, w.SHOCK_CAP, w.TEXT = None, None, "B"
    return w


def fresh(short=150, sym="ARM"):
    for f in (paper.FILLS, paper.ORDERS, paper.NOTES, paper.LOOKS):
        if f.exists():
            f.unlink()
    gw = GW()
    if short:
        gw.execs.append(fill(sym, "SLD", short, 180.0, "tt-cand2-0001", "20261117 10:05:00 US/Eastern"))
    paper.sync_fills(gw)
    return gw


STP = lambda q, act="BUY", sym="ARM", **k: {"symbol": sym, "action": act, "quantity": q, "type": "STP", "stop": 190, "tif": "GTC", **k}

# a. short 150: a close_only BUY STP 150 is sent, recorded as close_only, and the flag never reaches the broker
gw = fresh()
notes = world(gw).act({"place": [STP(150, close_only=True)], "cancel": []})
rows = paper.read(paper.ORDERS)
check("short 150 ARM: a close_only BUY STP 150 is sent; its row says close_only; the broker got no close_only field",
      len(gw.placed) == 1 and rows[-1].get("close_only") is True and "close_only" not in gw.placed[0] and not
      [n for n in notes if "not sent" in n], notes)

# b. cand2 at 12:00: the old stop fills during the look, so the position is 0 when the new close_only stop is sent
gw = fresh()
gw.execs.append(fill("ARM", "BOT", 150, 190.0, "tt-cand2-0002"))     # the old stop, filled after the state was built
notes = world(gw).act({"place": [STP(150, close_only=True)], "cancel": ["tt-cand2-0002"]})
check("the old stop filled during the look (cand2 at 12:00): the new close_only BUY is not sent, and the note says the "
      "position is +0 counting fills so far", not gw.placed and any("(close_only) not sent: your position in ARM is +0 counting "
                                                                     "fills so far, so it would open a position" in n for n in notes), notes)
gw2 = fresh()
gw2.execs.append(fill("ARM", "BOT", 150, 190.0, "tt-cand2-0002"))
world(gw2).act({"place": [STP(150)], "cancel": []})
check("the same order without the flag is sent as before (and would open a long when it fills)", len(gw2.placed) == 1)

# c. two close_only stops for one short: the second is not sent; d. with the first cancelled in the same submit, it is
gw = fresh()
world(gw).act({"place": [STP(150, close_only=True)], "cancel": []})
first = gw.placed[0]["ref"]
notes = world(gw).act({"place": [STP(150, close_only=True)], "cancel": []})
check("a second close_only BUY 150 while the first still works: not sent (already covers 150)", len(gw.placed) == 1
      and any("your close_only orders there already cover 150, so it would close more than you hold" in n for n in notes), notes)
notes = world(gw).act({"place": [STP(150, close_only=True)], "cancel": [first]})
check("with the first cancelled in the same submit, the new one is sent", len(gw.placed) == 2 and first in gw.cancelled, notes)
gw = fresh()
notes = world(gw).act({"place": [STP(100, close_only=True), STP(60, close_only=True)], "cancel": []})
check("in one submit: 100 then 60 against a short of 150: the 100 goes, the 60 does not (it would make 160)",
      [p["quantity"] for p in gw.placed] == [100] and any("already cover 100" in n for n in notes), notes)

# e. wrong side, flat, and a fill read failure
gw = fresh()
notes = world(gw).act({"place": [STP(50, act="SELL", close_only=True)], "cancel": []})
check("short 150: a close_only SELL would add to it: not sent", not gw.placed and any("so it would add to it" in n for n in notes), notes)
gw = fresh(short=0)
notes = world(gw).act({"place": [STP(10, close_only=True)], "cancel": []})
check("no position: a close_only order would open one: not sent", not gw.placed and any("would open a position" in n for n in notes))
gw = fresh()
gw.fail_execs = True
notes = world(gw).act({"place": [STP(150, close_only=True), STP(5, sym="MU")], "cancel": []})
check("fills that cannot be read: the close_only order is not sent (its condition can't be held); the plain order is",
      [p["symbol"] for p in gw.placed] == ["MU"] and any("your fills could not be read" in n for n in notes), notes)

# ================================================================ the check at each look (paper.close_only_excess)
gw = fresh()
world(gw).act({"place": [STP(100, close_only=True)], "cancel": []})
old = gw.placed[-1]["ref"]
world(gw).act({"place": [STP(50, close_only=True)], "cancel": []})
new = gw.placed[-1]["ref"]
gw.execs.append(fill("ARM", "BOT", 100, 175.0, "tt-cand2-0009"))      # the position falls to -50 by another order
paper.sync_fills(gw)
logs = []
paper.close_only_excess(gw, "cand2", log=logs.append)
note = [x for x in json.loads(paper.NOTES.read_text())["cand2"] if x.startswith("framework note (close_only)")]
check("short 150 falls to 50 with close_only stops 100 (older) and 50 (newer): the oldest that still fit are kept, so the "
      "100 (more than the 50 held) is cancelled and the 50 stays; one note says so",
      gw.cancelled == [old] and [o["ref"] for o in gw.open] == [new] and len(note) == 1
      and "the oldest that still fit were kept" in note[0], (gw.cancelled, note))
gw = fresh()
world(gw).act({"place": [STP(100, close_only=True)], "cancel": []})
world(gw).act({"place": [STP(50, close_only=True)], "cancel": []})
gw.execs.append(fill("ARM", "BOT", 50, 175.0, "tt-cand2-0010"))       # short 150 -> 100: the 100 + 50 no longer fit
paper.sync_fills(gw)
paper.close_only_excess(gw, "cand2")
check("short 150 falls to 100 with close_only 100 (older) + 50 (newer): only the newer 50 is cancelled",
      len(gw.cancelled) == 1 and [o["quantity"] for o in gw.open] == [100], (gw.cancelled, gw.open))
gw = fresh()
world(gw).act({"place": [STP(150, close_only=True)], "cancel": []})
gw.execs.append(fill("ARM", "BOT", 300, 175.0, "tt-cand2-0011"))      # now long 150: a close_only BUY would add
paper.sync_fills(gw)
paper.close_only_excess(gw, "cand2")
check("the position flips to long by another order: the close_only BUY (now adding) is cancelled", len(gw.cancelled) == 1 and not gw.open)
gw = fresh()
world(gw).act({"place": [STP(150, close_only=True)], "cancel": []})
n0 = len(gw.cancelled)
paper.close_only_excess(gw, "cand2")
plain = fresh()
world(plain).act({"place": [STP(150)], "cancel": []})
plain.execs.append(fill("ARM", "BOT", 150, 175.0, "tt-cand2-0012"))
paper.sync_fills(plain)
paper.close_only_excess(plain, "cand2")
check("a covered close_only order is left alone, and an order without the flag is never touched by the check",
      len(gw.cancelled) == n0 and not plain.cancelled)

# ================================================================ the review of 17 Nov
# an entry and its close_only stop in one submit: a market entry that fills within seconds lets the stop go
class GWFill(GW):
    def place(self, symbol, action, quantity, order_type, limit=None, aux=None, tif="DAY", ref=None, **f):
        out = super().place(symbol, action, quantity, order_type, limit=limit, aux=aux, tif=tif, ref=ref, **f)
        if order_type == "MKT":                                       # fills at once; reported by executions()
            self.execs.append(fill(symbol, "SLD" if action == "SELL" else "BOT", quantity, 100.0, ref))
            self.open = [o for o in self.open if o["ref"] != ref]
        return out


gw = GWFill()
for f in (paper.FILLS, paper.ORDERS, paper.NOTES, paper.LOOKS):
    if f.exists():
        f.unlink()
notes = world(gw).act({"place": [{"symbol": "MU", "action": "SELL", "quantity": 100, "type": "MKT", "tif": "DAY"},
                                 STP(100, sym="MU", close_only=True)], "cancel": []})
check("a market entry and its close_only stop in one submit: the entry fills, fills are read again, the stop is sent",
      [p["order_type"] for p in gw.placed] == ["MKT", "STP"] and not [n for n in notes if "not sent" in n], notes)
gw = GWFill()
for f in (paper.FILLS, paper.ORDERS, paper.NOTES, paper.LOOKS):
    if f.exists():
        f.unlink()
notes = world(gw).act({"place": [{"symbol": "MU", "action": "SELL", "quantity": 100, "type": "LMT", "limit": 200, "tif": "DAY"},
                                 STP(100, sym="MU", close_only=True)], "cancel": []})
check("a limit entry that has not filled: its close_only stop is not sent (it would open a position)",
      [p["order_type"] for p in gw.placed] == ["LMT"] and any("would open a position" in n for n in notes), notes)


class GWSlowCancel(GW):
    def cancel(self, ref=None):                                       # IBKR does not confirm: the order stays open
        self.cancelled.append(ref)
        return {"status": "Submitted", "messages": ["Cancel not confirmed within 30s; check orders()"]}


gw = GWSlowCancel()
for f in (paper.FILLS, paper.ORDERS, paper.NOTES, paper.LOOKS):
    if f.exists():
        f.unlink()
gw.execs.append(fill("ARM", "SLD", 150, 180.0, "tt-cand2-0001", "20261117 10:05:00 US/Eastern"))
paper.sync_fills(gw)
world(gw).act({"place": [STP(150, close_only=True)], "cancel": []})
a = gw.placed[0]["ref"]
notes = world(gw).act({"place": [STP(150, close_only=True)], "cancel": [a]})
check("an unconfirmed cancel: the old close_only stop still counts as cover, so the replacement is not sent (both could fill)",
      len(gw.placed) == 1 and any("already cover 150" in n for n in notes), notes)


class GWRefuse(GW):
    def place(self, symbol, action, quantity, order_type, limit=None, aux=None, tif="DAY", ref=None, **f):
        self.tries = getattr(self, "tries", 0) + 1
        if self.tries == 1:                                           # IBKR refuses the first (margin)
            self.placed.append({"symbol": symbol, "action": action, "quantity": quantity, "order_type": order_type, "ref": ref})
            return {"status": "Inactive", "messages": ["201 margin"]}
        return super().place(symbol, action, quantity, order_type, limit=limit, aux=aux, tif=tif, ref=ref, **f)


gw = GWRefuse()
for f in (paper.FILLS, paper.ORDERS, paper.NOTES, paper.LOOKS):
    if f.exists():
        f.unlink()
gw.execs.append(fill("ARM", "SLD", 150, 180.0, "tt-cand2-0001", "20261117 10:05:00 US/Eastern"))
paper.sync_fills(gw)
notes = world(gw).act({"place": [STP(150, close_only=True), STP(150, close_only=True)], "cancel": []})
check("a close_only order IBKR refuses is no cover: the next close_only order in the submit is sent",
      len(gw.placed) == 2 and not [n for n in notes if "(close_only) not sent" in n], notes)

# the recheck of 17 Nov: (1) a close_only market order that fills during the wait is not counted twice
trader._time.sleep = lambda s: None                                   # the 2 s waits, not needed with these fakes
gw = GWFill()
for f in (paper.FILLS, paper.ORDERS, paper.NOTES, paper.LOOKS):
    if f.exists():
        f.unlink()
gw.execs.append(fill("ARM", "SLD", 150, 180.0, "tt-cand2-0001", "20261117 10:05:00 US/Eastern"))
paper.sync_fills(gw)
notes = world(gw).act({"place": [{"symbol": "ARM", "action": "BUY", "quantity": 50, "type": "MKT", "tif": "DAY",
                                  "close_only": True}, STP(100, close_only=True)], "cancel": []})
check("short 150: a close_only BUY MKT 50 (fills at once) then a close_only BUY STP 100: both are sent (the filled 50 is "
      "not counted as cover too)", [p["order_type"] for p in gw.placed] == ["MKT", "STP"]
      and not [n for n in notes if "not sent" in n], notes)
# (2) a cancel that stays pending is named once, however often the check runs
gw = GWSlowCancel()
for f in (paper.FILLS, paper.ORDERS, paper.NOTES, paper.LOOKS):
    if f.exists():
        f.unlink()
gw.execs.append(fill("ARM", "SLD", 150, 180.0, "tt-cand2-0001", "20261117 10:05:00 US/Eastern"))
paper.sync_fills(gw)
world(gw).act({"place": [STP(150, close_only=True)], "cancel": []})
gw.execs.append(fill("ARM", "BOT", 150, 175.0, "tt-cand2-0020"))       # a take-profit closes the short
paper.sync_fills(gw)
for _ in range(3):
    paper.close_only_excess(gw, "cand2")
co_notes = [x for x in json.loads(paper.NOTES.read_text()).get("cand2", []) if x.startswith("framework note (close_only)")]
check("a close_only cancel IBKR leaves unconfirmed: one cancel sent, one note, over three runs of the check",
      len(gw.cancelled) == 1 and len(co_notes) == 1 and "an order is working until the broker confirms it" in co_notes[0],
      (gw.cancelled, co_notes))
# (3) the submit's own plain order closes the position: its close_only stop is cancelled at the end of act, not 30 min later
gw = GWFill()
for f in (paper.FILLS, paper.ORDERS, paper.NOTES, paper.LOOKS):
    if f.exists():
        f.unlink()
gw.execs.append(fill("ARM", "SLD", 150, 180.0, "tt-cand2-0001", "20261117 10:05:00 US/Eastern"))
paper.sync_fills(gw)
world(gw).act({"place": [STP(150, close_only=True)], "cancel": []})
stop = gw.placed[-1]["ref"]
world(gw).act({"place": [{"symbol": "ARM", "action": "BUY", "quantity": 150, "type": "MKT", "tif": "DAY"}], "cancel": []})
check("a plain market order in a later submit closes the short: the close_only stop is cancelled at the end of that act",
      stop in gw.cancelled and not any(o["ref"] == stop for o in gw.open), (gw.cancelled, gw.open))
gw = GWFill()
for f in (paper.FILLS, paper.ORDERS, paper.NOTES, paper.LOOKS):
    if f.exists():
        f.unlink()
world(gw).act({"place": [{"symbol": "MU", "action": "BUY", "quantity": 10, "type": "MKT", "tif": "DAY"}], "cancel": []})
check("a sleeve that never placed a close_only order: act places as before and makes no extra orders() read",
      len(gw.placed) == 1 and not gw.cancelled)

# the loop's guard between looks
import live  # noqa: E402
from datetime import datetime  # noqa: E402
gw = fresh()
world(gw).act({"place": [STP(150, close_only=True)], "cancel": []})
gw.execs.append(fill("ARM", "BOT", 150, 175.0, "tt-cand2-0013"))       # a take-profit closes the short between looks
import contextlib  # noqa: E402
saved = (live.gateway, live.log, paper.sleeves)
logs = []
live.gateway = contextlib.contextmanager(lambda: (yield gw))
live.log = logs.append
try:
    live._CO_GUARD.clear()
    live.close_only_guard(datetime(2026, 11, 17, 10, 10, tzinfo=live.ET))
    first = list(gw.cancelled)
    live.close_only_guard(datetime(2026, 11, 17, 10, 10, 30, tzinfo=live.ET))     # 30 s later: not again
    live._CO_GUARD.clear()
    live.close_only_guard(datetime(2026, 11, 17, 8, 10, tzinfo=live.ET))           # before the open: nothing
finally:
    live.gateway, live.log = saved[0], saved[1]
check("between looks, the loop's guard (once a minute, in the session) syncs fills and cancels the close_only stop the "
      "position no longer covers", len(first) == 1 and gw.cancelled == first and not gw.open, (first, gw.cancelled, logs))

# ================================================================ the state shows it
gw = fresh()
world(gw).act({"place": [STP(150, close_only=True)], "cancel": []})
st = {"value": 90000.0, "cash": 90000.0, "positions": [], "events": [], "gross": 0, "net": 0, "beta_net": 0,
      "orders": [{"ref": gw.placed[0]["ref"], "symbol": "ARM", "action": "BUY", "quantity": 150, "type": "STP", "stop": 190.0,
                  "limit": None, "trail_percent": None, "tif": "GTC", "cancel_pending": False, "close_only": True}]}
txt = trader.render_account(st, {}, "Header", max_gross=None)
check("the state lists a working close_only order with 'close_only' after its status", " GTC working close_only" in txt, txt[-300:])

# e. a stop typed "STOP" (cand1 at 11:30 on 18 Nov: three protective stops refused, 321 "Please enter a stop price"): its
# stop price goes to IBKR as for STP; the re-ask reads it as STP; other types are read as before
ty = lambda t: trader.clean_order({"symbol": "ARM", "action": "BUY", "quantity": 1, "type": t, "stop": 1, "limit": 2,
                                   "tif": "GTC"})[0]["type"]
check("clean_order: STOP is STP and STOP LIMIT / STOP LMT / stop_limit are STP LMT; listed and unknown types as before",
      [ty(t) for t in ("STOP", "stop", "STOP LIMIT", "stop_limit", "STOP LMT", "STP", "STP LMT", "LMT", "MMT", "TRAIL",
                       " STOP", "STOP ", "STOP  LIMIT", " stp lmt ", "")]
      == ["STP", "STP", "STP LMT", "STP LMT", "STP LMT", "STP", "STP LMT", "LMT", "MMT", "TRAIL",
          "STP", "STP", "STP LMT", "STP LMT", ""]
      and all(tools.missing_prices([{"symbol": "A", "action": "BUY", "quantity": 1, "type": t}])[0][1].endswith(
          trader.clean_order({"symbol": "A", "action": "BUY", "quantity": 1, "type": t})[0]["type"])
              for t in (" STOP", "STOP  LIMIT", "stop_limit", "STP"))
      and trader.clean_order({"symbol": "A", "action": "BUY", "quantity": 1, "type": None})[0]["type"] == "NONE")
gw = fresh()
notes = world(gw).act({"place": [dict(STP(150, close_only=True), type="STOP")], "cancel": []})
check("short 150 ARM: a close_only BUY STOP 150 with stop 190 goes to IBKR as STP with its stop price",
      len(gw.open) == 1 and gw.open[0]["order_type"] == "STP" and gw.open[0]["aux"] == 190
      and not [n for n in notes if "not sent" in n or "refused" in n], (gw.open, notes))
check("the re-ask: a STOP without a stop is missing its stop; with one, nothing is missing",
      tools.missing_prices([{"symbol": "ARM", "action": "BUY", "quantity": 2, "type": "STOP", "tif": "GTC"}])
      == [(1, "BUY 2 ARM STP", ["stop"], [])]    # named as it is read
      and tools.missing_prices([{"symbol": "ARM", "action": "BUY", "quantity": 2, "type": "STOP", "stop": 5, "tif": "GTC"}]) == [])

print("ALL OK" if all(ok) else f"FAILED ({ok.count(False)})")
sys.exit(0 if all(ok) else 1)
