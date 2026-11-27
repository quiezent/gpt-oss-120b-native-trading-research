"""close_only v2.1 (runs/analysis/close_only_v2/DESIGN.md; Clayton, 19 Nov 2026: "Yes if its a fix for future sleeves too").
The real trader.PaperWorld.act, paper.close_only_check and close_only.py on temporary ledger files and a fake client-10
gateway with a modify; offline, nothing under runs/ is written. The cases are the five critiques' replays of 17-19 Nov
(scenarios a-h) and their failure sequences.

    python close_only_v2_test.py
"""
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import close_only as co  # noqa: E402
import paper  # noqa: E402
import trader  # noqa: E402

co.V2 = True                                # read at call time by paper and trader (the words: a subprocess below)
trader._time.sleep = lambda s: None         # act's 2-second waits
ok = []


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail and not good else ""))


# ================================================================ the rule's pure parts
L = lambda ref, rem, g, pending=False, key=0: {"ref": ref, "rem": float(rem), "groups": co.groups(g), "pending": pending, "key": key}
check("groups: stops, limits, auctions and anything outside the session with both, a MKT in the session alone",
      [sorted(co.groups(t, f)) for t, f in (("STP", "GTC"), ("TRAIL", "DAY"), ("STP LMT", "DAY"), ("LMT", "DAY"), ("MOO", "DAY"),
                                            ("MKT", "OPG"), ("MOC", "DAY"), ("LOC", "DAY"), ("MKT", "DAY"))]
      == [["stop"], ["stop"], ["stop"], ["limit"], ["limit", "stop"], ["limit", "stop"], ["limit", "stop"], ["limit", "stop"], []]
      and sorted(co.groups("LMT", "DAY", in_session=False)) == ["limit", "stop"])
legs = [L("s", 200, "STP")]
check("fits: INTC -200 with a 200 stop: a MKT 100 fits alone; a second 100 stop does not; a 100 LMT does",
      co.fits(100, -200, "BUY", legs, co.groups("MKT"))[0] and not co.fits(100, -200, "BUY", legs, co.groups("STP"))[0]
      and co.fits(100, -200, "BUY", legs, co.groups("LMT"))[0])
check("fits: before the open a 100 MOO beside the 200 stop does not fit (both groups)",
      co.fits(100, -200, "BUY", legs, co.groups("MOO"))[1:] == ("more", 200.0, "stop"))
check("fits: a MKT counts a leg whose cancel is pending", not co.fits(100, -200, "BUY", [L("s", 150, "STP", pending=True)],
                                                                         frozenset())[0])
check("fits: the wrong side or flat is 'side'", co.fits(5, 10, "BUY", [], frozenset())[1] == "side"
      and co.fits(5, 0, "SELL", [], frozenset())[1] == "side")
lg = [L("a", 200, "STP", key=1), L("b", 100, "STP", key=2), L("t", 150, "LMT", key=3), L("m", 300, "MKT", key=4)]
check("shrink: position 120 with a working MKT 300 (in every group): the grouped legs go first (b, a, t to 0), the MKT "
      "last, to 120", dict(co.shrink(-120, lg)) == {"b": 0.0, "a": 0.0, "t": 0.0, "m": 120.0}, co.shrink(-120, [
          L("a", 200, "STP", key=1), L("b", 100, "STP", key=2), L("t", 150, "LMT", key=3), L("m", 300, "MKT", key=4)]))
lg = [L("a", 200, "STP", key=1), L("b", 100, "STP", key=2), L("t", 150, "LMT", key=3)]
check("shrink: position 120, no MKT: the stops (300) lose 180, newest first (b to 0, a to 120); the limit 150 to 120",
      dict(co.shrink(-120, lg)) == {"b": 0.0, "a": 120.0, "t": 120.0})
check("cover: a working MKT leg (no group) counts beside a new stop, a new limit and a new MKT (the review's blocker)",
      co.cover([L("m", 100, "MKT")], co.groups("STP"))[0] == 100 and co.cover([L("m", 100, "MKT")], co.groups("LMT"))[0] == 100
      and co.cover([L("m", 100, "MKT")], frozenset())[0] == 100 and co.cover([L("s", 100, "STP")], frozenset())[0] == 0)
lg = [L("s", 200, "STP", key=1), L("m", 100, "MKT", key=2)]
check("shrink: INTC -200 with the 200 stop and the MKT 100 not yet filled: the stop is cut to 100, the MKT kept",
      dict(co.shrink(-200, lg)) == {"s": 100.0})
st_ = {"modify": {"r": [{"quantity": 100, "result": {"modified": None}}, {"quantity": 100, "result": {"modified": True}}]}}
check("_tried_modify reads only the last modify row: confirmed last means not 'tried'; an unconfirmed last for the same "
      "total means 'tried'; a different total is not", not paper._tried_modify(st_, "r", 100)
      and paper._tried_modify({"modify": {"r": [{"quantity": 100, "result": {"modified": None}}]}}, "r", 100)
      and not paper._tried_modify({"modify": {"r": [{"quantity": 60, "result": {"modified": None}}]}}, "r", 100))
import re as _re  # noqa: E402
live_half = _re.search(r"^HALF_DAYS = (\{[^}]*\})", (ROOT / "live.py").read_text(encoding="utf-8"), _re.M).group(1)
check("half days: paper.HALF_DAYS is live.HALF_DAYS; 27 Nov 13:05 is outside the session and a held DAY exit has expired",
      eval(live_half) == paper.HALF_DAYS and not paper._session(datetime(2026, 11, 27, 13, 5, tzinfo=paper.ET))
      and paper._session(datetime(2026, 11, 27, 12, 59, tzinfo=paper.ET))
      and paper._expired({"tif": "DAY", "time": "2026-11-27T16:00:00+00:00"}, datetime(2026, 11, 27, 13, 5, tzinfo=paper.ET)))
lg = [L("a", 100, "STP", key=1), L("p", 100, "STP", pending=True, key=2)]
check("shrink: a pending-cancel leg is never changed; the older leg is cut instead", dict(co.shrink(150, lg)) == {"a": 50.0})
check("held_total: min(q, entries' fills, own fills + room); 0 off the closing side",
      co.held_total(100, 40, 0, 40, "SELL", [], co.groups("STP")) == 40
      and co.held_total(100, 100, 0, 150, "SELL", [L("o", 50, "STP")], co.groups("STP")) == 100
      and co.held_total(100, 100, 0, 120, "SELL", [L("o", 50, "STP")], co.groups("STP")) == 70
      and co.held_total(100, 100, 30, 0, "SELL", [], co.groups("STP")) == 30)

# ================================================================ a temporary ledger and a fake client 10 with a modify
TMP = Path(tempfile.mkdtemp())
paper.RUNS, paper.FILLS, paper.ORDERS = TMP, TMP / "fills.jsonl", TMP / "orders.jsonl"
paper.SLEEVES, paper.LOOKS, paper.NOTES, paper.SPLITS = TMP / "sleeves.json", TMP / "looks.json", TMP / "notes.json", TMP / "splits.jsonl"
paper.ARCHIVE = TMP / "archive"
paper.SLEEVES.write_text(json.dumps({"cand2": {"start_cash": 90000.0, "start": "2026-11-16T11:00:00+00:00", "model": "x"}}))
CLOCK = [datetime(2026, 11, 23, 10, 0, tzinfo=paper.ET)]
paper._now = lambda: CLOCK[0]
paper.now_iso = lambda: CLOCK[0].astimezone(timezone.utc).isoformat(timespec="seconds")   # the ledger's times follow it
N = [0]


def ex(sym, side, qty, ref, px=100.0):
    N[0] += 1
    return {"exec_id": f"e{N[0]}", "time": "20261123 10:00:00 US/Eastern", "symbol": sym, "side": side, "shares": qty,
            "price": px, "ref": ref, "commission": 1.0, "client_id": 10}


class GW:
    def __init__(self):
        self.execs, self.open, self.placed, self.cancelled, self.modified = [], [], [], [], []
        self.modify_result, self.place_status, self.fill_mkt = None, "Submitted", False

    def executions(self, days=7):
        return list(self.execs)

    def orders(self):
        return [dict(o) for o in self.open]

    def quotes(self, syms):
        return [{"symbol": s, "last": 100.0, "bid": 99.9, "ask": 100.1, "close": 100.0} for s in syms]

    def place(self, symbol, action, quantity, order_type, limit=None, aux=None, tif="DAY", ref=None, **f):
        self.placed.append({"symbol": symbol, "action": action, "quantity": quantity, "order_type": order_type, "ref": ref,
                            "tif": tif, **f})
        if self.place_status in ("Inactive", "Cancelled"):
            return {"status": self.place_status, "messages": ["201 refused"]}
        self.open.append({"ref": ref, "symbol": symbol, "action": action, "quantity": quantity, "order_type": order_type,
                          "aux": aux, "limit": limit, "tif": tif, "status": "Submitted", "remaining": quantity, "filled": 0,
                          "order_id": 1000 + len(self.placed)})          # as Gateway.orders() rows carry it
        if self.fill_mkt and order_type == "MKT" and tif != "OPG":
            self.fill(ref, quantity)
        return {"status": self.place_status, "messages": []}

    def cancel(self, ref=None):
        self.cancelled.append(ref)
        if not any(o["ref"] == ref for o in self.open):
            raise RuntimeError(f"Ref {ref!r} is not an open order")
        self.open = [o for o in self.open if o["ref"] != ref]
        return {"status": "Cancelled"}

    def modify(self, ref=None, quantity=None, **k):
        self.modified.append((ref, quantity))
        if getattr(self, "modify_raise", None) is not None:
            e = self.modify_raise
            if callable(e) and not isinstance(e, BaseException):
                e = e(ref, k)
            if e is not None:
                raise e
        if self.modify_result is not None:
            return self.modify_result
        for o in self.open:
            if (ref is not None and o["ref"] == ref) or (k.get("order_id") is not None and o.get("order_id") == k["order_id"]):
                o["quantity"], o["remaining"] = quantity, quantity - o["filled"]
                return {"modified": True, "quantity": quantity, "messages": []}
        raise RuntimeError("not open")

    def fill(self, ref, qty):
        o = next(o for o in self.open if o["ref"] == ref)
        self.execs.append(ex(o["symbol"], "BOT" if o["action"] == "BUY" else "SLD", qty, ref))
        o["filled"] += qty
        o["remaining"] -= qty
        if o["remaining"] <= 0:
            self.open.remove(o)


def world(gw):
    w = trader.PaperWorld.__new__(trader.PaperWorld)
    w.gw, w.sleeve, w.paper, w.dry_run, w.log = gw, "cand2", paper, False, None
    w.MAX_GROSS, w.SHOCK_CAP, w.TEXT = None, None, "B"
    return w


def fresh(pos=0, sym="X", at=(10, 0)):
    for f in (paper.FILLS, paper.ORDERS, paper.NOTES, paper.LOOKS):
        if f.exists():
            f.unlink()
    CLOCK[0] = datetime(2026, 11, 23, *at, tzinfo=paper.ET)
    gw = GW()
    if pos:
        gw.execs.append(ex(sym, "BOT" if pos > 0 else "SLD", abs(pos), "tt-cand2-0001"))
    paper.sync_fills(gw)
    return gw


def O(act, q, typ, sym="X", tif="GTC", co_=True, **k):
    o = {"symbol": sym, "action": act, "quantity": q, "type": typ, "tif": tif, **k}
    if co_:
        o["close_only"] = True
    if typ in ("STP", "STP LMT"):
        o.setdefault("stop", 90.0 if act == "SELL" else 110.0)
    if typ in ("LMT", "STP LMT", "LOC"):
        o.setdefault("limit", 110.0 if act == "SELL" else 90.0)
    if typ == "TRAIL":
        o.setdefault("trail_percent", 2.0)
    return o


ref_of = lambda gw, act, typ, sym="X": next(o["ref"] for o in gw.open if o["action"] == act and o["order_type"] == typ
                                            and o["symbol"] == sym)
held_ref = lambda: next(iter(paper.held_waiting("cand2")))
notes_now = lambda: paper.take_notes("cand2")
pos_of = lambda sym="X": paper.book("cand2")["positions"].get(sym, {}).get("quantity", 0)

# (a) base-90k-b 19 Nov 14:00: INTC -200 with a resting 200 stop; a MKT 100 close goes out, and the stop shrinks to 100
gw = fresh(-200)
world(gw).act({"place": [O("BUY", 200, "STP")], "cancel": []})
stop = ref_of(gw, "BUY", "STP")
gw.fill_mkt = True
notes = world(gw).act({"place": [O("BUY", 100, "MKT", tif="DAY")], "cancel": []})
o = next(o for o in gw.open if o["ref"] == stop)
check("(a) INTC: the MKT 100 close is sent beside the 200 stop; it fills; act's tail check reduces the stop to 100",
      any(p["order_type"] == "MKT" for p in gw.placed) and pos_of() == -100 and o["quantity"] == 100
      and gw.modified == [(stop, 100)], (notes, gw.modified))
n = " ".join(notes_now())
check("(a) the note: position now -100, the stop reduced to 100 left to fill", f"{stop} BUY 200 X STP was reduced to 100 left "
      "to fill" in n and "your position in X was -100 at the check" in n, n)

# (b) cand1-ml 19 Nov 14:30: AMD -45 with a 45 stop; a 15 LMT take-profit goes out (limit group); when it fills the stop shrinks
gw = fresh(-45)
world(gw).act({"place": [O("BUY", 45, "STP")], "cancel": []})
stop = ref_of(gw, "BUY", "STP")
world(gw).act({"place": [O("BUY", 15, "LMT", tif="DAY")], "cancel": []})
tp = ref_of(gw, "BUY", "LMT")
gw.fill(tp, 15)
paper.close_only_check(gw, "cand2")
check("(b) AMD: the 15 LMT is sent beside the 45 stop; after it fills the stop is 30",
      next(o for o in gw.open if o["ref"] == stop)["quantity"] == 30 and pos_of() == -30)
notes_now()

# (c) cand2 19 Nov 09:00: BUY 30 PANW MOO GTC, then its SELL 30 STP close_only: held; released at 30 at 09:30:45
gw = fresh(0, at=(9, 0))
notes = world(gw).act({"place": [O("BUY", 30, "MOO", co_=False), O("SELL", 30, "STP")], "cancel": []})
h = held_ref()
check("(c) PANW at 09:00: the MOO goes out, its stop is held (not at the broker) with a note naming the entry",
      [p["order_type"] for p in gw.placed] == ["MKT"] and any(f"held as {h}" in x and "it waits for tt-cand2-" in x
                                                              for x in notes), notes)
st = world(gw).state.__func__ if False else None
CLOCK[0] = datetime(2026, 11, 23, 9, 30, 45, tzinfo=paper.ET)
gw.fill(ref_of(gw, "BUY", "MKT"), 30)                  # the MOO fills in the auction; its order has ended
paper.close_only_check(gw, "cand2")
rel = [p for p in gw.placed if p["ref"] == h]
check("(c) 09:30:45: the check sees the entry Filled and still releases the stop at 30 under its own ref, then unlinks it",
      len(rel) == 1 and rel[0]["quantity"] == 30 and rel[0]["order_type"] == "STP" and h not in paper.held_waiting("cand2")
      and h in paper.co_state("cand2")["unlinked"], gw.placed)
n = " ".join(notes_now())
check("(c) the note: position now +30, the held stop went to the broker at 30", f"your held {h} SELL 30 X STP went to the "
      "broker at 30" in n and "was +30 at the check" in n, n)
paper.close_only_check(gw, "cand2")
check("(c) a later check never sends the ref again", len([p for p in gw.placed if p["ref"] == h]) == 1)

# (d) cand1 18 Nov 09:00: XLY -5; SELL 200 MOO and BUY 200 STP close_only: held (not released against the old 5 shares)
gw = fresh(-5, at=(9, 0))
world(gw).act({"place": [O("SELL", 200, "MOO", co_=False), O("BUY", 200, "STP")], "cancel": []})
h = held_ref()
check("(d) XLY at 09:00: the 200 stop is held, and act's tail check does not release it against the 5 held before",
      not [p for p in gw.placed if p["ref"] == h])
CLOCK[0] = datetime(2026, 11, 23, 9, 30, 45, tzinfo=paper.ET)
gw.fill(ref_of(gw, "SELL", "MKT"), 200)
paper.close_only_check(gw, "cand2")
check("(d) 09:30:45: released at 200 (the entry's 200 shares) on -205",
      [p["quantity"] for p in gw.placed if p["ref"] == h] == [200] and pos_of() == -205)
notes_now()

# (e) fable-ml 19 Nov 09:00 in the session: AMZN +18: SELL 9 LMT and SELL 9 STP, both close_only: both sent
gw = fresh(18)
world(gw).act({"place": [O("SELL", 9, "LMT", tif="DAY"), O("SELL", 9, "STP")], "cancel": []})
check("(e) AMZN +18: the 9 LMT and the 9 STP are both sent", len(gw.placed) == 2)
notes_now()

# (f) a bracket: BUY 100 LMT, then a 100 stop and a 100 take-profit, both close_only; fills 40, then 60; the TP fills 100
gw = fresh(0)
world(gw).act({"place": [O("BUY", 100, "LMT", tif="DAY", co_=False, limit=100.0), O("SELL", 100, "STP"),
                         O("SELL", 100, "LMT", tif="DAY", limit=120.0)], "cancel": []})
hs = sorted(paper.held_waiting("cand2"), key=lambda r: paper.co_state("cand2")["key"][r])
entry = ref_of(gw, "BUY", "LMT")
check("(f) both exits are held behind the working LMT entry", len(hs) == 2 and len(gw.placed) == 1)
gw.fill(entry, 40)
paper.close_only_check(gw, "cand2")
check("(f) the entry fills 40: the stop and the take-profit go out at 40 each (different groups)",
      sorted(p["quantity"] for p in gw.placed if p["ref"] in hs) == [40, 40])
gw.fill(entry, 60)
paper.close_only_check(gw, "cand2")
check("(f) the entry fills its last 60 (it ends): both grow to 100 by modify at that check, then are unlinked",
      sorted(q for r, q in gw.modified if r in hs) == [100, 100] and all(r in paper.co_state("cand2")["unlinked"] for r in hs),
      gw.modified)
tp = hs[1]
gw.fill(tp, 100)
paper.close_only_check(gw, "cand2")
check("(f) the take-profit fills 100: the stop is cancelled (no position left)", hs[0] in gw.cancelled and pos_of() == 0)
notes_now()

# (g) cand2-ml 18 Nov 14:00: OKTA +4 with a resting 4 stop; BUY 5 LMT entry and a SELL 5 STP: held; released at 5
gw = fresh(4)
world(gw).act({"place": [O("SELL", 4, "STP")], "cancel": []})
world(gw).act({"place": [O("BUY", 5, "LMT", tif="DAY", co_=False), O("SELL", 5, "STP", stop=85.0)], "cancel": []})
h = held_ref()
check("(g) OKTA: the new 5 stop is held (the stops would close 9 of 4), not released against the old 4",
      not [p for p in gw.placed if p["ref"] == h])
gw.fill(ref_of(gw, "BUY", "LMT"), 5)
paper.close_only_check(gw, "cand2")
check("(g) the entry fills 5: released at 5 beside the old 4 (9 held)", [p["quantity"] for p in gw.placed if p["ref"] == h] == [5])
notes_now()

# (h) base-ml 18 Nov 09:30: BUY MKT close_only on a name it holds none of, no entry: not sent, as on 17-19 Nov
gw = fresh(0)
notes = world(gw).act({"place": [O("BUY", 85, "MKT", tif="DAY")], "cancel": []})
check("(h) no position and no entry: not sent, 'so it would open a position'", not gw.placed and any(
    "(close_only) not sent: your position in X is +0 counting fills so far, so it would open a position" in x for x in notes), notes)

# (i) fills-races #1: before the open a MOO close beside a full stop is not sent (they can fill together at the open)
gw = fresh(100, at=(9, 0))
world(gw).act({"place": [O("SELL", 100, "STP")], "cancel": []})
notes = world(gw).act({"place": [O("SELL", 100, "MOO", tif="DAY")], "cancel": []})
check("(i) 09:00: a SELL 100 MOO close beside the 100 stop is not sent; the note names the stop orders",
      len(gw.placed) == 1 and any("your close_only orders there that could fill with it already cover 100, so it would close more than you hold"
                                  in x for x in notes), notes)

# (j) two full stops in the session: the second is not sent (a gap fills both)
gw = fresh(100)
world(gw).act({"place": [O("SELL", 100, "STP")], "cancel": []})
notes = world(gw).act({"place": [O("SELL", 100, "TRAIL")], "cancel": []})
check("(j) a 100 TRAIL beside the 100 stop is not sent (stops count together)", len(gw.placed) == 1 and any(
    "orders there that could fill with it already cover 100" in x for x in notes), notes)
notes_now()

# (k) a modify IBKR refuses: the leg is cancelled once, with IBKR's reason
gw = fresh(-200)
world(gw).act({"place": [O("BUY", 200, "STP")], "cancel": []})
stop = ref_of(gw, "BUY", "STP")
gw.execs.append(ex("X", "BOT", 100, "tt-cand2-0009"))     # another order closed 100
gw.modify_result = {"modified": False, "error_code": 110, "messages": ["110 bad tick"]}
paper.close_only_check(gw, "cand2")
n = " ".join(notes_now())
check("(k) refused modify: the stop is cancelled, and the note gives IBKR's reason", stop in gw.cancelled and "110 bad tick" in n
      and "could not be reduced to 100" in n, n)

# (l) a modify not confirmed (None, 2102): noted; the next check cancels instead of sending it again
gw = fresh(-200)
world(gw).act({"place": [O("BUY", 200, "STP")], "cancel": []})
stop = ref_of(gw, "BUY", "STP")
gw.execs.append(ex("X", "BOT", 100, "tt-cand2-0009"))
gw.modify_result = {"modified": None, "error_code": 2102, "messages": ["2102 still being processed"]}
paper.close_only_check(gw, "cand2")
n1 = " ".join(notes_now())
paper.close_only_check(gw, "cand2")
n2 = " ".join(notes_now())
check("(l) unconfirmed: the first check says so; the second sends no second modify and cancels the stop",
      "has not confirmed" in n1 and len(gw.modified) == 1 and stop in gw.cancelled and "did not confirm" in n2, (n1, n2))

# (k2-k6) the Gateway Owner's GatewayError reasons (tws 4b4269d): nothing was sent in each
class GatewayError(Exception):
    def __init__(self, msg, reason=None):
        super().__init__(msg)
        self.reason = reason


def reduce_case(raise_with):
    gw = fresh(-200)
    world(gw).act({"place": [O("BUY", 200, "STP")], "cancel": []})
    stop = ref_of(gw, "BUY", "STP")
    gw.execs.append(ex("X", "BOT", 100, "tt-cand2-0009"))
    gw.modify_raise = raise_with
    paper.close_only_check(gw, "cand2")
    return gw, stop, " ".join(notes_now())


gw, stop, n = reduce_case(GatewayError("new total not above filled", "filled"))
check("(k2) reason 'filled': the remainder is cancelled", stop in gw.cancelled, n)
gw, stop, n = reduce_case(GatewayError("order ended", "ended"))
check("(k3) reason 'ended': nothing cancelled, no note claiming a change", stop not in gw.cancelled and "reduced" not in n, n)
gw, stop, n = reduce_case(GatewayError("not open", "not_open"))
check("(k4) reason 'not_open': nothing cancelled", stop not in gw.cancelled, n)
gw, stop, n = reduce_case(GatewayError("connection lost", None))
tries = len(gw.modified)
paper.close_only_check(gw, "cand2")
paper.close_only_check(gw, "cand2")
check("(k5) nothing sent (a connection loss): the stop stands and the modify is tried again, three times in all",
      stop not in gw.cancelled and tries == 1 and len(gw.modified) == 3, (tries, len(gw.modified)))
paper.close_only_check(gw, "cand2")
check("(k5) after three unsent tries the next check cancels instead of trying a fourth", stop in gw.cancelled and len(gw.modified) == 3,
      len(gw.modified))
gw, stop, n = reduce_case(lambda ref, k: GatewayError("ambiguous ref", "ambiguous") if k.get("order_id") is None else None)
check("(k6) an ambiguous ref is tried once by its order_id", any(q == 100 for _, q in gw.modified[-1:]) and stop not in gw.cancelled
      and len(gw.modified) == 2, gw.modified)
row = {"time": "t", "sleeve": "cand2"}
for bad in (float("nan"), 0, -5, None, True):
    out = paper.resize(type("G", (), {"modify": lambda self, **k: (_ for _ in ()).throw(AssertionError("sent"))})(), "cand2", "r", bad)
    check(f"(k7) a total of {bad!r} is never sent", out[0] == "unsent", out)

# (k8-k11) the week7 review (20 Nov)
# (k8) a leg to be cut to 0 whose cancel failed is left; the excess goes to the next leg
gw = fresh(100)
world(gw).act({"place": [O("SELL", 60, "STP")], "cancel": []})
world(gw).act({"place": [O("SELL", 40, "STP", stop=85.0)], "cancel": []})
a, b_ = [p["ref"] for p in gw.placed]
gw.execs.append(ex("X", "SLD", 50, "tt-cand2-0009"))      # a plain SELL 50 closed half
real_cancel = gw.cancel
gw.cancel = lambda ref=None: (gw.cancelled.append(ref), (_ for _ in ()).throw(RuntimeError("504 lost")))[1] if ref == b_ else real_cancel(ref)
paper.close_only_check(gw, "cand2")
first = list(gw.modified)
paper.close_only_check(gw, "cand2")
check("(k8) B (40, newest) is cut to 0 and its cancel fails, A goes to 50; at the next check A is reduced to 10 so 50 "
      "cover +50", first == [(a, 50)] and gw.modified[-1:] == [(a, 10)] and gw.cancelled == [b_], (first, gw.modified, gw.cancelled))
gw.cancel = real_cancel
notes_now()

# (k9) a linked exit whose raise sent nothing stays linked and is raised at a later check
gw = fresh(0)
world(gw).act({"place": [O("BUY", 100, "LMT", co_=False), O("SELL", 100, "STP")], "cancel": []})
h = held_ref()
gw.fill(ref_of(gw, "BUY", "LMT"), 60)
paper.close_only_check(gw, "cand2")
gw.fill(ref_of(gw, "BUY", "LMT"), 40)                     # the entry has ended
gw.modify_raise = GatewayError("echo and status disagree", "size_mismatch")
paper.close_only_check(gw, "cand2")
gw.modify_raise = None
paper.close_only_check(gw, "cand2")
check("(k9) the raise to 100 sent nothing once (size_mismatch); the leg stays linked and the next check raises it",
      gw.modified[-1] == (h, 100) and next(o for o in gw.open if o["ref"] == h)["quantity"] == 100, gw.modified)
notes_now()

# (k10) a raise not confirmed (None) keeps its room in the check: a second held exit is not released into it
gw = fresh(0)
world(gw).act({"place": [O("BUY", 100, "LMT", co_=False), O("SELL", 100, "STP"), O("SELL", 100, "TRAIL")], "cancel": []})
gw.fill(ref_of(gw, "BUY", "LMT"), 60)
paper.close_only_check(gw, "cand2")
stp = next(p["ref"] for p in gw.placed if p["order_type"] == "STP")
placed_before = len(gw.placed)
gw.fill(ref_of(gw, "BUY", "LMT"), 40)
gw.modify_result = {"modified": None, "status": "Submitted", "messages": ["2102 still being processed"]}
paper.close_only_check(gw, "cand2")
check("(k10) STP raise to 100 unconfirmed: the TRAIL is not released at 40 beside it", len(gw.placed) == placed_before
      and gw.modified[-1] == (stp, 100), (gw.placed[placed_before:], gw.modified))
paper.close_only_check(gw, "cand2")
check("(k10) at the next check too, the unconfirmed 100 still counts: no TRAIL", len(gw.placed) == placed_before, gw.placed[placed_before:])
gw.modify_result = None
notes_now()

# (k11) an order that filled during the modify is "gone": no note says it still stands
gw = fresh(-200)
world(gw).act({"place": [O("BUY", 200, "STP")], "cancel": []})
stop = ref_of(gw, "BUY", "STP")
gw.execs.append(ex("X", "BOT", 100, "tt-cand2-0009"))
gw.modify_result = {"modified": None, "status": "Filled", "messages": ["Order Filled during the modify; check orders()"]}
paper.close_only_check(gw, "cand2")
n = " ".join(notes_now())
check("(k11) filled during the modify: no 'not confirmed' claim and no cancel", "has not confirmed" not in n and stop not in gw.cancelled, n)
gw.modify_result = None

# (m) the model cancels its held exit: cancelled in the ledger, never sent, gone from the state
gw = fresh(0)
world(gw).act({"place": [O("BUY", 50, "LMT", tif="DAY", co_=False), O("SELL", 50, "STP")], "cancel": []})
h = held_ref()
world(gw).act({"place": [], "cancel": [h]})
gw.fill(ref_of(gw, "BUY", "LMT"), 50)
paper.close_only_check(gw, "cand2")
check("(m) a held exit the model cancelled is never released, and its cancel never reached the broker",
      not [p for p in gw.placed if p["ref"] == h] and h not in gw.cancelled and h in paper.cancel_requested("cand2"))
notes_now()

# (n) a held DAY exit not released by the close: cancelled in the ledger at the next check
gw = fresh(0)
world(gw).act({"place": [O("BUY", 50, "LMT", tif="GTC", co_=False), O("SELL", 50, "STP", tif="DAY")], "cancel": []})
h = held_ref()
CLOCK[0] = datetime(2026, 11, 24, 9, 0, tzinfo=paper.ET)
paper.close_only_check(gw, "cand2")
n = " ".join(notes_now())
check("(n) next morning: the held DAY stop was cancelled at the close, never sent", h not in paper.held_waiting("cand2")
      and "DAY was cancelled at the close; it never went to the broker" in n, n)

# (o) the entry is cancelled with no fill: the held exit is cancelled in the ledger
gw = fresh(0)
world(gw).act({"place": [O("BUY", 50, "LMT", tif="DAY", co_=False), O("SELL", 50, "STP")], "cancel": []})
h, entry = held_ref(), ref_of(gw, "BUY", "LMT")
world(gw).act({"place": [], "cancel": [entry]})
paper.close_only_check(gw, "cand2")
n = " ".join(notes_now())
check("(o) entry cancelled unfilled: the held stop is cancelled, 'ended with no shares filled since ... was placed'",
      h not in paper.held_waiting("cand2") and "ended with no shares filled since your held" in n
      and "it was cancelled and never went to the broker" in n, n)

# (p) IBKR refuses the release: noted once, never retried
gw = fresh(0)
world(gw).act({"place": [O("BUY", 50, "LMT", tif="DAY", co_=False), O("SELL", 50, "STP")], "cancel": []})
h = held_ref()
gw.fill(ref_of(gw, "BUY", "LMT"), 50)
gw.place_status = "Inactive"
paper.close_only_check(gw, "cand2")
paper.close_only_check(gw, "cand2")
n = " ".join(notes_now())
check("(p) a refused release is noted with IBKR's message and never sent again",
      len([p for p in gw.placed if p["ref"] == h]) == 1 and "was sent to the broker at 50 and refused: Inactive 201 refused" in n, n)

# (q) an unconfirmed release ("Unknown") counts as released: never sent twice
gw = fresh(0)
world(gw).act({"place": [O("BUY", 50, "LMT", tif="DAY", co_=False), O("SELL", 50, "STP")], "cancel": []})
h = held_ref()
gw.fill(ref_of(gw, "BUY", "LMT"), 50)
gw.place_status = "Unknown"
paper.close_only_check(gw, "cand2")
gw.open = [o for o in gw.open if o["ref"] != h]          # IBKR never had it
paper.close_only_check(gw, "cand2")
check("(q) 'Unknown': released once, never re-sent, and its note says the broker has not confirmed it",
      len([p for p in gw.placed if p["ref"] == h]) == 1 and "has not confirmed it yet" in " ".join(notes_now()))

# (r) the state and its words: a held exit is listed as held, with the refs it waits for, and the held line once
gw = fresh(0)
world(gw).act({"place": [O("BUY", 50, "LMT", tif="DAY", co_=False), O("SELL", 50, "STP")], "cancel": []})
h = held_ref()
w = world(gw)
orders = [o for o in paper.open_orders(gw, "cand2")]
ws = {"value": 90000.0, "cash": 90000.0, "positions": [], "events": [], "gross": 0.0, "net": 0.0, "beta_net": 0.0,
      "orders": []}
st = paper.co_state("cand2")
rows = [{"ref": h, "symbol": "X", "action": "SELL", "quantity": 50, "type": "STP", "stop": 90.0, "limit": None,
         "trail_percent": None, "tif": "GTC", "cancel_pending": False, "close_only": True, "held": True,
         "after": list(st["held"][h]["held"]["after"])}]
lines = trader.render_account(dict(ws, orders=rows), {}, "B") if hasattr(trader, "render_account") else ""
txt = lines if isinstance(lines, str) else "\n".join(lines)
check("(r) render: 'held close_only (waits for <entry>)' and the held line, once",
      f"{h} X SELL 50 STP - 90 - GTC held close_only (waits for " in txt and txt.count(trader.HELD_LINE) == 1, txt[-600:])
bst = {"value": 90000.0, "positions": [{"symbol": "SPY", "quantity": 100, "close": 600.0}], "orders": []}
stp = {"symbol": "SPY", "action": "SELL", "quantity": 100, "type": "STP", "cancel_pending": False}
br = [trader.book_risk(dict(bst, orders=x), "20261120") for x in ([], [dict(stp, held=True)], [stp])]
check("(r) book_risk: a held stop is not a working stop (as with no stop; a working stop reads differently)",
      br[0] is not None and br[0] == br[1] and br[1] != br[2], br)

# (s) a pending-cancel leg is never modified
gw = fresh(-200)
world(gw).act({"place": [O("BUY", 200, "STP")], "cancel": []})
stop = ref_of(gw, "BUY", "STP")
next(o for o in gw.open if o["ref"] == stop)["status"] = "PendingCancel"
gw.execs.append(ex("X", "BOT", 100, "tt-cand2-0009"))
paper.close_only_check(gw, "cand2")
check("(s) PendingCancel: no modify and no second cancel", gw.modified == [] and stop not in gw.cancelled)
notes_now()

# (t) a partly filled leg: the new total is its own fills plus the new remainder
gw = fresh(-200)
world(gw).act({"place": [O("BUY", 200, "STP")], "cancel": []})
stop = ref_of(gw, "BUY", "STP")
gw.fill(stop, 30)                                        # the stop fills 30: -170
gw.execs.append(ex("X", "BOT", 120, "tt-cand2-0009"))    # another order closes 120: -50
paper.close_only_check(gw, "cand2")
check("(t) a 200 stop with 30 filled on -50: total set to 80 (30 filled + 50 left)", gw.modified == [(stop, 80)], gw.modified)
notes_now()

# (v) the review's FIS case (cand2, 17 Nov 15:00): SELL 363 MKT fills 300 within the 2 s; its BUY 363 TRAIL close_only is
# held with its entry's snapshot at 0, so act's tail check sends it at 300, and it grows to 363 with the last 63
class GWp(GW):
    def place(self, symbol, action, quantity, order_type, limit=None, aux=None, tif="DAY", ref=None, **f):
        out = super().place(symbol, action, quantity, order_type, limit, aux, tif, ref, **f)
        if order_type == "MKT" and getattr(self, "part", None):
            self.fill(ref, self.part)
            self.part = None
        return out


gw = fresh(0)
gw.__class__ = GWp
gw.part = 300
world(gw).act({"place": [O("SELL", 363, "MKT", tif="DAY", co_=False), O("BUY", 363, "TRAIL")], "cancel": []})
st_v = paper.co_state("cand2")
h = next(r for r in st_v["held"])
rel = [p["quantity"] for p in gw.placed if p["ref"] == h]
check("(v) FIS: the TRAIL is held with its same-submit entry at 0, and act's tail check sends it at 300 on -300",
      list(st_v["held"][h]["held"]["after"].values()) == [0.0] and rel == [300] and pos_of() == -300, (rel, st_v["held"][h]["held"]))
gw.fill(ref_of(gw, "SELL", "MKT"), 63)
paper.close_only_check(gw, "cand2")
check("(v) the entry's last 63 fill: the TRAIL grows to 363 by modify", gw.modified[-1] == (h, 363), gw.modified)
notes_now()

# (w) the review's blocker: a working MKT close counts beside a new stop (both could fill)
gw = fresh(-100)
world(gw).act({"place": [O("BUY", 100, "MKT", tif="DAY")], "cancel": []})          # not filled (fill_mkt off)
notes = world(gw).act({"place": [O("BUY", 100, "STP")], "cancel": []})
check("(w) a working MKT 100 close on -100: a new 100 stop is not sent", len(gw.placed) == 1 and any(
    "would close more than you hold" in x for x in notes), notes)

# (x) a leg whose earlier cancel failed is still reduced (not taken as cancel-pending)
gw = fresh(-200)
world(gw).act({"place": [O("BUY", 200, "STP")], "cancel": []})
stop = ref_of(gw, "BUY", "STP")
paper.append(paper.ORDERS, {"time": paper.now_iso(), "sleeve": "cand2", "cancel": stop, "error": "GatewayError: 504"})
gw.execs.append(ex("X", "BOT", 100, "tt-cand2-0009"))
paper.close_only_check(gw, "cand2")
check("(x) after a failed cancel the stop is still reduced to 100", gw.modified == [(stop, 100)], gw.modified)
notes_now()

# (y) two held exits on one entry (an older STP, a newer TRAIL): the older grows first; the newer finds no room and ends
gw = fresh(0)
world(gw).act({"place": [O("BUY", 100, "LMT", tif="DAY", co_=False), O("SELL", 100, "STP"), O("SELL", 100, "TRAIL")],
               "cancel": []})
keys = paper.co_state("cand2")["key"]
stp_h, trl_h = sorted(paper.held_waiting("cand2"), key=lambda r: keys[r])
entry = ref_of(gw, "BUY", "LMT")
gw.fill(entry, 60)
paper.close_only_check(gw, "cand2")
gw.fill(entry, 40)
paper.close_only_check(gw, "cand2")
n = " ".join(notes_now())
check("(y) the STP goes out at 60 then grows to 100; the TRAIL (same group) is never sent and ends with the right words",
      [p["quantity"] for p in gw.placed if p["ref"] == stp_h] == [60] and (stp_h, 100) in gw.modified
      and not [p for p in gw.placed if p["ref"] == trl_h] and trl_h not in paper.held_waiting("cand2")
      and "already cover it" in n, (gw.modified, n))

# (z) HOLD: nothing is released while place_ok is False; released once it is True
gw = fresh(0)
world(gw).act({"place": [O("BUY", 50, "LMT", tif="DAY", co_=False), O("SELL", 50, "STP")], "cancel": []})
h = held_ref()
gw.fill(ref_of(gw, "BUY", "LMT"), 50)
paper.close_only_check(gw, "cand2", place_ok=False)
first = [p for p in gw.placed if p["ref"] == h]
paper.close_only_check(gw, "cand2")
check("(z) HOLD: not released while place_ok is False (and not cancelled); released at the next check",
      not first and [p["quantity"] for p in gw.placed if p["ref"] == h] == [50])
notes_now()

# (aa) notes are written even if the check stops part-way
gw = fresh(0)
world(gw).act({"place": [O("BUY", 50, "LMT", tif="DAY", co_=False), O("SELL", 50, "STP")], "cancel": []})
gw.fill(ref_of(gw, "BUY", "LMT"), 50)
real_resize = paper.resize
paper.resize = lambda *a, **k: (_ for _ in ()).throw(OSError("boom"))
gw.execs.append(ex("Y", "SLD", 10, "tt-cand2-0010"))
world(gw).act({"place": [O("BUY", 10, "STP", sym="Y"), O("BUY", 10, "STP", sym="Y", stop=111.0)], "cancel": []})
paper.close_only_check(gw, "cand2")
paper.resize = real_resize
check("(aa) a failure after a release still leaves the release's note", any("went to the broker at 50" in x for x in notes_now()))

# (bb) the switch turned off with a held exit: cancelled in the ledger, with a note; the model's cancel stays local
gw = fresh(0)
world(gw).act({"place": [O("BUY", 50, "LMT", tif="DAY", co_=False), O("SELL", 50, "STP")], "cancel": []})
h = held_ref()
co.V2 = False
paper.close_only_check(gw, "cand2")
n = " ".join(notes_now())
check("(bb) switch off: the held exit is cancelled in the ledger with a note, never sent", h not in paper.held_waiting("cand2")
      and f"your held {h}" in n and not [p for p in gw.placed if p["ref"] == h], n)
co.V2 = True

# (cc) calibration: a held exit never released is not scored; a released one is, with its release's result
import calib_report  # noqa: E402
gw = fresh(0)
world(gw).act({"place": [O("BUY", 50, "LMT", tif="DAY", co_=False, p_fill=0.5, p_win=0.5),
                         O("SELL", 50, "STP", p_fill=0.4, p_win=0.4)], "cancel": []})
h = held_ref()
before = [r["ref"] for r in calib_report.load(TMP)[0]]
gw.fill(ref_of(gw, "BUY", "LMT"), 50)
paper.close_only_check(gw, "cand2")
after = {r["ref"]: r for r in calib_report.load(TMP)[0]}
check("(cc) calib_report: the held exit is left out until released, then scored with its release's result",
      h not in before and h in after and (after[h].get("result") or {}).get("status") == "Submitted")
notes_now()

# ================================================================ the Researcher's exit-size rule (P, the late gate, 25 Nov 2026)
# T = min(q, Fe + P, F + room); P = min(max(0, q - E), H) once none of the exit's entries still works, else 0. E what its
# entries had left to fill at the hold, H the shares held then on its closing side less its entries' own fills since their
# snapshots (models-w3 d13f1c02, week7/co_note/P_DECISION.md). The cases L1-L10 are the Researcher's.
check("part_above: P = min(max(0, q - E), H); H 0 gives 0; a row held before 25 Nov (no E or H) 0",
      co.part_above(100, 80, 20) == 20 and co.part_above(7, 5, 2) == 2 and co.part_above(80, 80, 20) == 0
      and co.part_above(50, 80, 20) == 0 and co.part_above(100, 50, 20) == 20 and co.part_above(100, 50, 0) == 0
      and co.part_above(100, None, 20) == 0 and co.part_above(100, 50, None) == 0)
check("held_total with P: min(q, Fe + P, F + room); off the closing side only its own fills; room still bounds it",
      co.held_total(100, 0, 0, -20, "BUY", [], co.groups("TRAIL"), p=20) == 20
      and co.held_total(100, 80, 20, -100, "BUY", [], co.groups("TRAIL"), p=20) == 100
      and co.held_total(100, 0, 0, 20, "BUY", [], co.groups("TRAIL"), p=20) == 0
      and co.held_total(100, 0, 0, -20, "BUY", [L("o", 20, "STP")], co.groups("STP"), p=20) == 0
      and co.held_total(100, 30, 0, -10, "BUY", [], co.groups("STP"), p=20) == 10)
held_row = lambda h: paper.co_state("cand2")["held"][h]
sent_at = lambda gw, h: [p["quantity"] for p in gw.placed if p["ref"] == h]
first_held = lambda: next(iter(paper.co_state("cand2")["held"]))
CLAUSE = "; your other close_only orders there that could fill with it already cover the rest of your position"


def book20():
    """-20 held; SELL 80 LMT resting (the entry) and BUY 100 STP close_only (held): the Researcher's book."""
    g = fresh(-20)
    world(g).act({"place": [O("SELL", 80, "LMT", tif="DAY", co_=False), O("BUY", 100, "STP")], "cancel": []})
    return g, first_held(), ref_of(g, "SELL", "LMT")


# (L1) nothing goes out while the entry has 0 fills; it fills 80: the exit goes out at 100
gw, h, ent = book20()
check("(L1) -20, SELL 80 LMT resting, BUY 100 STP: held with E 80 and H 20, nothing out at the tail check",
      sent_at(gw, h) == [] and held_row(h)["held"].get("left") == 80 and held_row(h)["held"].get("had") == 20, gw.placed)
paper.close_only_check(gw, "cand2")
check("(L1) and nothing at a later check while the entry still has 0 fills", sent_at(gw, h) == [], gw.placed)
gw.fill(ent, 80)
paper.close_only_check(gw, "cand2")
n = " ".join(notes_now())
check("(L1) the entry fills 80 (ended): the exit goes out at 100 (Fe 80 + P 20), unlinked, no room clause",
      sent_at(gw, h) == [100] and h in paper.co_state("cand2")["unlinked"] and "went to the broker at 100" in n
      and CLAUSE not in n, (gw.placed, n))

# (L2) the entry fills 30 and still works: the exit goes out at 30, shown growing toward 100; the entry completes: 100
gw, h, ent = book20()
gw.fill(ent, 30)
paper.close_only_check(gw, "cand2")
check("(L2) the entry fills 30 and still works: out at 30 (P waits)", sent_at(gw, h) == [30] and pos_of() == -50, gw.placed)
w2 = world(gw)
w2.data, w2.dry_run = __import__("types").SimpleNamespace(dates=["20261120"], ix={}), True
row2 = next(o for o in w2.state(1, {}, live={"X": 100.0})["orders"] if o["ref"] == h)
check("(L2) the state: at the broker for 30, growing toward 80 as the entry fills (30 + its 50 left; P comes once, when "
      "it has ended, so it is not in the figure: the Researcher, 25 Nov)",
      row2.get("grows_to") == 80 and row2.get("after") == [ent] and not row2.get("held"), row2)
gw.fill(ent, 50)
paper.close_only_check(gw, "cand2")
check("(L2) the entry completes: it grows to 100 and is unlinked", gw.modified[-1:] == [(h, 100)]
      and h in paper.co_state("cand2")["unlinked"], gw.modified)
notes_now()

# (L3) the 30-share exit fills while the entry works: unlinked; the entry's later fills get nothing (as before P)
gw, h, ent = book20()
gw.fill(ent, 30)
paper.close_only_check(gw, "cand2")
gw.fill(h, 30)
paper.close_only_check(gw, "cand2")
gw.fill(ent, 50)
paper.close_only_check(gw, "cand2")
check("(L3) the exit at 30 fills while its entry works: unlinked, and the entry's last 50 get nothing from it",
      sent_at(gw, h) == [30] and gw.modified == [] and h in paper.co_state("cand2")["unlinked"], (gw.placed, gw.modified))
notes_now()

# (L4) the sleeve cancels the entry at 0 fills: the exit goes out at 20 (P, for the 20 held at the hold)
gw, h, ent = book20()
world(gw).act({"place": [], "cancel": [ent]})
paper.close_only_check(gw, "cand2")
check("(L4) entry cancelled unfilled: out at 20, the shares held at the hold, and unlinked", sent_at(gw, h) == [20]
      and h in paper.co_state("cand2")["unlinked"], gw.placed)
notes_now()
gw = fresh(0)                                            # from a flat book (H 0): cancelled, as before P
world(gw).act({"place": [O("SELL", 80, "LMT", tif="DAY", co_=False), O("BUY", 100, "STP")], "cancel": []})
h, ent = held_ref(), ref_of(gw, "SELL", "LMT")
world(gw).act({"place": [], "cancel": [ent]})
paper.close_only_check(gw, "cand2")
n = " ".join(notes_now())
check("(L4) from a flat book: cancelled in the ledger, never sent ('ended with no shares filled')", sent_at(gw, h) == []
      and h not in paper.held_waiting("cand2") and "ended with no shares filled since your held" in n, n)
gw = fresh(-20)                                          # H 20 but an old stop covers it: cancelled, with that reason
world(gw).act({"place": [O("BUY", 20, "STP")], "cancel": []})
world(gw).act({"place": [O("SELL", 80, "LMT", tif="DAY", co_=False), O("BUY", 100, "STP", stop=111.0)], "cancel": []})
h, ent = held_ref(), ref_of(gw, "SELL", "LMT")
world(gw).act({"place": [], "cancel": [ent]})
paper.close_only_check(gw, "cand2")
n = " ".join(notes_now())
check("(L4) H 20 but the old 20 stop covers it: cancelled with 'already cover it', not 'no shares filled'",
      sent_at(gw, h) == [] and h not in paper.held_waiting("cand2") and "already cover it" in n
      and "ended with no shares filled" not in n, n)

# (L5) the rally: the entry fills 80 between two checks (nothing was out): at the next check it goes out at 100
gw, h, ent = book20()
gw.fill(ent, 80)
paper.close_only_check(gw, "cand2")
check("(L5) the rally fills the entry between checks: released at 100 at the next check", sent_at(gw, h) == [100], gw.placed)
notes_now()

# (L6) the review's finding 1: flat; SELL 50 LMT and BUY 100 STP; a separate SELL 50 MKT fills: nothing; the entry fills: 50
gw = fresh(0)
world(gw).act({"place": [O("SELL", 50, "LMT", tif="DAY", co_=False), O("BUY", 100, "STP")], "cancel": []})
h, ent = held_ref(), ref_of(gw, "SELL", "LMT")
gw.execs.append(ex("X", "SLD", 50, "tt-cand2-0009"))     # another order's 50: -50
paper.close_only_check(gw, "cand2")
check("(L6) flat at the hold (H 0): another order's 50 shares get nothing", sent_at(gw, h) == [] and held_row(h)["held"].get("had") == 0)
gw.fill(ent, 50)
paper.close_only_check(gw, "cand2")
check("(L6) its entry fills 50: out at 50, not 100", sent_at(gw, h) == [50], gw.placed)
notes_now()

# (L7) flat; SELL 50 MKT and BUY 100 STP in one submit, the MKT filling 30 inside the 2 seconds: H 0 (its own fills)
gw = fresh(0)
gw.__class__ = GWp
gw.part = 30
world(gw).act({"place": [O("SELL", 50, "MKT", tif="DAY", co_=False), O("BUY", 100, "STP")], "cancel": []})
h = first_held()
check("(L7) H leaves out the entry's own 30 filled in the re-read: H 0, E 50", held_row(h)["held"].get("had") == 0
      and held_row(h)["held"].get("left") == 50 and list(held_row(h)["held"]["after"].values()) == [0.0], held_row(h)["held"])
gw.fill(ref_of(gw, "SELL", "MKT"), 20)
paper.close_only_check(gw, "cand2")
check("(L7) the MKT's last 20: the exit totals 50, its entry's fills, with P 0", (sent_at(gw, h) + [m[1] for m in gw.modified
      if m[0] == h])[-1:] == [50] and pos_of() == -50, (gw.placed, gw.modified))
notes_now()

# (L8) HPE and ADI as on Tuesday: 100 and 7
gw = fresh(-20)
world(gw).act({"place": [O("SELL", 80, "MKT", tif="DAY", co_=False), O("BUY", 100, "TRAIL")], "cancel": []})
h = first_held()
check("(L8) HPE: held (the MKT not filled within the 2 s), nothing out at the tail check", sent_at(gw, h) == []
      and held_row(h)["held"].get("left") == 80 and held_row(h)["held"].get("had") == 20, gw.placed)
gw.fill(ref_of(gw, "SELL", "MKT"), 80)
paper.close_only_check(gw, "cand2")
check("(L8) HPE: the MKT fills 80: the TRAIL goes out at 100 on -100", sent_at(gw, h) == [100] and pos_of() == -100, gw.placed)
notes_now()
gw = fresh(-2)
world(gw).act({"place": [O("BUY", 2, "STP")], "cancel": []})
old = ref_of(gw, "BUY", "STP")
world(gw).act({"place": [O("SELL", 5, "MKT", tif="DAY", co_=False), O("BUY", 7, "STP", stop=340.0)], "cancel": [old]})
h = first_held()
check("(L8) ADI: the old stop cancelled, the new STP held (E 5, H 2)", old in gw.cancelled and sent_at(gw, h) == []
      and held_row(h)["held"].get("left") == 5 and held_row(h)["held"].get("had") == 2, (gw.placed, gw.cancelled))
gw.fill(ref_of(gw, "SELL", "MKT"), 5)
paper.close_only_check(gw, "cand2")
check("(L8) ADI: the MKT fills 5: the STP goes out at 7 on -7", sent_at(gw, h) == [7] and pos_of() == -7, gw.placed)
notes_now()

# (L9) an old BUY 20 stop standing on -20; SELL 80 MKT and BUY 100 STP: 80, with the room clause in the note
gw = fresh(-20)
world(gw).act({"place": [O("BUY", 20, "STP")], "cancel": []})
world(gw).act({"place": [O("SELL", 80, "MKT", tif="DAY", co_=False), O("BUY", 100, "STP", stop=111.0)], "cancel": []})
h = first_held()
notes_now()
gw.fill(ref_of(gw, "SELL", "MKT"), 80)
paper.close_only_check(gw, "cand2")
n = " ".join(notes_now())
check("(L9) the old stop covers 20: out at 80, and the note says why", sent_at(gw, h) == [80]
      and f"went to the broker at 80{CLAUSE}" in n, n)
gw = fresh(-20)                                          # a LMT exit beside the old stop (the limit group is empty): 100
world(gw).act({"place": [O("BUY", 20, "STP")], "cancel": []})
world(gw).act({"place": [O("SELL", 80, "MKT", tif="DAY", co_=False), O("BUY", 100, "LMT")], "cancel": []})
h = first_held()
gw.fill(ref_of(gw, "SELL", "MKT"), 80)
paper.close_only_check(gw, "cand2")
n = " ".join(notes_now())
check("(L9) a LMT exit beside the old stop (another group): out at 100, no clause", sent_at(gw, h) == [100] and CLAUSE not in n, n)

# (L10) HOLD present when the entries end: released after HOLD lifts
gw, h, ent = book20()
gw.fill(ent, 80)
paper.close_only_check(gw, "cand2", place_ok=False)
first = sent_at(gw, h)
paper.close_only_check(gw, "cand2")
check("(L10) HOLD when the entry ends: nothing sent; at the next check without HOLD, 100", first == [] and sent_at(gw, h) == [100])
notes_now()

# (L11) the grow note's clause: linked at 30 (entry 30 of 80), an old 20 stop beside it; the entry completes: 80 + clause
gw = fresh(-20)
world(gw).act({"place": [O("BUY", 20, "STP")], "cancel": []})
world(gw).act({"place": [O("SELL", 80, "LMT", tif="DAY", co_=False), O("BUY", 100, "STP", stop=111.0)], "cancel": []})
h, ent = first_held(), ref_of(gw, "SELL", "LMT")
gw.fill(ent, 30)
paper.close_only_check(gw, "cand2")
gw.fill(ent, 50)
paper.close_only_check(gw, "cand2")
n = " ".join(notes_now())
check("(L11) out at 30, then grown to 80 (not 100) with the clause: the old 20 stop covers the rest",
      sent_at(gw, h) == [30] and gw.modified[-1:] == [(h, 80)] and f"went from 30 to 80 at the broker{CLAUSE}" in n, (gw.modified, n))
gw = fresh(-20)                                          # the position itself smaller (no other leg): no clause
world(gw).act({"place": [O("SELL", 80, "LMT", tif="DAY", co_=False), O("BUY", 100, "STP")], "cancel": []})
h, ent = first_held(), ref_of(gw, "SELL", "LMT")
gw.execs.append(ex("X", "BOT", 30, "tt-cand2-0009"))     # a plain BUY 30: -20 + 30 = +10 before the entry fills
gw.fill(ent, 80)                                         # +10 - 80 = -70
paper.close_only_check(gw, "cand2")
n = " ".join(notes_now())
check("(L11) the position smaller than Fe + P with no other leg: out at 70, and no clause", sent_at(gw, h) == [70] and CLAUSE not in n, n)

# (L13) another stop covers part of the position, but the exit's own size is what holds it: no clause
gw = fresh(-20)
world(gw).act({"place": [O("BUY", 20, "STP")], "cancel": []})
world(gw).act({"place": [O("SELL", 80, "MKT", tif="DAY", co_=False), O("BUY", 50, "STP", stop=111.0)], "cancel": []})
h = first_held()
gw.fill(ref_of(gw, "SELL", "MKT"), 80)
paper.close_only_check(gw, "cand2")
n = " ".join(notes_now())
check("(L13) BUY 50 for SELL 80 beside the old 20 stop on -100: out at 50, its size, and no clause", sent_at(gw, h) == [50]
      and CLAUSE not in n, n)

# (L12) the entry filled in full by the fills just booked, though the orders read first still list it: P counts at once
gw, h, ent = book20()
rows = gw.orders()                                       # the entry listed as working
gw.fill(ent, 80)
paper.sync_fills(gw)
paper.close_only_check(gw, "cand2", rows=rows, synced=True)
check("(L12) the entry still listed but filled in full by the fills just booked: out at 100 at this check",
      sent_at(gw, h) == [100], gw.placed)
notes_now()

# (P2b) ADI with the old stop's cancel still pending: nothing out before the entry fills; confirmed before it fills: 7
gw = fresh(-2)
world(gw).act({"place": [O("BUY", 2, "STP")], "cancel": []})
old = ref_of(gw, "BUY", "STP")


def pending_cancel(ref=None):
    gw.cancelled.append(ref)
    for o in gw.open:
        if o["ref"] == ref:
            o["status"] = "PendingCancel"
    return {"status": "PendingCancel"}


gw.cancel = pending_cancel
world(gw).act({"place": [O("SELL", 5, "MKT", tif="DAY", co_=False), O("BUY", 7, "STP", stop=340.0)], "cancel": [old]})
h = held_ref()
check("(P2b) the old stop CANCEL-PENDING: the new STP held, nothing out", sent_at(gw, h) == [] and h in paper.held_waiting("cand2"))
gw.open = [o for o in gw.open if o["ref"] != old]         # IBKR confirms the cancel
gw.fill(ref_of(gw, "SELL", "MKT"), 5)
paper.close_only_check(gw, "cand2")
check("(P2b) confirmed before the entry fills: 7 when it fills", sent_at(gw, h) == [7], gw.placed)
notes_now()
gw = fresh(-2)                                            # still pending at the check where the entry ends: 5 (accepted)
world(gw).act({"place": [O("BUY", 2, "STP")], "cancel": []})
old = ref_of(gw, "BUY", "STP")
gw.cancel = pending_cancel
world(gw).act({"place": [O("SELL", 5, "MKT", tif="DAY", co_=False), O("BUY", 7, "STP", stop=340.0)], "cancel": [old]})
h = held_ref()
gw.fill(ref_of(gw, "SELL", "MKT"), 5)
paper.close_only_check(gw, "cand2")
check("(P2b) still pending when the entry ends: 5 (the pending leg's 2 count; accepted, the Researcher's daily count "
      "watches for it)", sent_at(gw, h) == [5], gw.placed)
notes_now()

# (L14) the third review of 25 Nov: the entry still listed but filled in full by the fills just booked, an old stop's cancel
# pending (room 80): released at 80 and unlinked at that check, so P is never used later when the room frees
gw = fresh(-20)
world(gw).act({"place": [O("BUY", 20, "STP")], "cancel": []})
old = ref_of(gw, "BUY", "STP")
gw.cancel = pending_cancel
world(gw).act({"place": [O("SELL", 80, "LMT", tif="DAY", co_=False), O("BUY", 100, "TRAIL")], "cancel": [old]})
h, ent = first_held(), ref_of(gw, "SELL", "LMT")
rows = gw.orders()
gw.fill(ent, 80)
paper.sync_fills(gw)
paper.close_only_check(gw, "cand2", rows=rows, synced=True)
check("(L14) released at 80 beside the pending 20 and unlinked at once", sent_at(gw, h) == [80]
      and h in paper.co_state("cand2")["unlinked"], (gw.placed, paper.co_state("cand2")["unlinked"]))
gw.open = [o for o in gw.open if o["ref"] != old]          # the old stop's cancel confirmed
paper.close_only_check(gw, "cand2")
check("(L14) the room frees later: no growth ('as far as it fits then, and never later')", gw.modified == [], gw.modified)
notes_now()

# (L14b) the same for a linked exit: out at 30 for its entry's first 30; the entry completes between the orders read and
# the fill sync with an old stop's cancel pending: grown as far as it fits (80) and unlinked, never grown later
gw = fresh(-20)
world(gw).act({"place": [O("BUY", 20, "STP")], "cancel": []})
old = ref_of(gw, "BUY", "STP")
gw.cancel = pending_cancel
world(gw).act({"place": [O("SELL", 80, "LMT", tif="DAY", co_=False), O("BUY", 100, "TRAIL")], "cancel": [old]})
h, ent = first_held(), ref_of(gw, "SELL", "LMT")
gw.fill(ent, 30)
paper.close_only_check(gw, "cand2")
rows = gw.orders()
gw.fill(ent, 50)
paper.sync_fills(gw)
paper.close_only_check(gw, "cand2", rows=rows, synced=True)
check("(L14b) out at 30, then grown to 80 beside the pending 20 and unlinked at that check", sent_at(gw, h) == [30]
      and gw.modified[-1:] == [(h, 80)] and h in paper.co_state("cand2")["unlinked"], (gw.placed, gw.modified))
gw.open = [o for o in gw.open if o["ref"] != old]
paper.close_only_check(gw, "cand2")
check("(L14b) the room frees later: no further growth", gw.modified[-1:] == [(h, 80)] and len(gw.modified) == 1, gw.modified)
notes_now()

# (L15) the open orders not read at act's start (the 'unread' path): held with no E or H, so P is 0
class GWf(GW):
    def orders(self):
        if getattr(self, "fail_orders", 0):
            self.fail_orders -= 1
            raise RuntimeError("504 not connected")
        return super().orders()


gw = fresh(-20)
gw.__class__ = GWf
world(gw).act({"place": [O("BUY", 5, "STP", stop=120.0)], "cancel": []})   # a close_only order, so act reads the orders
gw.fail_orders = 1
world(gw).act({"place": [O("SELL", 80, "LMT", tif="DAY", co_=False), O("BUY", 100, "STP")], "cancel": []})
h = next(r for r, w in paper.co_state("cand2")["held"].items() if w["quantity"] == 100)
check("(L15) unread at the hold: no 'left' or 'had' recorded", "left" not in held_row(h)["held"] and "had" not in held_row(h)["held"],
      held_row(h)["held"])
gw.fill(ref_of(gw, "SELL", "LMT"), 80)
paper.close_only_check(gw, "cand2")
check("(L15) the entry's 80: out at its entry's fills only (P 0), within the room beside the 5 stop", sent_at(gw, h) == [80],
      gw.placed)
notes_now()

# (L16) a held row names only the entries still working: two entries, one cancelled with no fill
gw = fresh(-10)
world(gw).act({"place": [O("SELL", 30, "LMT", tif="DAY", co_=False), O("SELL", 20, "LMT", tif="DAY", co_=False, limit=111.0),
                         O("BUY", 60, "STP")], "cancel": []})
h = first_held()
e1, e2 = [p["ref"] for p in gw.placed if p["order_type"] == "LMT"]
world(gw).act({"place": [], "cancel": [e1]})
paper.close_only_check(gw, "cand2")
w16 = world(gw)
w16.data, w16.dry_run = __import__("types").SimpleNamespace(dates=["20261120"], ix={}), True
row16 = next(o for o in w16.state(1, {}, live={"X": 100.0})["orders"] if o["ref"] == h)
check("(L16) the held row waits for the entry still working only", row16.get("held") and row16.get("after") == [e2], row16)
notes_now()

# (L17) the final review of 25 Nov: a release IBKR answers with no status, at the check where its entries have ended, is
# unlinked ("entries ended"), so it never grows later when the room frees, even if IBKR does have it
gw = fresh(-20)
world(gw).act({"place": [O("BUY", 20, "STP")], "cancel": []})
old = ref_of(gw, "BUY", "STP")
gw.cancel = pending_cancel
world(gw).act({"place": [O("SELL", 80, "MKT", tif="DAY", co_=False), O("BUY", 100, "STP", stop=111.0)], "cancel": [old]})
h = first_held()
gw.fill(ref_of(gw, "SELL", "MKT"), 80)
gw.place_status = None                                   # IBKR's answer has no status; the order is listed working
paper.close_only_check(gw, "cand2")
gw.place_status = "Submitted"
unl = [r.get("why") for r in paper.read(paper.ORDERS) if r.get("unlink") == h]
check("(L17) sent at 80 (refused in the ledger: no status) with its entry ended: unlinked 'entries ended'",
      sent_at(gw, h) == [80] and unl == ["entries ended"], (gw.placed, unl))
gw.open = [o for o in gw.open if o["ref"] != old]
paper.close_only_check(gw, "cand2")
check("(L17) the room frees later: never grown", [m for m in gw.modified if m[0] == h] == [], gw.modified)
notes_now()

# (L18) a held row names an ended entry whose fills count for it: SELL 30 LMT (E1) and SELL 50 LMT (E2), a BUY 80 STP and
# a BUY 80 TRAIL close_only, both held; E1 fills 30: the STP goes out at 30, the TRAIL (no room) waits for E1's 30 and E2
gw = fresh(0)
world(gw).act({"place": [O("SELL", 30, "LMT", tif="DAY", co_=False), O("SELL", 50, "LMT", tif="DAY", co_=False, limit=111.0),
                         O("BUY", 80, "STP"), O("BUY", 80, "TRAIL")], "cancel": []})
e1, e2 = [p["ref"] for p in gw.placed if p["order_type"] == "LMT"]
gw.fill(e1, 30)
paper.close_only_check(gw, "cand2")
trl = next(r for r, w in paper.held_waiting("cand2").items() if w["order_type"] == "TRAIL")
w18 = world(gw)
w18.data, w18.dry_run = __import__("types").SimpleNamespace(dates=["20261120"], ix={}), True
row18 = next(o for o in w18.state(1, {}, live={"X": 100.0})["orders"] if o["ref"] == trl)
check("(L18) the held TRAIL waits for E1 (ended, its 30 count for it) and E2 (working)", row18.get("held")
      and row18.get("after") == [e1, e2], row18)
notes_now()

# (L19) 'grows toward' from the booked fills (the Researcher, 25 Nov): released at 50 while the entry fills 30 more during
# the release, so the exit lags the ledger: the row reads 'grows toward 100' (80 filled + 20 left), not 70
class GWl(GW):
    def place(self, symbol, action, quantity, order_type, limit=None, aux=None, tif="DAY", ref=None, **f):
        out = super().place(symbol, action, quantity, order_type, limit, aux, tif, ref, **f)
        if getattr(self, "lag_entry", None) and action == "SELL":
            self.fill(self.lag_entry, 30)
            self.lag_entry = None
        return out


import live  # noqa: E402
live.log = lambda m: None                                # no line reaches a real log here
gw = fresh(0)
gw.__class__ = GWl
world(gw).act({"place": [O("BUY", 100, "LMT", tif="DAY", co_=False), O("SELL", 100, "STP")], "cancel": []})
h, ent = first_held(), ref_of(gw, "BUY", "LMT")
gw.fill(ent, 50)
gw.lag_entry = ent
snap19 = live.catch_up_sleeve(gw, "cand2", None)
w19 = world(gw)
w19.gw = live.OrdersAt(gw, snap19)
w19.data, w19.dry_run = __import__("types").SimpleNamespace(dates=["20261120"], ix={}), True
row19 = next(o for o in w19.state(1, {}, live={"X": 100.0})["orders"] if o["ref"] == h)
check("(L19) released at 50, the entry at 80 of 100 by the booked fills: 'grows toward 100'", sent_at(gw, h) == [50]
      and pos_of() == 80 and row19.get("grows_to") == 100, (row19, pos_of()))
notes_now()

# (L20) the hold note says so when the hold's reads failed (the Researcher's words, 25 Nov), and only then
NOTE_CLAUSE = " (your orders or fills could not be read, so shares you already held are not counted for it)"
gw = fresh(-20)
gw.__class__ = GWf
world(gw).act({"place": [O("BUY", 5, "STP", stop=120.0)], "cancel": []})
notes_now()
gw.fail_orders = 1
n_fail = world(gw).act({"place": [O("SELL", 80, "LMT", tif="DAY", co_=False), O("BUY", 100, "STP")], "cancel": []})
gw2 = fresh(-20)
n_ok = world(gw2).act({"place": [O("SELL", 80, "LMT", tif="DAY", co_=False), O("BUY", 100, "STP")], "cancel": []})
held_fail = [x for x in n_fail if "(close_only) held as" in x]
held_ok = [x for x in n_ok if "(close_only) held as" in x]
check("(L20) the hold note carries the clause when the reads failed, and not otherwise",
      len(held_fail) == 1 and held_fail[0].endswith(NOTE_CLAUSE) and len(held_ok) == 1 and NOTE_CLAUSE not in held_ok[0],
      (held_fail, held_ok))
notes_now()

# (P3) the entry partly filled before the hold: SELL 80 LMT, 30 filled (-50); then BUY 100 STP close_only: E 50, H 50
gw = fresh(-20)
world(gw).act({"place": [O("SELL", 80, "LMT", tif="DAY", co_=False)], "cancel": []})
ent = ref_of(gw, "SELL", "LMT")
gw.fill(ent, 30)
paper.sync_fills(gw)
world(gw).act({"place": [O("BUY", 100, "STP")], "cancel": []})
h = first_held()
check("(P3) E is the 50 the entry had left at the hold, H the 50 held; nothing out while the entry works",
      held_row(h)["held"].get("left") == 50 and held_row(h)["held"].get("had") == 50 and sent_at(gw, h) == [], held_row(h)["held"])
gw.fill(ent, 50)
paper.close_only_check(gw, "cand2")
check("(P3) the entry's last 50: out at 100 on -100", sent_at(gw, h) == [100], gw.placed)
notes_now()

# (P6) the other side at the hold: +20 long, SELL 80 LMT and BUY 100 STP close_only: H 0, so only its entry's fills
gw = fresh(20)
world(gw).act({"place": [O("SELL", 80, "LMT", tif="DAY", co_=False), O("BUY", 100, "STP")], "cancel": []})
h, ent = held_ref(), ref_of(gw, "SELL", "LMT")
check("(P6) on +20: H 0", held_row(h)["held"].get("had") == 0 and sent_at(gw, h) == [])
gw.fill(ent, 30)                                          # +20 - 30 = -10
paper.close_only_check(gw, "cand2")
gw.fill(ent, 50)                                          # -60
paper.close_only_check(gw, "cand2")
check("(P6) 30 filled (-10): out at 10; the last 50 (-60): grows to 60", sent_at(gw, h) == [10] and gw.modified[-1:] == [(h, 60)],
      (gw.placed, gw.modified))
notes_now()

# (P7) two entries: -10, SELL 30 LMT and SELL 20 LMT, BUY 60 STP close_only: E 50, H 10; P only once both have ended
gw = fresh(-10)
world(gw).act({"place": [O("SELL", 30, "LMT", tif="DAY", co_=False), O("SELL", 20, "LMT", tif="DAY", co_=False, limit=111.0),
                         O("BUY", 60, "STP")], "cancel": []})
h = first_held()
e1, e2 = [p["ref"] for p in gw.placed if p["order_type"] == "LMT"]
gw.fill(e1, 30)
paper.close_only_check(gw, "cand2")
first = sent_at(gw, h)
gw.fill(e2, 20)
paper.close_only_check(gw, "cand2")
check("(P7) the first entry's 30 (the second still works): out at 30; the second's 20: grows to 60 (Fe 50 + P 10)",
      first == [30] and gw.modified[-1:] == [(h, 60)] and pos_of() == -60, (first, gw.modified))
notes_now()

# (P8) an exit sized to its entry (q <= E): P 0, as before
gw = fresh(-20)
world(gw).act({"place": [O("SELL", 80, "LMT", tif="DAY", co_=False), O("BUY", 80, "STP")], "cancel": []})
h, ent = held_ref(), ref_of(gw, "SELL", "LMT")
gw.fill(ent, 80)
paper.close_only_check(gw, "cand2")
check("(P8) BUY 80 for SELL 80 on -20: out at 80, its entry's fills", sent_at(gw, h) == [80], gw.placed)
notes_now()

# (P9) a row held before 25 Nov (no "left" or "had"): the rule it was placed under, never for other shares
gw = fresh(-20)
world(gw).act({"place": [O("SELL", 80, "LMT", tif="DAY", co_=False)], "cancel": []})
ent = ref_of(gw, "SELL", "LMT")
r9 = paper.place(gw, "cand2", "X", "BUY", 100, "STP", aux=110.0, tif="GTC", close_only=True, held={"after": {ent: 0.0}})
gw.fill(ent, 80)
paper.close_only_check(gw, "cand2")
check("(P9) no E or H: out at 80, its entry's fills only", sent_at(gw, r9["ref"]) == [80], gw.placed)
notes_now()

# (P12) the review's S6: -20 with an older SELL 10 LMT (A); one submit: BUY 100 STP close_only, then SELL 70 LMT (B), listed
# after it (not its entry): E 10, H 20; B's fills never count for it
gw = fresh(-20)
world(gw).act({"place": [O("SELL", 10, "LMT", tif="DAY", co_=False)], "cancel": []})
a_ = ref_of(gw, "SELL", "LMT")
world(gw).act({"place": [O("BUY", 100, "STP"), O("SELL", 70, "LMT", tif="DAY", co_=False, limit=111.0)], "cancel": []})
h = first_held()
b_ = next(p["ref"] for p in gw.placed if p["order_type"] == "LMT" and p["quantity"] == 70)
gw.fill(b_, 70)
paper.close_only_check(gw, "cand2")
first = sent_at(gw, h)
gw.fill(a_, 10)
paper.close_only_check(gw, "cand2")
check("(P12) held on A only: B's 70 give nothing; A's 10 (ended): out at 30 (Fe 10 + P 20)",
      list(held_row(h)["held"]["after"]) == [a_] and first == [] and sent_at(gw, h) == [30], (first, gw.placed))
notes_now()

# (P13) the review's S4: an old BUY 20 stop on -20; SELL 80 LMT and BUY 100 STP; the sleeve cancels the old stop with its
# entry at 0 fills: nothing goes out until the entry fills (HELD_LINE stays true)
gw = fresh(-20)
world(gw).act({"place": [O("BUY", 20, "STP")], "cancel": []})
old = ref_of(gw, "BUY", "STP")
world(gw).act({"place": [O("SELL", 80, "LMT", tif="DAY", co_=False), O("BUY", 100, "STP", stop=111.0)], "cancel": []})
h, ent = held_ref(), ref_of(gw, "SELL", "LMT")
world(gw).act({"place": [], "cancel": [old]})
paper.close_only_check(gw, "cand2")
check("(P13) the old stop cancelled, the entry at 0 fills: still held, nothing out", sent_at(gw, h) == [] and h in paper.held_waiting("cand2"))
gw.fill(ent, 80)
paper.close_only_check(gw, "cand2")
check("(P13) the entry's 80: out at 100 (the old stop is gone)", sent_at(gw, h) == [100], gw.placed)
notes_now()

# (P15) the cap at work: flat with a resting SELL 50 LMT (A); one submit holds a BUY 100 STP close_only on A (H 0), then
# sends a plain SELL 50 MKT (B) that fills at once; A fills later: 50, A's fills only
gw = fresh(0)
world(gw).act({"place": [O("SELL", 50, "LMT", tif="DAY", co_=False)], "cancel": []})
gw.fill_mkt = True
world(gw).act({"place": [O("BUY", 100, "STP"), O("SELL", 50, "MKT", tif="DAY", co_=False)], "cancel": []})
gw.fill_mkt = False
h = first_held()
gw.fill(ref_of(gw, "SELL", "LMT"), 50)
paper.close_only_check(gw, "cand2")
check("(P15) H 0 at the hold: when A ends, out at 50 (not 100) on -100", held_row(h)["held"].get("had") == 0
      and sent_at(gw, h) == [50] and pos_of() == -100, (gw.placed, held_row(h)["held"]))
notes_now()

# (M) the check's own cancels carry by: "close_only" in their ledger row; the sleeve's do not
gw = fresh(-20)
world(gw).act({"place": [O("BUY", 20, "STP")], "cancel": []})
stp = ref_of(gw, "BUY", "STP")
gw.execs.append(ex("X", "BOT", 20, "tt-cand2-0009"))      # a plain BUY 20 closed the 20
paper.close_only_check(gw, "cand2")
world(gw).act({"place": [O("BUY", 5, "LMT", co_=False)], "cancel": []})
mine_lmt = ref_of(gw, "BUY", "LMT")
world(gw).act({"place": [], "cancel": [mine_lmt]})
rows_c = [r for r in paper.read(paper.ORDERS) if r.get("cancel")]
check("(M) the check's cancel row is marked by close_only; the sleeve's own cancel row is not",
      any(r["cancel"] == stp and r.get("by") == "close_only" for r in rows_c)
      and any(r["cancel"] == mine_lmt and "by" not in r for r in rows_c), rows_c)
notes_now()

# (R1) the race of 24 Nov (MPWR 09:38, ADI and LRCX 12:40): the exit fills between the orders read and the fill sync
gw = fresh(-2)
world(gw).act({"place": [O("BUY", 2, "STP")], "cancel": []})
stp = ref_of(gw, "BUY", "STP")
notes_now()
rows = gw.orders()                                       # still lists the stop as working
gw.fill(stp, 2)
paper.sync_fills(gw)                                     # its fill booked: position 0
paper.close_only_check(gw, "cand2", rows=rows, synced=True)
n = " ".join(notes_now())
check("(R1) a stop filled in full by the fills just booked: no cancel sent, no note", gw.cancelled == [] and not n, (gw.cancelled, n))
gw = fresh(-5)
world(gw).act({"place": [O("BUY", 5, "STP")], "cancel": []})
stp = ref_of(gw, "BUY", "STP")
notes_now()
rows = gw.orders()
gw.fill(stp, 2)
paper.sync_fills(gw)
paper.close_only_check(gw, "cand2", rows=rows, synced=True)
check("(R1) partly filled in between (-3, 3 left): nothing changed", gw.cancelled == [] and gw.modified == [] and not notes_now())
gw = fresh(-2)
world(gw).act({"place": [O("BUY", 2, "STP")], "cancel": []})
stp = ref_of(gw, "BUY", "STP")
notes_now()
gw.execs.append(ex("X", "BOT", 2, "tt-cand2-0009"))      # another order closed the position; the stop still working
paper.close_only_check(gw, "cand2")
check("(R1) a stop not filled with the position closed by another order: still cancelled, as before",
      gw.cancelled == [stp] and "so a cancel was sent for" in " ".join(notes_now()))

# (C) live.co_before_state: the check inside each sleeve's catch-up, before its state
import live  # noqa: E402
logged = []
live.log = logged.append
gw = fresh(0)
world(gw).act({"place": [O("BUY", 50, "LMT", tif="DAY", co_=False), O("SELL", 50, "STP")], "cancel": []})
h = held_ref()
notes_now()
gw.fill(ref_of(gw, "BUY", "LMT"), 50)                     # fills between looks; no check has run since
rows = gw.orders()
paper.sync_fills(gw)
out = live.co_before_state(gw, "cand2", rows, dry_run=True)
check("(C) a dry run: no check, the rows returned as read, the exit still held", out is rows and sent_at(gw, h) == []
      and h in paper.held_waiting("cand2"))
out = live.co_before_state(gw, "cand2", rows)
st_rows = live.settle_orders(out, "cand2")
shown = [o["ref"] for o in paper.open_orders(live.OrdersAt(gw, st_rows), "cand2")]
check("(C) the check releases the exit at 50 before the state, and the state's orders (read again after it) list it at "
      "the broker, not held", sent_at(gw, h) == [50] and out is not rows and h in shown and h not in paper.held_waiting("cand2"),
      (sent_at(gw, h), shown))
check("(C) its note is in the state's notes, and its time is logged", any("went to the broker at 50" in x for x in notes_now())
      and any("close_only check before the state: 1 change(s)" in m for m in logged), logged)
logged.clear()
out2 = live.co_before_state(gw, "cand2", out)
check("(C) nothing to do: the rows as read, no second read, nothing logged", out2 is out and not logged, logged)
co.V2 = False
out3 = live.co_before_state(gw, "cand2", rows)
co.V2 = True
check("(C) switch off: nothing", out3 is rows)

# (C2) one pass (the third review of 25 Nov): Y's entry fills while the check waits for IBKR to take X's release; the
# re-read books it, the state shows Y's exit still held beside it, and the next check releases it
class GWy(GW):
    def place(self, symbol, action, quantity, order_type, limit=None, aux=None, tif="DAY", ref=None, **f):
        out = super().place(symbol, action, quantity, order_type, limit, aux, tif, ref, **f)
        if getattr(self, "y_entry", None) and symbol == "X":
            self.fill(self.y_entry, 30)
            self.y_entry = None
        return out


gw = fresh(0)
gw.__class__ = GWy
world(gw).act({"place": [O("BUY", 50, "LMT", tif="DAY", co_=False), O("SELL", 50, "STP")], "cancel": []})
world(gw).act({"place": [O("BUY", 30, "LMT", sym="Y", tif="DAY", co_=False), O("SELL", 30, "STP", sym="Y")], "cancel": []})
hx, hy = sorted(paper.held_waiting("cand2"), key=lambda r: paper.co_state("cand2")["key"][r])
gw.y_entry = ref_of(gw, "BUY", "LMT", sym="Y")
gw.fill(ref_of(gw, "BUY", "LMT"), 50)                     # X's entry, between looks
notes_now()
logged.clear()
rows = gw.orders()
paper.sync_fills(gw)
out = live.co_before_state(gw, "cand2", rows)
check("(C2) one pass: X's exit released; Y's (its entry filled during X's release) still held for this state",
      sent_at(gw, hx) == [50] and sent_at(gw, hy) == [] and hy in paper.held_waiting("cand2")
      and any("close_only check before the state: 1 change(s)" in m for m in logged), (gw.placed, logged))
paper.close_only_check(gw, "cand2")
check("(C2) the next check releases Y's exit at 30", sent_at(gw, hy) == [30], gw.placed)
notes_now()

# (C3) live.catch_up_sleeve itself (the review of 25 Nov: a mutant that dropped co_before_state's result passed every test):
# orders read, fills synced, the check, a new read, settled; and a held name with no mark is quoted
gw = fresh(0)
world(gw).act({"place": [O("BUY", 50, "LMT", tif="DAY", co_=False), O("SELL", 50, "STP")], "cancel": []})
h = held_ref()
gw.fill(ref_of(gw, "BUY", "LMT"), 50)
notes_now()
calls = []
for m in ("orders", "executions", "place", "quotes"):
    f0 = getattr(gw, m)
    setattr(gw, m, (lambda f0, m: lambda *a, **k: (calls.append(m), f0(*a, **k))[1])(f0, m))
marks = {}
snap = live.catch_up_sleeve(gw, "cand2", marks)
shown = [o["ref"] for o in paper.open_orders(live.OrdersAt(gw, snap), "cand2")]
check("(C3) catch_up_sleeve: the exit released before the state and listed as working; X quoted for its mark",
      h in shown and sent_at(gw, h) == [50] and h not in paper.held_waiting("cand2") and marks.get("X") is not None,
      (shown, marks))
check("(C3) the order of calls: orders, then fills, then the release, then orders and fills again, then the quote",
      calls[:5] == ["orders", "executions", "place", "orders", "executions"] and calls[-1] == "quotes", calls)
gw2 = fresh(0)
world(gw2).act({"place": [O("BUY", 50, "LMT", tif="DAY", co_=False), O("SELL", 50, "STP")], "cancel": []})
h2 = held_ref()
gw2.fill(ref_of(gw2, "BUY", "LMT"), 50)
snap = live.catch_up_sleeve(gw2, "cand2", None, dry_run=True)
check("(C3) a dry run: nothing released; the exit stays held", sent_at(gw2, h2) == [] and h2 in paper.held_waiting("cand2"))
notes_now()

# (C4) the second review of 25 Nov: a release IBKR has not acknowledged ("Unknown") is not listed by the second pass's read;
# it stays linked (not "ended at the broker"), and grows when its entry fills after IBKR lists it
class GWu(GW):
    def place(self, symbol, action, quantity, order_type, limit=None, aux=None, tif="DAY", ref=None, **f):
        out = super().place(symbol, action, quantity, order_type, limit, aux, tif, ref, **f)
        if getattr(self, "unack", None) and action == self.unack:
            self.hidden = [o for o in self.open if o["ref"] == ref]
            self.open = [o for o in self.open if o["ref"] != ref]
            return {"status": "Unknown", "messages": ["not acknowledged within the wait"]}
        return out


gw = fresh(0)
gw.__class__ = GWu
world(gw).act({"place": [O("BUY", 100, "LMT", tif="DAY", co_=False), O("SELL", 100, "STP")], "cancel": []})
h, ent = held_ref(), ref_of(gw, "BUY", "LMT")
gw.fill(ent, 40)
gw.unack = "SELL"
notes_now()
logged.clear()
rows = gw.orders()
paper.sync_fills(gw)
out = live.co_before_state(gw, "cand2", rows)
st4 = paper.co_state("cand2")
check("(C4) released at 40 'Unknown' and not listed at the read after it: kept linked, not 'ended at the broker'",
      sent_at(gw, h) == [40] and h not in st4["unlinked"] and h in st4["released"], (gw.placed, st4["unlinked"]))
gw.unack = None
gw.open += gw.hidden                                       # IBKR lists it
gw.fill(ent, 60)
paper.close_only_check(gw, "cand2")
check("(C4) IBKR lists it and the entry's last 60 fill: it grows to 100", gw.modified[-1:] == [(h, 100)], gw.modified)
notes_now()
gw = fresh(0)                                              # never listed: unlinked once UNACKED_S has passed
gw.__class__ = GWu
world(gw).act({"place": [O("BUY", 100, "LMT", tif="DAY", co_=False), O("SELL", 100, "STP")], "cancel": []})
h, ent = held_ref(), ref_of(gw, "BUY", "LMT")
gw.fill(ent, 40)
gw.unack = "SELL"
paper.close_only_check(gw, "cand2")
paper.close_only_check(gw, "cand2")
kept = h not in paper.co_state("cand2")["unlinked"]
CLOCK[0] = CLOCK[0] + timedelta(seconds=paper.UNACKED_S + 1)
paper.close_only_check(gw, "cand2")
check("(C4) never listed: kept linked within UNACKED_S, unlinked after", kept and h in paper.co_state("cand2")["unlinked"])
notes_now()

# (C6) a modify that sends nothing ("unsent") starts no second pass: the reduce is tried once in the catch-up, not twice
gw = fresh(100)
world(gw).act({"place": [O("SELL", 100, "STP")], "cancel": []})
stp = ref_of(gw, "SELL", "STP")
gw.execs.append(ex("X", "SLD", 40, "tt-cand2-0009"))     # another order sold 40: the stop is 40 over the position
gw.modify_raise = GatewayError("echo and status disagree", "size_mismatch")
rows = gw.orders()
paper.sync_fills(gw)
logged.clear()
live.co_before_state(gw, "cand2", rows)
gw.modify_raise = None
check("(C6) an unsent reduce: one modify in the catch-up, and no new read (nothing changed at the broker)",
      [m for m in gw.modified if m[0] == stp] == [(stp, 60)] and not any("before the state" in m for m in logged), (gw.modified, logged))
notes_now()

# (C7) a reduce met by an order that had ended at the broker ("gone": IBKR reports it filled during the modify): the
# orders are read again for the state
gw = fresh(100)
world(gw).act({"place": [O("SELL", 100, "STP")], "cancel": []})
stp = ref_of(gw, "SELL", "STP")
gw.execs.append(ex("X", "SLD", 40, "tt-cand2-0009"))
gw.modify_result = {"modified": None, "status": "Filled", "messages": ["Order Filled during the modify; check orders()"]}
rows = gw.orders()
paper.sync_fills(gw)
out = live.co_before_state(gw, "cand2", rows)
gw.modify_result = None
check("(C7) a 'gone' reduce: the orders read again for the state", out is not rows, gw.modified)
notes_now()
# (W) the words: the Researcher's final sentence and HELD_LINE (models-w3 d13f1c02, 25 Nov), the old clauses gone
import tools  # noqa: E402
NEW = ("it goes out for that order's fills since the hold (all of them if in the same submit) and, while it is at the broker, "
       "grows with them up to its size, as far as it fits. When that order has ended, the part of its size above what that "
       "order had left to fill at the hold goes out too, for other shares you held there at the hold, as far as it fits "
       "then, and never later.")
HELD_PIN = ("(held: a close_only order that is not at the broker, so it cannot fill; it goes out when the order it waits for "
            "fills, for those fills; it is cancelled if that order ends with nothing for it to close, and a DAY one at the "
            "close.)")
check("(W) CLOSE_DESC_V2 carries the Researcher's final sentence once, between 'At the next check after that order fills, ' "
      "and ' At each look'; no 'never for other shares' or 'next check' half; HELD_LINE without 'only'",
      tools.CLOSE_DESC_V2.count(NEW) == 1 and "never for other shares" not in tools.CLOSE_DESC_V2
      and "that part goes out at the next check" not in tools.CLOSE_DESC_V2 and tools.HELD_LINE == HELD_PIN
      and "At the next check after that order fills, " + NEW + " At each look" in tools.CLOSE_DESC_V2)

# (u) V2 off: the rule of 17-19 Nov (the 15 LMT beside a full stop is not sent), held exits never made
co.V2 = False
gw = fresh(-45)
world(gw).act({"place": [O("BUY", 45, "STP")], "cancel": []})
notes = world(gw).act({"place": [O("BUY", 15, "LMT", tif="DAY"), O("BUY", 10, "MOO", co_=False), O("SELL", 10, "STP")],
                       "cancel": []})
check("(u) switch off: v1 refuses the 15 LMT ('already cover 45') and holds nothing",
      any("your close_only orders there already cover 45, so it would close more than you hold" in x for x in notes)
      and not paper.co_state("cand2")["held"], notes)
co.V2 = True

# ================================================================ the words, with the switch on and off (a fresh process)
probe = ("import sys; sys.path.insert(0, sys.argv[1]); import close_only; close_only.V2 = sys.argv[2] == '1'; "
         "import tools, trader, mcp_server, json; "
         "print(json.dumps([tools.CLOSE_DESC == tools.CLOSE_DESC_V2, tools.CLOSE_DESC in trader.TOOL_LIVE['description'], "
         "tools.CLOSE_DESC in mcp_server.SUBMIT_DESC['B']]))")
d0 = Path(tempfile.mkdtemp())
res = []
for flag in ("1", "0"):
    p = subprocess.run([sys.executable, "-c", probe, str(ROOT), flag], capture_output=True, text=True, encoding="utf-8", cwd=d0,
                       env=dict(os.environ, TRADER_SUBMIT_FILE=str(d0 / "s.json"), TRADER_LOG_FILE=str(d0 / "l.jsonl")))
    res.append(json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else p.stderr[-300:])
check("words: switch on, text B's form and Fable's submit carry CLOSE_DESC_V2; off, CLOSE_DESC_V1 as on 17-19 Nov",
      res == [[True, True, True], [False, True, True]], res)

print(f"{sum(ok)} of {len(ok)} checks passed" if not all(ok) else "ALL OK")
sys.exit(0 if all(ok) else 1)
