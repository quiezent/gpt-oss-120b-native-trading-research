"""dividends.book on a fake gateway and temporary ledgers. Offline.

    python dividends_test.py
"""
import atexit
import json
import shutil
import tempfile
from pathlib import Path

import dividends
import paper


class FakeGW:
    """X: 0.50 ex 20261021 on a 50.00 close; Y: a 2:1 split-sized step on 20261021 (not a dividend)."""
    RAW = {"X": {"20261019": 50.0, "20261020": 50.0, "20261021": 49.5, "20261022": 49.6},
           "Y": {"20261019": 100.0, "20261020": 100.0, "20261021": 50.0, "20261022": 50.0}}

    def bars(self, sym, duration, bar_size, what, rth):
        raw = self.RAW[sym]
        f = {"X": lambda d: (1 - 0.5 / 50.0) if d < "20261021" else 1.0, "Y": lambda d: 0.5 if d < "20261021" else 1.0}[sym]
        return [{"date": d, "close": c * (f(d) if what == "ADJUSTED_LAST" else 1.0)} for d, c in raw.items()]


def fill(sleeve, sym, side, n, day, i):
    return {"exec_id": f"e{i}", "sleeve": sleeve, "symbol": sym, "side": side, "shares": n, "price": 50.0,
            "commission": 1.0, "time": f"{day} 10:00:00 US/Eastern"}


def check(name, ok, got=""):
    print(("ok   " if ok else "FAIL ") + name + (f": {got}" if got else ""))
    return ok


with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    paper.FILLS, paper.SLEEVES, paper.DIVIDENDS, paper.NOTES = d / "f.jsonl", d / "s.json", d / "dv.jsonl", d / "n.json"
    paper.SPLITS, dividends.LOOK_DIR = d / "sp.jsonl", d / "live"
    paper.SLEEVES.write_text(json.dumps({s: {"start_cash": 30000.0, "start": "2026-10-19T13:00:00+00:00", "model": "m"}
                                         for s in ("long", "short", "late", "gone")}), encoding="utf-8")
    rows = [fill("long", "X", "BOT", 100, "20261019", 1), fill("short", "X", "SLD", 10, "20261020", 2),
            fill("late", "X", "BOT", 100, "20261021", 3),                                    # bought on the ex-date
            fill("gone", "X", "BOT", 100, "20261019", 4), fill("gone", "X", "SLD", 100, "20261019", 5),   # closed before
            fill("long", "Y", "BOT", 10, "20261019", 6)]
    for r in rows:
        paper.append(paper.FILLS, r)
    log = []
    got = dividends.book(FakeGW(), today="20261022", log=log.append)
    by = {(r["sleeve"], r["symbol"]): r["amount"] for r in got}
    b = paper.book("long")
    notes = json.loads(paper.NOTES.read_text(encoding="utf-8"))
    ok = [check("long credited 100 x 0.50", by.get(("long", "X")) == 50.0, by),
          check("short charged 10 x 0.50", by.get(("short", "X")) == -5.0, by),
          check("bought on the ex-date: nothing", ("late", "X") not in by),
          check("closed before the ex-date: nothing", ("gone", "X") not in by),
          check("split-sized step not booked, and logged", ("long", "Y") not in by and any("not booked" in l for l in log), log),
          check("book() carries it", b["dividends"] == 50.0 and abs(b["cash"] - (30000 - 100 * 50 - 1 - 10 * 50 - 1 + 50)) < 1e-9
                and abs(b["realized"] - (-2 + 50)) < 1e-9, {k: b[k] for k in ("cash", "realized", "dividends")}),
          check("the model's note", "credited $50.00" in " ".join(notes.get("long", [])) and
                "charged $5.00" in " ".join(notes.get("short", [])), notes),
          check("a second run books nothing", dividends.book(FakeGW(), today="20261022", log=log.append) == []),
          check("an ex-date after today waits", dividends.book(FakeGW(), today="20261020", log=log.append) == [])]


# ---------------------------------------------------------------- splits (dividends.book_splits)
class SplitGW:
    """The account's positions, snapshot quotes and daily bars of a few symbols; `fail` names the calls that raise.
    bars: {symbol: {what: {date: close}}}; IBKR's TRADES and ADJUSTED_LAST bars are both split-adjusted."""

    def __init__(self, positions, quotes=None, bars=None, fail=()):
        self.pos, self.q, self.b, self.fail = positions, quotes or {}, bars or {}, set(fail)

    def positions(self):
        if "positions" in self.fail:
            raise ConnectionError("positions timed out")
        return [{"symbol": s, "quantity": q, "avg_cost": 1.0} for s, q in self.pos.items()]

    def quotes(self, syms):
        if "quotes" in self.fail:
            raise ConnectionError("quote timed out")
        return [self.q.get(s, {"symbol": s, "error": "no data"}) for s in syms]

    def bars(self, sym, duration, bar_size, what, rth):
        if "bars" in self.fail:
            raise ConnectionError("bars timed out")
        return [{"date": d, "close": c} for d, c in sorted(self.b.get(sym, {}).get(what, {}).items())]


def split_case(sleeves, rows, looks=None, splits=()):
    """Fresh ledgers in a new temp folder: sleeves {name: model}, fills, look prices {(day YYYYMMDD, sleeve): {sym: price}}."""
    d = Path(tempfile.mkdtemp())
    atexit.register(shutil.rmtree, d, True)
    paper.FILLS, paper.SLEEVES, paper.DIVIDENDS, paper.NOTES = d / "f.jsonl", d / "s.json", d / "dv.jsonl", d / "n.json"
    paper.SPLITS, dividends.LOOK_DIR = d / "sp.jsonl", d / "live"
    dividends.SUSPECTED.clear()
    paper.SLEEVES.write_text(json.dumps({s: {"start_cash": 30000.0, "start": "2026-10-19T13:00:00+00:00", "model": m}
                                         for s, m in sleeves.items()}), encoding="utf-8")
    for r in rows:
        paper.append(paper.FILLS, r)
    for r in splits:
        paper.append(paper.SPLITS, r)
    for (day, s), px in (looks or {}).items():
        p = d / "live" / f"{day[:4]}-{day[4:6]}-{day[6:]}" / f"{s}-1530.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"state": {"positions": [{"symbol": k, "quantity": 1, "avg_cost": 1.0, "close": v}
                                                         for k, v in px.items()]}}), encoding="utf-8")
    return d


def sfill(sleeve, sym, side, n, day, i, px=150.0):
    return {"exec_id": f"s{i}", "sleeve": sleeve, "symbol": sym, "side": side, "shares": n, "price": px,
            "commission": 1.0, "time": f"{day} 10:00:00 US/Eastern"}


def qty(sleeve, sym):
    return paper.book(sleeve)["positions"].get(sym, {}).get("quantity")


def notes_of():
    return json.loads(paper.NOTES.read_text(encoding="utf-8")) if paper.NOTES.exists() else {}


def loud(log):
    return [l for l in log if l.startswith("SPLIT SUSPECTED, NOT BOOKED")]


# SOXX at 07:30 on Thu 5 Nov: the books hold 20 and 15 pre-split shares, the account 105; IBKR's bars are already re-based
SOXX_BARS = {"TRADES": {"20261103": 155.0, "20261104": 150.0}, "ADJUSTED_LAST": {"20261103": 155.0, "20261104": 150.0}}
SOXX_ROWS = [sfill("a", "SOXX", "BOT", 20, "20261103", 1, 460.0), sfill("b", "SOXX", "BOT", 15, "20261104", 2, 452.0)]
SOXX_LOOKS = {("20261104", "a"): {"SOXX": 450.4}, ("20261104", "b"): {"SOXX": 450.4}}
split_case({"a": "m", "b": "m"}, SOXX_ROWS, SOXX_LOOKS)
gw = SplitGW({"SOXX": 105}, {"SOXX": {"symbol": "SOXX", "last": 143.1}}, {"SOXX": SOXX_BARS})
log = []
got = dividends.book_splits(gw, today="20261105", log=log.append)
n = notes_of()
ok += [check("SOXX 3-for-1 booked with ex_date that day", [(r["symbol"], r["ex_date"], r["ratio"]) for r in got]
             == [("SOXX", "20261105", 3)], got),
       check("books show 60 and 45 through paper.book", (qty("a", "SOXX"), qty("b", "SOXX")) == (60, 45)),
       check("average cost divided by 3", abs(paper.book("a")["positions"]["SOXX"]["avg_cost"] - 460.0 / 3) < 1e-9),
       check("positions_at the next day shows 60", paper.positions_at("a", "20261106") == {"SOXX": 60}),
       check("each holder gets the note", n.get("a") == ["SOXX split 3-for-1 effective Thu 5 Nov: your +20 shares are now +60."]
             and n.get("b") == ["SOXX split 3-for-1 effective Thu 5 Nov: your +15 shares are now +45."], n),
       check("SPLIT BOOKED logged with both signals", any(l.startswith("SPLIT BOOKED: SOXX 3-for-1 ex 20261105") and "account +105"
                                                         and "look price 450.40" in l for l in log), log),
       check("a second call books nothing", dividends.book_splits(gw, today="20261105", log=log.append) == []
             and len(paper.read(paper.SPLITS)) == 1 and not loud(log), log),
       check("the next day books nothing", dividends.book_splits(gw, today="20261106", log=log.append) == [] and not loud(log))]

# the same split missed at 07:30, seen at the 16:15 record: sleeve a sold 30 new shares today, the account holds 75; IBKR's
# bars have today's close, and the quote's close field is the pre-split close
split_case({"a": "m", "b": "m"}, SOXX_ROWS + [sfill("a", "SOXX", "SLD", 30, "20261105", 3, 135.0)], SOXX_LOOKS)
bars = {w: {**v, "20261105": 131.9} for w, v in SOXX_BARS.items()}
log = []
got = dividends.book_splits(SplitGW({"SOXX": 75}, {"SOXX": {"symbol": "SOXX", "last": 131.9, "close": 450.41}},
                                    {"SOXX": bars}), today="20261105", log=log.append)
ok += [check("16:15 with today's fills: booked, books 30 and 45", len(got) == 1 and (qty("a", "SOXX"), qty("b", "SOXX")) == (30, 45),
             log)]

# a last equal to the quote's (pre-split) close is no trade since: the bid/ask is the price now
split_case({"a": "m", "b": "m"}, SOXX_ROWS, SOXX_LOOKS)
got = dividends.book_splits(SplitGW({"SOXX": 105}, {"SOXX": {"symbol": "SOXX", "last": 450.41, "close": 450.41, "bid": 143.0,
                                                             "ask": 143.4}}, {"SOXX": SOXX_BARS}), today="20261105", log=[].append)
ok += [check("a stale last: the bid/ask decides, booked", [r["split"] for r in got] == ["3-for-1"], got)]

# the real runs/splits.jsonl (SOXX ex 5 Nov booked by hand) is never booked again
split_case({"a": "m", "b": "m"}, SOXX_ROWS, SOXX_LOOKS, splits=paper.read(Path(__file__).resolve().parent / "runs" / "splits.jsonl"))
log = []
ok += [check("SOXX of runs/splits.jsonl not booked twice", dividends.book_splits(gw, today="20261105", log=log.append) == []
             and len(paper.read(paper.SPLITS)) == 1 and not loud(log), log)]

# reverse 1-for-10 with a remainder: books 25 and 30 (55), the account 5 (5.5 rounded down)
split_case({"a": "m", "b": "m"}, [sfill("a", "R", "BOT", 25, "20261103", 1, 10.0), sfill("b", "R", "BOT", 30, "20261103", 2, 10.0)],
           {("20261104", "a"): {"R": 9.8}})
log = []
cash0 = paper.book("a")["cash"]
got = dividends.book_splits(SplitGW({"R": 5}, {"R": {"symbol": "R", "bid": 97.0, "ask": 99.0}}, {"R": {"TRADES": {"20261104": 98.0}}}),
                            today="20261105", log=log.append)
b = paper.book("a")
n = notes_of()
ok += [check("1-for-10 booked", [(r["ratio"], r["split"], r.get("cash_price")) for r in got] == [(0.1, "1-for-10", 98.0)], got),
       check("1-for-10: whole shares 2 and 3, average cost 100", (qty("a", "R"), qty("b", "R")) == (2, 3)
             and abs(b["positions"]["R"]["avg_cost"] - 100.0) < 1e-9, b["positions"]),
       check("1-for-10: a's half share paid in cash at 98", abs(b["cash"] - cash0 - 49.0) < 1e-9
             and abs(b["realized"] - (-1 + 0.5 * (98.0 - 100.0))) < 1e-9, (b["cash"] - cash0, b["realized"])),
       check("1-for-10: the remainder is in the log", any("SPLIT BOOKED: R 1-for-10" in l and "+0.5 share left over" in l for l in log), log),
       check("1-for-10: the notes", n.get("a") == ["R split 1-for-10 effective Thu 5 Nov: your +25 shares are now +2. The +0.5 share "
                                                   "left over was settled in cash at $98.00 a share (+49.00 to your cash)."]
             and n.get("b") == ["R split 1-for-10 effective Thu 5 Nov: your +30 shares are now +3."], n),
       check("1-for-10: positions_at the next day", paper.positions_at("a", "20261106") == {"R": 2})]

# 3-for-2: a short of 15 becomes 22 and a half; the account -22
split_case({"a": "m"}, [sfill("a", "T", "SLD", 15, "20261103", 1, 60.0)], {("20261104", "a"): {"T": 61.0}})
log = []
cash0 = paper.book("a")["cash"]
got = dividends.book_splits(SplitGW({"T": -22}, {"T": {"symbol": "T", "last": 40.5}}, {"T": {"TRADES": {"20261104": 40.7}}}),
                            today="20261105", log=log.append)
b = paper.book("a")
ok += [check("3-for-2 booked on a short", [r["split"] for r in got] == ["3-for-2"] and qty("a", "T") == -22, (got, log)),
       check("3-for-2: the short half share charged in cash", abs(b["cash"] - cash0 + 0.5 * 40.5) < 1e-9, b["cash"] - cash0)]

# quantity says 3, price says 2 or 1: not booked, one loud line a day
for px, name in ((225.2, "2"), (449.0, "1")):
    split_case({"a": "m", "b": "m"}, SOXX_ROWS, SOXX_LOOKS)
    g = SplitGW({"SOXX": 105}, {"SOXX": {"symbol": "SOXX", "last": px}}, {"SOXX": {"TRADES": {"20261104": 450.4}}})
    log = []
    r1 = dividends.book_splits(g, today="20261105", log=log.append)
    r2 = dividends.book_splits(g, today="20261105", log=log.append)
    ok += [check(f"quantity 3, price {name}: nothing booked, the loud line once", r1 == r2 == [] and not paper.read(paper.SPLITS)
                 and len(loud(log)) == 1 and (qty("a", "SOXX"), qty("b", "SOXX")) == (20, 15) and not notes_of(), log)]
dividends.book_splits(g, today="20261106", log=log.append)
ok += [check("the loud line comes again on another day", len(loud(log)) == 2, log)]

# price says 3 but the quantities agree: nothing (the dividends path's step log stays)
split_case({"a": "m"}, [sfill("a", "Y", "BOT", 10, "20261019", 1)])


class StepGW(SplitGW, FakeGW):
    bars = FakeGW.bars


log = []
got = dividends.book(StepGW({"Y": 10}, {"Y": {"symbol": "Y", "last": 50.0}}), today="20261022", log=log.append)
ok += [check("price 2, quantities equal: no split, the step log stays", not paper.read(paper.SPLITS) and not loud(log)
             and any("not booked (a split or bad data?)" in l for l in log), log)]

# a plain mismatch (account 100, books 60, price unchanged) and a name nobody holds: nothing, nothing loud
split_case({"a": "m"}, [sfill("a", "M", "BOT", 60, "20261103", 1, 50.0)], {("20261104", "a"): {"M": 50.0}})
log = []
got = dividends.book_splits(SplitGW({"M": 100, "ZZZ": 300}, {"M": {"symbol": "M", "last": 50.2}, "ZZZ": {"symbol": "ZZZ", "last": 1.0}}),
                            today="20261105", log=log.append)
ok += [check("plain mismatch and an unheld name: nothing", got == [] and not paper.read(paper.SPLITS) and not loud(log) and not log, log)]

# a gateway that raises on positions(), or on the quote: logged, and the dividends are still booked
X = FakeGW.RAW["X"]
xb = {"TRADES": dict(X), "ADJUSTED_LAST": {d: c * ((1 - 0.5 / 50.0) if d < "20261021" else 1.0) for d, c in X.items()}}
for fail, name in ((("positions",), "positions()"), (("quotes",), "the quote")):
    split_case({"a": "m"}, [fill("a", "X", "BOT", 100, "20261019", 1), sfill("a", "S", "BOT", 10, "20261019", 2)],
               {("20261021", "a"): {"S": 300.0}})
    log = []
    got = dividends.book(SplitGW({"X": 100, "S": 30}, {"S": {"symbol": "S", "last": 101.0}},
                                 {"X": xb, "S": {"TRADES": {"20261021": 300.0}}}, fail=fail), today="20261022", log=log.append)
    ok += [check(f"{name} raises: logged, the dividend still booked", [(r["symbol"], r["amount"]) for r in got] == [("X", 50.0)]
                 and any("failed" in l for l in log) and not paper.read(paper.SPLITS), log)]

# a dividend and a split on the same day: 20 shares held into a 3-for-1 that also pays 0.10 a (new) share
D = {"20261103": 50.0, "20261104": 50.0, "20261105": 49.9}
db = {"TRADES": dict(D), "ADJUSTED_LAST": {k: c * ((1 - 0.1 / 50.0) if k < "20261105" else 1.0) for k, c in D.items()}}
split_case({"a": "m"}, [sfill("a", "DV", "BOT", 20, "20261103", 1, 150.0)], {("20261104", "a"): {"DV": 150.3}})
log = []
got = dividends.book(SplitGW({"DV": 60}, {"DV": {"symbol": "DV", "last": 49.9}}, {"DV": db}), today="20261105", log=log.append)
n = notes_of()
ok += [check("same day: split booked and the dividend paid on 60 shares", [r["split"] for r in paper.read(paper.SPLITS)] == ["3-for-1"]
             and [(r["quantity"], r["amount"]) for r in got] == [(20, 6.0)] and qty("a", "DV") == 60, (got, log)),
       check("same day: both notes", len(n.get("a", [])) == 2 and "(+60 shares after the split) were credited $6.00" in n["a"][1], n)]

# a sleeve on hold: its book is restated and its note waits for its next look, as a dividend's does
split_case({"a": "m", "h": "hold"}, [sfill("a", "SOXX", "BOT", 20, "20261103", 1), sfill("h", "SOXX", "BOT", 5, "20261103", 2)],
           {("20261104", "a"): {"SOXX": 450.4}})
got = dividends.book_splits(SplitGW({"SOXX": 75}, {"SOXX": {"symbol": "SOXX", "last": 149.0}}, {"SOXX": SOXX_BARS}), today="20261105",
                            log=[].append)
ok += [check("a sleeve on hold: restated, its note queued", len(got) == 1 and qty("h", "SOXX") == 15 and len(notes_of().get("h", [])) == 1)]

# an already-booked split, then a later second split of the same name
first = {"symbol": "W", "ex_date": "20261102", "ratio": 3}
split_case({"a": "m"}, [sfill("a", "W", "BOT", 10, "20261030", 1, 300.0)], {("20261109", "a"): {"W": 100.0}}, splits=[first])
log = []
r0 = dividends.book_splits(SplitGW({"W": 30}, {"W": {"symbol": "W", "last": 100.0}}), today="20261102", log=log.append)
got = dividends.book_splits(SplitGW({"W": 60}, {"W": {"symbol": "W", "last": 50.5}}, {"W": {"TRADES": {"20261109": 50.0}}}),
                            today="20261110", log=log.append)
ok += [check("the first split not booked again", r0 == [] and not loud(log), log),
       check("a second split of the same name booked", [(r["ex_date"], r["ratio"]) for r in paper.read(paper.SPLITS)]
             == [("20261102", 3), ("20261110", 2)] and qty("a", "W") == 60 and notes_of().get("a") ==
             ["W split 2-for-1 effective Tue 10 Nov: your +30 shares are now +60."], (got, notes_of()))]

# a news gap on top of shares no sleeve holds: books +20, account +25 (5-for-4 by quantity), look 230, quote 200 (-13%, which
# reads 5-for-4 too), and IBKR's close of the look's day not re-based (230.50): no split, the loud line
split_case({"a": "m"}, [sfill("a", "AMZN", "BOT", 20, "20261103", 1, 230.0)], {("20261104", "a"): {"AMZN": 230.0}})
log = []
got = dividends.book_splits(SplitGW({"AMZN": 25}, {"AMZN": {"symbol": "AMZN", "last": 200.0}}, {"AMZN": {"TRADES": {"20261104": 230.5}}}),
                            today="20261105", log=log.append)
ok += [check("a -13% gap with a 5-for-4 quantity: not booked, the loud line", got == [] and not paper.read(paper.SPLITS)
             and qty("a", "AMZN") == 20 and not notes_of() and len(loud(log)) == 1, (got, log))]
# the same with a short and a -33% move against a 3-for-2 quantity
split_case({"a": "m"}, [sfill("a", "AAPL", "SLD", 10, "20261103", 1, 270.0)], {("20261104", "a"): {"AAPL": 270.0}})
got = dividends.book_splits(SplitGW({"AAPL": -15}, {"AAPL": {"symbol": "AAPL", "last": 180.0}}, {"AAPL": {"TRADES": {"20261104": 271.0}}}),
                            today="20261105", log=[].append)
ok += [check("a -33% move with a 3-for-2 quantity: not booked", got == [] and qty("a", "AAPL") == -10, got)]
# no look price, or no IBKR close of the look's day: nothing is booked
split_case({"a": "m", "b": "m"}, SOXX_ROWS)
log = []
got = dividends.book_splits(SplitGW({"SOXX": 105}, {"SOXX": {"symbol": "SOXX", "last": 143.1, "close": 450.41}}, {"SOXX": SOXX_BARS}),
                            today="20261105", log=log.append)
ok += [check("no look price: not booked, the loud line", got == [] and len(loud(log)) == 1 and "no look price" in loud(log)[0], log)]

# a real 3-for-1 that the quantity cannot show: 10 shares no sleeve holds (account 135 against 35), books that net to zero, and
# a 1-for-10 that takes 5 shares to 0: nothing booked, and the loud line, each once a day
for name, rows, acct, look, bars_ in (
        ("shares no sleeve holds", SOXX_ROWS, {"SOXX": 135}, 450.4, 150.14),
        ("books that net to zero", [sfill("a", "SOXX", "BOT", 20, "20261103", 1), sfill("b", "SOXX", "SLD", 20, "20261103", 2)], {}, 450.4,
         150.14),
        ("a 1-for-10 to an account at 0", [sfill("a", "SOXX", "BOT", 5, "20261103", 1, 10.0)], {}, 10.0, 100.0)):
    split_case({"a": "m", "b": "m"}, rows, {("20261104", "a"): {"SOXX": look}})
    log = []
    g = SplitGW(acct, {"SOXX": {"symbol": "SOXX", "last": bars_ * 0.97}}, {"SOXX": {"TRADES": {"20261104": bars_}}})
    r1 = dividends.book_splits(g, today="20261105", log=log.append)
    r2 = dividends.book_splits(g, today="20261105", log=log.append)
    ok += [check(f"a split with {name}: not booked, the loud line once", r1 == r2 == [] and not paper.read(paper.SPLITS)
                 and len(loud(log)) == 1, log)]

# a 07:30 quote whose last is still the pre-split print (450.41) beside a re-based close: IBKR's re-basing and the quantity book it
split_case({"a": "m", "b": "m"}, SOXX_ROWS, SOXX_LOOKS)
log = []
got = dividends.book_splits(SplitGW({"SOXX": 105}, {"SOXX": {"symbol": "SOXX", "last": 450.41, "close": 150.14}}, {"SOXX": SOXX_BARS}),
                            today="20261105", log=log.append)
ok += [check("a pre-split last at 07:30: booked, the quote not used", [r["split"] for r in got] == ["3-for-1"]
             and (qty("a", "SOXX"), qty("b", "SOXX")) == (60, 45) and any("not used" in l for l in log), log)]
# a quote that reads another ratio (2) against IBKR's 3 and the quantity's 3: not booked
split_case({"a": "m", "b": "m"}, SOXX_ROWS, SOXX_LOOKS)
got = dividends.book_splits(SplitGW({"SOXX": 105}, {"SOXX": {"symbol": "SOXX", "last": 225.2}}, {"SOXX": SOXX_BARS}), today="20261105",
                            log=[].append)
ok += [check("the quote reads 2, IBKR and the quantity 3: not booked", got == [] and qty("a", "SOXX") == 20, got)]

# a row added by hand on top of the one booked: the split is applied once
split_case({"a": "m"}, [sfill("a", "SOXX", "BOT", 20, "20261103", 1)],
           splits=[{"symbol": "SOXX", "ex_date": "20261105", "ratio": 3}, {"symbol": "SOXX", "ex_date": "20261105", "ratio": 3}])
ok += [check("a repeated (symbol, ex_date) row is applied once", qty("a", "SOXX") == 60
             and paper.positions_at("a", "20261106") == {"SOXX": 60}, (qty("a", "SOXX"), paper.positions_at("a", "20261106")))]

# A reset's residue (the TWS Gateway Owner, 15 Nov): while the account lists names no sleeve holds, its quantities are no
# split signal. Nothing is booked that day; one line says so; a name whose re-basing shows a ratio still gets its loud line.
STALE = {"AAA": 100, "BBB": -50, "CCC": 7}                       # three names in no sleeve's book
split_case({"a": "m", "b": "m"}, SOXX_ROWS, SOXX_LOOKS)
dividends.RESIDUE_SAID.clear()
gw = SplitGW({"SOXX": 105, **STALE}, {"SOXX": {"symbol": "SOXX", "last": 143.1}}, {"SOXX": SOXX_BARS})
log = []
got = dividends.book_splits(gw, "20261105", log.append) + dividends.book_splits(gw, "20261105", log.append)
ok += [check("residue listed: a real-looking 3-for-1 is not booked; the books keep 20 and 15",
             got == [] and (qty("a", "SOXX"), qty("b", "SOXX")) == (20, 15) and paper.read(paper.SPLITS) == [], got),
       check("residue listed: one 'reset, not split' line a day, naming the count, and the name's loud line once with the reason",
             sum("reset, not split" in l and "3 names no sleeve holds" in l for l in log) == 1
             and sum(l.startswith("SPLIT SUSPECTED, NOT BOOKED: SOXX") and "reset's residue" in l for l in log) == 1, log)]
split_case({"a": "m"}, [sfill("a", "XYZ", "BOT", 20, "20261103", 1, 100.0)], {("20261104", "a"): {"XYZ": 100.0}})
dividends.RESIDUE_SAID.clear()
gw = SplitGW({"XYZ": 40, **STALE}, {"XYZ": {"symbol": "XYZ", "last": 100.5}},
             {"XYZ": {"TRADES": {"20261103": 99.0, "20261104": 100.0}, "ADJUSTED_LAST": {"20261103": 99.0, "20261104": 100.0}}})
log = []
got = dividends.book_splits(gw, "20261105", log.append)
ok += [check("residue listed: a stale quantity that falls on 2-for-1 by chance, prices unchanged: not booked, and no loud line "
             "for the name (it is the reset)", got == [] and qty("a", "XYZ") == 20
             and not any(l.startswith("SPLIT SUSPECTED") for l in log) and any("reset, not split" in l for l in log), log)]
split_case({"a": "m", "b": "m"}, SOXX_ROWS, SOXX_LOOKS)
dividends.RESIDUE_SAID.clear()
gw = SplitGW({"SOXX": 105, "AAA": 100, "BBB": -50}, {"SOXX": {"symbol": "SOXX", "last": 143.1}}, {"SOXX": SOXX_BARS})
log = []
got = dividends.book_splits(gw, "20261105", log.append)
ok += [check("two stray names no sleeve holds are not a reset: the split is booked as before",
             [(r["symbol"], r["ratio"]) for r in got] == [("SOXX", 3)] and not any("reset, not split" in l for l in log), (got, log))]
print("ALL OK" if all(ok) else "FAILED")
