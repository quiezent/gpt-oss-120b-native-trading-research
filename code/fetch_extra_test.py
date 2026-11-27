"""Offline checks of fetch_extra's 30-minute bar upkeep (merge_bars30, update, the 16:15 record's call): IBKR's overnight
revisions replace a stored session, a session a failed fetch missed is filled, a session IBKR has since put on another
price scale (a split) stays as traded, nothing is ever half-written. A temporary data folder and fake gateways; nothing
under data/ or runs/ is read or written ($0).

    python fetch_extra_test.py
"""
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import fetch_extra  # noqa: E402

ok = []


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail and not good else ""))


SLOTS = [f"{h:02d}:{m:02d}:00" for h in range(9, 16) for m in (0, 30)][1:]          # 09:30 ... 15:30, 13 bars


def day(d, px, vol=1000, zone="US/Eastern", n=13, bump=None):
    """13 bars of session d around price px; bump = (slot index, cents) moves one bar's close."""
    out = []
    for i, t in enumerate(SLOTS[:n]):
        c = round(px + 0.01 * i, 2)
        if bump and bump[0] == i:
            c = round(c + 0.01 * bump[1], 2)
        out.append({"date": f"{d} {t} {zone}", "open": round(px + 0.01 * i, 2), "high": round(c + 0.05, 2),
                    "low": round(px - 0.05, 2), "close": c, "volume": vol + i})
    return out


def sessions(bars):
    return sorted({b["date"][:8] for b in bars})


# --- merge_bars30 --------------------------------------------------------------------------------------------------
old = day("20261116", 100) + day("20261117", 101) + day("20261119", 103)
new = day("20261116", 100) + day("20261117", 101, bump=(5, 1)) + day("20261118", 102) + day("20261119", 103) + day("20261120", 104)
bars, st = fetch_extra.merge_bars30(old, new)
check("a session IBKR revised by a cent replaces the stored one", st["revised"] == ["20261117"]
      and [b for b in bars if b["date"].startswith("20261117")] == day("20261117", 101, bump=(5, 1)), st)
check("an unchanged session is counted as same", st["same"] == 2, st)
check("a missed session (18 Nov, as GM 17 Nov) is filled, and today's is added", st["added"] == ["20261118", "20261120"], st)
check("one bar per slot, in time order", [b["date"][:17] for b in bars] == sorted({b["date"][:17] for b in bars})
      and len(bars) == 5 * 13, len(bars))

big = day("20261117", 101, bump=(3, 40))                                              # 40 cents on $101: about 40 bp
_, st = fetch_extra.merge_bars30(day("20261117", 101), big)
check("a 40 bp revision (the largest seen, MU) is a revision, not another scale", st["revised"] == ["20261117"] and not st["rescaled"], st)

# SOXX: 3-for-1 from 5 Nov. Stored as traded: 451 before, 143 from 5 Nov. IBKR's history now: all on the new scale.
old = day("20261103", 451) + day("20261104", 450) + day("20261105", 143) + day("20261106", 140)
new = (day("20261103", 150.33) + day("20261104", 150) + day("20261105", 143, bump=(2, 1)) + day("20261106", 140)
       + day("20261109", 139))
bars, st = fetch_extra.merge_bars30(old, new)
check("SOXX: sessions before the split stay as traded", st["rescaled"] == ["20261103", "20261104"]
      and [b for b in bars if b["date"][:8] in ("20261103", "20261104")] == day("20261103", 451) + day("20261104", 450), st)
check("SOXX: sessions after it are still revised and added", st["revised"] == ["20261105"] and st["added"] == ["20261109"]
      and st["same"] == 1, st)

# BITO: 1-for-2 from 19 Nov; 19 Nov missed by the desk. Stored 8.30 before, 16.48 on 20 Nov; IBKR now all at about 2x.
old = day("20261117", 8.5) + day("20261118", 8.3) + day("20261120", 16.48)
new = day("20261117", 17.0) + day("20261118", 16.6) + day("20261119", 16.95) + day("20261120", 16.48, bump=(0, 1))
bars, st = fetch_extra.merge_bars30(old, new)
check("BITO: a missed session between the last old-scale one and the next stored one is ambiguous without the ledger "
      "(it may be the ex-date or before it): not filled", st["ambiguous"] == ["20261119"] and "20261119" not in sessions(bars), st)
check("BITO: old-scale sessions kept, 20 Nov revised", st["rescaled"] == ["20261117", "20261118"] and st["revised"] == ["20261120"]
      and [b["close"] for b in bars if b["date"].startswith("20261118")] == [b["close"] for b in day("20261118", 8.3)], st)
bars, st = fetch_extra.merge_bars30(old, new, ["20261119"])
check("BITO: with the ledger's ex-date 19 Nov the missed session is the first on the new scale, and is filled",
      st["added"] == ["20261119"] and not st["ambiguous"] and sessions(bars) == ["20261117", "20261118", "20261119", "20261120"], st)

old = day("20261104", 450) + day("20261105", 143)                                    # 3 Nov never stored
new = day("20261103", 150.3) + day("20261104", 150) + day("20261105", 143)
bars, st = fetch_extra.merge_bars30(old, new)
check("a missing session before a rescaled one is not added (its price as traded is unknown)",
      st["before_split"] == ["20261103"] and "20261103" not in sessions(bars), st)

# The reviewer's case (21 Nov): SOXX stored as traded to 3 Nov, 4 Nov missed, the 16:15 record of 5 Nov fetches 5 sessions.
old = day("20261030", 452) + day("20261102", 451) + day("20261103", 452)
new = day("20261030", 150.67) + day("20261102", 150.33) + day("20261103", 150.67) + day("20261104", 150.5) + day("20261105", 143)
bars, st = fetch_extra.merge_bars30(old, new)
check("a missed last session before the split, no ledger: ambiguous, not filled at IBKR's adjusted price; the ex-date's added",
      st["ambiguous"] == ["20261104"] and st["added"] == ["20261105"] and "20261104" not in sessions(bars), st)
bars, st = fetch_extra.merge_bars30(old, new, ["20261105"])
check("... with the ledger's ex-date 5 Nov: 4 Nov is before the split, not filled; 5 Nov added",
      st["before_split"] == ["20261104"] and st["added"] == ["20261105"] and not st["ambiguous"], st)
bars, st = fetch_extra.merge_bars30([], day("20261103", 150.67) + day("20261104", 150.5) + day("20261105", 143), ["20261105"])
check("no stored session in the window: the ledger's split keeps IBKR's adjusted sessions before it out",
      st["before_split"] == ["20261103", "20261104"] and st["added"] == ["20261105"], st)
bars, st = fetch_extra.merge_bars30(day("20261116", 50), day("20261116", 50, bump=(2, 1)) + day("20261117", 51), ["20261020"])
check("a ledger split before the window changes nothing", st["revised"] == ["20261116"] and st["added"] == ["20261117"], st)

# SCALE_TOL and the median, pinned
shift = lambda bars, f: [dict(b, open=b["open"] * f, high=b["high"] * f, low=b["low"] * f, close=b["close"] * f) for b in bars]
_, st = fetch_extra.merge_bars30(day("20261116", 40), shift(day("20261116", 40), 1.015))
check("every bar 1.5% off: a revision (inside SCALE_TOL 2%)", st["revised"] == ["20261116"] and not st["rescaled"], st)
_, st = fetch_extra.merge_bars30(day("20261116", 40), shift(day("20261116", 40), 1.03))
check("every bar 3% off: another scale", st["rescaled"] == ["20261116"], st)
_, st = fetch_extra.merge_bars30(day("20261116", 40), shift(day("20261116", 40), 2 / 3))
check("a 3-for-2 split: another scale", st["rescaled"] == ["20261116"], st)
_, st = fetch_extra.merge_bars30(day("20261116", 40), shift(day("20261116", 40), 0.8))
check("a 5-for-4 split: another scale", st["rescaled"] == ["20261116"], st)
one = day("20261116", 40)
one[4] = dict(one[4], close=one[4]["close"] * 1.3)
_, st = fetch_extra.merge_bars30(day("20261116", 40), one)
check("one bad print corrected by 30%, twelve bars equal: a revision (the median, not the mean)",
      st["revised"] == ["20261116"] and not st["rescaled"], st)

old = day("20261116", 50) + day("20261117", 51)
new = day("20261117", 51, n=12, bump=(1, 2))
bars, st = fetch_extra.merge_bars30(old, new)
check("IBKR now has fewer bars of a session: the stored one is kept", st["short"] == ["20261117"]
      and [b for b in bars if b["date"].startswith("20261117")] == day("20261117", 51), st)
check("a stored session IBKR no longer serves (GM 16 Nov, HLT 4 Nov) is kept", "20261116" in sessions(bars), sessions(bars))

old = day("20261116", 60, zone="America/New_York")
new = day("20261116", 60, bump=(4, 1))
bars, st = fetch_extra.merge_bars30(old, new)
check("a stored session with another zone label (ARKK 16 Nov) is replaced whole: no slot twice",
      len(bars) == 13 and st["revised"] == ["20261116"] and all(b["date"].endswith("US/Eastern") for b in bars), st)

_, st = fetch_extra.merge_bars30(day("20261116", 60), [dict(b, volume=float(b["volume"])) for b in day("20261116", 60)])
check("volume as 1000.0 against 1000 is no revision", st["same"] == 1 and not st["revised"], st)

bars, st = fetch_extra.merge_bars30([], day("20261120", 10))
check("an empty file gets the fetched sessions", st["added"] == ["20261120"] and len(bars) == 13, st)

# --- update ---------------------------------------------------------------------------------------------------------
tmp = Path(tempfile.mkdtemp(prefix="fetch_extra_"))
fetch_extra.ROOT = tmp
fetch_extra.time.sleep = lambda s: None
d30 = tmp / "data" / "bars30"
d30.mkdir(parents=True)
(d30 / "AAA.json").write_text(json.dumps({"symbol": "AAA", "what": "TRADES", "bars": day("20261119", 20)}), encoding="utf-8")
(d30 / "BRK_B.json").write_text(json.dumps({"symbol": "BRK B", "what": "TRADES", "bars": day("20261119", 500)}), encoding="utf-8")
before_bbb = json.dumps({"symbol": "BBB", "what": "TRADES", "bars": day("20261119", 30)})
(d30 / "BBB.json").write_text(before_bbb, encoding="utf-8")
FETCH = {"AAA": day("20261119", 20, bump=(0, 1)) + day("20261120", 21), "BRK B": day("20261119", 500) + day("20261120", 501)}
calls = []


class GW:
    def bars(self, s, duration, bar_size, what, rth):
        calls.append((s, duration, bar_size, what, rth))
        if s not in FETCH:
            raise RuntimeError("162: HMDS query returned no data")
        if what != "TRADES":
            return [{"date": "20261120", "open": 1, "high": 1, "low": 1, "close": 0.3, "volume": 0}]
        return FETCH[s]


stats = {}
bad = fetch_extra.update(GW(), ["AAA", "BBB", "BRK B"], "bars30", duration="5 D", stats=stats)
doc = json.loads((d30 / "AAA.json").read_text(encoding="utf-8"))
check("update: 30-minute TRADES regular hours, the duration asked", calls[0] == ("AAA", "5 D", "30 mins", "TRADES", True), calls)
check("update: revised and today's sessions written", stats["AAA"]["revised"] == ["20261119"] and stats["AAA"]["added"] == ["20261120"]
      and sessions(doc["bars"]) == ["20261119", "20261120"] and doc["what"] == "TRADES" and doc["symbol"] == "AAA", stats)
check("update: a symbol with a space is the file with an underscore", stats["BRK B"]["added"] == ["20261120"]
      and len(json.loads((d30 / "BRK_B.json").read_text(encoding="utf-8"))["bars"]) == 26, stats)
check("update: a failed fetch is reported and its file left as it was",
      bad == ["BBB: 162: HMDS query returned no data"] and (d30 / "BBB.json").read_text(encoding="utf-8") == before_bbb, bad)
check("update: no temp file left", not list(d30.glob("*.tmp")), list(d30.glob("*")))

line = fetch_extra.bars30_summary(stats, bad, "20261120")
check("the log line: today's sessions, revisions, fills, splits, failures",
      line.startswith("30-minute bars: 2 sessions of 20261120 added; 1 earlier sessions revised by IBKR since fetched, 1 unchanged; "
                      "0 missed sessions filled []") and "1 failed ['BBB: 162" in line, line)
st2 = {"X": {"revised": [], "same": 0, "added": ["20261117", "20261120"], "rescaled": ["20261104"], "before_split": ["20261103"],
             "ambiguous": ["20261105"], "short": []}}
line = fetch_extra.bars30_summary(st2, [], "20261120")
check("the log line names a filled session, a split name and the sessions not filled", "1 missed sessions filled ['X 20261117']" in line
      and "1 sessions kept as traded after a split ['X']" in line
      and "2 missed sessions not filled (IBKR's price is split-adjusted, or may be) ['X 20261103', 'X 20261105']" in line, line)

real_replace = os.replace
n = {"i": 0}


def flaky(a, b):
    n["i"] += 1
    if n["i"] <= 2:
        raise PermissionError("in use")
    return real_replace(a, b)


fetch_extra.os.replace = flaky
FETCH["AAA"] = day("20261120", 21, bump=(1, 1))
bad = fetch_extra.update(GW(), ["AAA"], "bars30", duration="5 D", stats={})
check("a replace refused twice (a reader holding the file) is retried and lands", bad == [] and n["i"] == 3
      and json.loads((d30 / "AAA.json").read_text(encoding="utf-8"))["bars"][-12]["close"] == day("20261120", 21, bump=(1, 1))[1]["close"],
      (bad, n))
before = (d30 / "AAA.json").read_text(encoding="utf-8")
fetch_extra.os.replace = lambda a, b: (_ for _ in ()).throw(PermissionError("in use"))
FETCH["AAA"] = day("20261120", 21, bump=(2, 1))
st_fail = {}
bad = fetch_extra.update(GW(), ["AAA"], "bars30", duration="5 D", stats=st_fail)
check("a replace refused five times: reported, the file as it was, no temp file, and not counted in the log's stats",
      bad and bad[0].startswith("AAA:") and (d30 / "AAA.json").read_text(encoding="utf-8") == before and not list(d30.glob("*.tmp"))
      and "AAA" not in st_fail, (bad, st_fail))
fetch_extra.os.replace = real_replace

import paper  # noqa: E402

real_splits = paper.splits
paper.splits = lambda: [{"symbol": "AAA", "ex_date": "20261120", "ratio": 2}]
(d30 / "AAA.json").write_text(json.dumps({"symbol": "AAA", "what": "TRADES", "bars": day("20261118", 40)}), encoding="utf-8")
FETCH["AAA"] = day("20261118", 20) + day("20261119", 20.1) + day("20261120", 20.2)      # IBKR adjusted 18-19 Nov for a 2:1
st_l = {}
bad = fetch_extra.update(GW(), ["AAA"], "bars30", duration="5 D", stats=st_l)
check("update passes the split ledger's ex-dates: a missed 19 Nov before the 20 Nov split is not filled",
      bad == [] and st_l["AAA"]["rescaled"] == ["20261118"] and st_l["AAA"]["before_split"] == ["20261119"]
      and st_l["AAA"]["added"] == ["20261120"], st_l)
paper.splits = lambda: (_ for _ in ()).throw(OSError("locked"))
st_l = {}
bad = fetch_extra.update(GW(), ["AAA"], "bars30", duration="5 D", stats=st_l)
check("an unreadable split ledger is reported, and the bars are still merged", bad == ["split ledger not read: OSError: locked"]
      and "AAA" in st_l, (bad, st_l))
paper.splits = real_splits

(tmp / "data" / "iv").mkdir(parents=True)
(tmp / "data" / "iv" / "AAA.json").write_text(json.dumps({"symbol": "AAA", "what": "OPTION_IMPLIED_VOLATILITY", "bars": [
    {"date": "20261119", "open": 1, "high": 1, "low": 1, "close": 0.2, "volume": 0},
    {"date": "20261120", "open": 1, "high": 1, "low": 1, "close": 0.25, "volume": 0}]}), encoding="utf-8")
bad = fetch_extra.update(GW(), ["AAA"], "iv")
iv = json.loads((tmp / "data" / "iv" / "AAA.json").read_text(encoding="utf-8"))
check("iv: dates replaced by the fetched ones, as before (no per-session rule)", bad == [] and calls[-1][1:] == ("1 M", "1 day", "OPTION_IMPLIED_VOLATILITY", True)
      and [b["close"] for b in iv["bars"]] == [0.2, 0.3] and iv["what"] == "OPTION_IMPLIED_VOLATILITY", iv)

# --- the 16:15 record calls it with 5 sessions and logs the line ----------------------------------------------------
import live  # noqa: E402
import paper  # noqa: E402
import trader  # noqa: E402

lt = tmp / "ledgers"
lt.mkdir()
for k, f in (("FILLS", "fills.jsonl"), ("ORDERS", "orders.jsonl"), ("SLEEVES", "sleeves.json"),
             ("DIVIDENDS", "dividends.jsonl"), ("SPLITS", "splits.jsonl")):
    setattr(paper, k, lt / f)
live.LIVE = lt / "live"
paper.SLEEVES.write_text(json.dumps({"x-ml": {"start_cash": 30000.0, "start": "2026-11-23T12:00:00+00:00", "model": "m",
                                              "ml": True, "ml_tool": "ML4"}}), encoding="utf-8")


class GW2(GW):
    def quotes(self, syms):
        return [{"symbol": s, "last": 600.0} for s in syms]

    def account(self):
        return {"NetLiquidation": 720000.0}


@contextlib.contextmanager
def fake_gateway():
    yield GW2()


logged = []
live.gateway = fake_gateway
paper.sync_fills = lambda gw: None
live.reconcile = lambda gw: None
live.book_dividends = lambda gw=None: None
live.macro_rebuild = lambda: None
live.log = lambda m, *a, **k: logged.append(m)
trader.universe = lambda: ["AAA"]
calls.clear()
FETCH["AAA"] = day("20261120", 21, bump=(3, 1))
live.record()
check("record(): with an ML sleeve, the universe's last 5 sessions of 30-minute bars", calls == [("AAA", "5 D", "30 mins", "TRADES", True)], calls)
check("record(): the bars line is logged", any(m.startswith("30-minute bars: ") and "earlier sessions revised" in m for m in logged), logged)

print(f"\nfetch_extra_test.py: {'ALL OK' if all(ok) else f'{ok.count(False)} FAILED'}")
sys.exit(0 if all(ok) else 1)
