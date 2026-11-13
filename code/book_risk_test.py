"""trader.book_risk and its line in render_account: a hand-checked two-name book, stops coverage, the default render
unchanged, and the line on a stored look (NAV9 + ML2, 5 Nov 15:30). Offline: data/bars and stored look files only.

    python book_risk_test.py
"""
import json
import sys
from pathlib import Path

import numpy as np

import trader

ROOT = Path(__file__).resolve().parent
ok = []


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail else ""))


BEFORE = "20261106"
amd, smh = trader._daily("AMD"), trader._daily("SMH")
st = {"value": 30000.0, "positions": [{"symbol": "AMD", "quantity": -20, "close": 404.40},
                                      {"symbol": "SMH", "quantity": 10, "close": 447.73}],
      "orders": [{"symbol": "AMD", "action": "BUY", "quantity": 20, "type": "STP", "stop": 430.0}]}
line = trader.book_risk(st, BEFORE)
cal = sorted(x for x in trader._daily("SPY") if x < BEFORE)[-251:]
day = np.array([-20 * 404.40 * (amd[c][1] / amd[p][1] - 1) + 10 * 447.73 * (smh[c][1] / smh[p][1] - 1) for p, c in zip(cal, cal[1:])])
gap = np.array([-20 * 404.40 * (amd[c][0] / amd[p][1] - 1) + 10 * 447.73 * (smh[c][0] / smh[p][1] - 1) for p, c in zip(cal, cal[1:])])
want_day, want_gap = trader.fmt_money(-day.min()), trader.fmt_money(-gap.min())
check("worst day by hand", want_day in line, f"{want_day} in: {line[:140]}")
check("worst gap by hand", want_gap in line, want_gap)
check("the last day counted is the one before", cal[-1] == "20261105" and "250 sessions" in line, cal[-1])
bare = 100 * (10 * 447.73) / (20 * 404.40 + 10 * 447.73)
check("AMD's stop covers it; SMH has none", f"{bare:.0f}% of its gross has no working stop" in line, f"{bare:.0f}%")
st2 = dict(st, orders=[dict(st["orders"][0], cancel_pending=True)])
check("a stop pending cancel doesn't count", "100% of its gross has no working stop" in trader.book_risk(st2, BEFORE))
check("no positions, no line", trader.book_risk(dict(st, positions=[]), BEFORE) is None)
check("a name with no bars is flagged, not fatal",
      "ZZZZ" in (trader.book_risk(dict(st, positions=st["positions"] + [{"symbol": "ZZZZ", "quantity": 5, "close": 10.0}]), BEFORE) or ""))

# render: the default is unchanged; the line sits after the per-position risk lines
look = json.loads((ROOT / "runs/live/2026-11-05/reserve-2-1530.json").read_text(encoding="utf-8"))
s = dict(look["state"], events=look.get("events") or [])
feats = {}
a = trader.render_account(s, feats, "H", None, 2.0)
b = trader.render_account(s, feats, "H", None, 2.0, book="BOOK LINE")
check("without the flag the account text is unchanged", "BOOK LINE" not in a and b.replace("\nBOOK LINE", "") == a)
check("the line comes before the open orders", b.index("BOOK LINE") < b.index("Open orders"))
real = trader.book_risk(s, "20261105")
print("   NAV9 + ML2, 5 Nov 15:30:", real)
check("a real stored book renders a line", real and "worst overnight gap" in real)

# the review of dfa3048: bars that stop early, zero-volume placeholders, a book with no history at all
tq = trader.book_risk({"value": 30000.0, "positions": [{"symbol": "TQQQ", "quantity": 300, "close": 100.0}], "orders": []}, "20261106")
check("bars that stop early are named, with their last date", tq and "TQQQ (no bars after 20261019)" in tq, (tq or "")[-120:])
sh = trader.book_risk({"value": 30000.0, "positions": [{"symbol": "SPY", "quantity": 20, "close": 735.0},
                                                       {"symbol": "SHAZ", "quantity": -100, "close": 30.0}], "orders": []}, "20261106")
check("zero-volume placeholder bars don't make a fake worst day", sh and "156" not in sh and "149" not in sh, sh[:150])
check("a book with no price history gets no line",
      trader.book_risk({"value": 30000.0, "positions": [{"symbol": "ZZZZ", "quantity": 5, "close": 10.0}], "orders": []}, BEFORE) is None)
check("no negative zero anywhere", all("-0.0" not in (x or "") and "$-" not in (x or "") for x in (line, tq, sh, real)))
print("ALL OK" if all(ok) else "FAILED")
sys.exit(0 if all(ok) else 1)
