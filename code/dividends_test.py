"""dividends.book on a fake gateway and temporary ledgers. Offline.

    python dividends_test.py
"""
import json
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
print("ALL OK" if all(ok) else "FAILED")
