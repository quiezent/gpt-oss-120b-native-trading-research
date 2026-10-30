"""rates.line on fixed figures: a normal look, an FOMC afternoon before the next refresh, stale figures, a missing or
broken file. Offline.

    python rates_test.py
"""
import json
import tempfile
from datetime import datetime
from pathlib import Path

import rates

FIG = {"fetched_et": "2026-10-28T07:30:05-04:00", "source": "FRED DFEDTARL, DFEDTARU, DGS3MO", "target": [3.75, 4.0],
       "target_as_of": "2026-10-28", "last_change": {"date": "2026-09-17", "bps": 25}, "bill3m": 4.22,
       "bill3m_as_of": "2026-10-27"}


def at(iso, fig=FIG, raw=None):
    with tempfile.TemporaryDirectory() as d:
        rates.FILE = Path(d) / "rates.json"
        if raw is not None:
            rates.FILE.write_text(raw, encoding="utf-8")
        elif fig is not None:
            rates.FILE.write_text(json.dumps(fig), encoding="utf-8")
        return rates.line(datetime.fromisoformat(iso))


def check(name, got, want=(), absent=()):
    ok = all(w in got for w in want) and not any(a in got for a in absent)
    print(("ok   " if ok else "FAIL ") + name + ": " + got)
    return ok


ok = [check("morning of FOMC day", at("2026-10-28T09:00:00-04:00"),
            ["fed funds target 3.75-4.00% (raised 25 bp at the FOMC decision of Wed 16 Sep)",
             "3-month Treasury yield 4.22% (constant maturity, Tue 27 Oct)", "next FOMC decision Wed 28 Oct, 14:00 ET",
             "(Kalshi: hold 91%, +25 bp 7%), then Wed 9 Dec (Kalshi: +25 bp 70%, hold 23%); Kalshi prices of Fri 23 Oct"],
            ["after these"]),
      check("FOMC afternoon, figures from 07:30", at("2026-10-28T14:30:00-04:00"),
            ["the FOMC decided on Wed 28 Oct at 14:00 ET, after these figures (see tape)", "next FOMC decision Wed 9 Dec"],
            ["Kalshi"]),                            # the calendar's odds were priced before the decision
      check("next morning, fresh fetch", at("2026-10-29T09:00:00-04:00", dict(FIG, fetched_et="2026-10-29T07:30:00-04:00")),
            ["next FOMC decision Wed 9 Dec"], ["after these"]),
      check("figures 10 days old", at("2026-11-08T09:00:00-05:00"), ["next FOMC decision"],
            ["fed funds", "Treasury", "after these", "Kalshi"]),     # the calendar's Kalshi prices are stale too
      check("no file", at("2026-10-28T09:00:00-04:00", fig=None) or "(empty)", ["(empty)"]),
      check("broken file", at("2026-10-28T09:00:00-04:00", raw="{not json") or "(empty)", ["(empty)"])]
print("ALL OK" if all(ok) else "FAILED")
