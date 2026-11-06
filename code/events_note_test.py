"""events.note's next-earnings clause on fixed rows: the Alpha Vantage calendar's date, when live has one, over the E1
estimate (INTC "overdue" on 29 Oct, its report day; TSLA's 21 Oct report hidden by its delivery 8-Ks), and the E1
estimate as before when there is none (the simulator's rows). Offline.

    python events_note_test.py
"""
import sys

from ml_research import events

E1_INTC = {"next_earn_overdue": True, "next_earn_day": "2026-10-22", "next_earn_when": "after close", "next_earn_sessions": -5}
E1_TSLA = {"next_earn_overdue": False, "next_earn_day": "2027-01-01", "next_earn_when": "closed day", "next_earn_sessions": 50}
E1_SOON = {"next_earn_overdue": False, "next_earn_day": "2026-11-03", "next_earn_when": "after close", "next_earn_sessions": 3}


def cal(day, when, k, reported=False):
    return {"calendar_day": day, "calendar_when": when, "calendar_sessions": k, "calendar_reported": reported}


def check(name, row, want=(), absent=()):
    got = events.note(row)
    ok = all(w in got for w in want) and not any(a in got for a in absent)
    print(("ok   " if ok else "FAIL ") + name + ": " + (got or "(empty)"))
    return ok


ok = [check("INTC 29 Oct 15:30: calendar today, E1 overdue", dict(E1_INTC, **cal("2026-10-29", "after close", 0)),
            ["next earnings 2026-10-29 (after close; Alpha Vantage calendar)"], ["overdue", "est."]),
      check("TSLA 20 Oct: calendar tomorrow, E1 in January", dict(E1_TSLA, **cal("2026-10-21", "after close", 1)),
            ["next earnings 2026-10-21 (after close; Alpha Vantage calendar)"], ["2027"]),
      check("calendar 15 sessions out: no clause, E1 not shown", dict(E1_INTC, **cal("2026-11-20", "before open", 15)),
            absent=["earnings"]),
      check("reported this morning: the 8-K clause, no next date",
            dict(E1_SOON, earnings_since_prev_close=True, recent_8k=[{"accepted_et": "2026-10-27 07:30", "when": "before open",
                                                                      "items": "2.02 results"}],
                 **cal("2026-10-27", "before open", 0, reported=True)),
            ["EARNINGS since the previous close"], ["next earnings"]),
      check("no calendar (simulator): E1 overdue as before", dict(E1_INTC), ["next earnings est. 2026-10-22 (overdue)"]),
      check("no calendar (simulator): E1 within 10 sessions as before", dict(E1_SOON),
            ["next earnings est. 2026-11-03 (after close)"], ["Alpha Vantage"]),
      check("AV time of day mapped", {"calendar_day": "2026-11-03", "calendar_sessions": 2,
                                       "calendar_when": events.AV_WHEN.get("post-market")}, ["(after close;"])]


def fields(name, day, when, today, earn_et, want):
    """events.calendar_fields on an Alpha Vantage row; earn_et: earnings 8-K times as ET 'YYYY-MM-DD HH:MM'."""
    earn_t = [events.et_epoch(x[:10], int(x[11:13]), int(x[14:16])) for x in earn_et]
    got = events.calendar_fields(day, when, today, earn_t)
    good = all(got.get(k) == v for k, v in want.items()) and (want or not got)
    print(("ok   " if good else "FAIL ") + name + ": " + str(got))
    return good


ok += [fields("INTC 29 Oct: dated, after close, not yet reported", "2026-10-29", "post-market", "2026-10-29",
              ["2026-07-23 16:09"], {"calendar_day": "2026-10-29", "calendar_when": "after close", "calendar_sessions": 0,
                                     "calendar_reported": False}),
       fields("AMD on Fri 30 Oct for Tue 3 Nov: two sessions out", "2026-11-03", "post-market", "2026-10-30",
              ["2026-08-04 16:16"], {"calendar_sessions": 2, "calendar_reported": False}),
       fields("DHR 21 Oct: its 17:43 filing read as 13:43 the day before still counts as reported", "2026-10-21",
              "pre-market", "2026-10-21", ["2026-10-20 13:43"], {"calendar_reported": True}),
       fields("CNC 27 Oct: 19:05 read as 15:05 on 26 Oct, reported", "2026-10-27", "pre-market", "2026-10-27",
              ["2026-10-26 15:05"], {"calendar_reported": True}),
       fields("SW: no time of day, AV's projection: no fields (E1 stays)", "2026-11-04", "", "2026-10-22", [], {}),
       fields("a 2.02 two weeks before a date 8 sessions out does not hide it", "2026-11-12", "pre-market", "2026-11-02",
              ["2026-10-20 16:05"], {"calendar_sessions": 8, "calendar_reported": False})]


def shared_cache():
    """events._submissions_live with several processes on one folder: a part-written copy is fetched again, not raised,
    and a fetch leaves a whole file and no temp file."""
    import json
    import tempfile
    from datetime import datetime, timezone
    from pathlib import Path

    from ml_research import edgar_fetch as ef
    now, got = datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc), []
    real_dir, real_get = events.LIVE_DIR, ef.get
    with tempfile.TemporaryDirectory() as d:
        events.LIVE_DIR = Path(d)
        ef.get = lambda url: json.dumps({"filings": {"recent": {"form": ["8-K"]}}})
        try:
            f = Path(d) / "20261102" / "CIK0000000001.json"
            f.parent.mkdir()
            f.write_text('{"filings": {"rec', encoding="utf-8")
            j, src = events._submissions_live(1, now, got, "X")
            good = src == "fetched" and j["filings"]["recent"]["form"] == ["8-K"]
            print(("ok   " if good else "FAIL ") + f"a part-written cache file is fetched again: {src}")
            whole = json.loads(f.read_text(encoding="utf-8"))["filings"]["recent"]["form"] == ["8-K"] and not list(f.parent.glob("*.tmp"))
            print(("ok   " if whole else "FAIL ") + "the fetch leaves a whole file and no temp file")
            j, src = events._submissions_live(1, now, got, "X")
            again = src == "cache"
            print(("ok   " if again else "FAIL ") + f"the next call reads the cache: {src}")
            return good and whole and again
        finally:
            events.LIVE_DIR, ef.get = real_dir, real_get


ok.append(shared_cache())
print("ALL OK" if all(ok) else "FAILED")
sys.exit(0 if all(ok) else 1)
