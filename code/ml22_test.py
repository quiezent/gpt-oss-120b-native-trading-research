"""ML2.2 checks (no network, no gateway). The build: a release dated today counts once FRED has posted all of it (for GDP,
the growth-rate series itself); ISM and FOMC minutes count on their own day after their hour; the calendar is swapped in
whole; one build at a time. The loop's intraday rebuild: starts 40 minutes after a release with no value, waits 30
minutes between tries, stops at 6 tries or 15:00 ET, never runs two at once or beside another build, kills a hung one."""
import json
import tempfile
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import live
from ml_research import macro_fetch as mf

ET = ZoneInfo("America/New_York")
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(("ok   " if cond else "FAIL ") + name)


def fake_fred(series, release_dates):
    """series: {id: (vintage_dates, [(obs_date, value, rt_start)])}; every row's realtime_end is open."""
    def get(path, **p):
        if path == "release/dates":
            return {"release_dates": [{"date": d} for d in release_dates]}
        if path == "series/vintagedates":
            return {"vintage_dates": series[p["series_id"]][0]}
        if path == "series/observations":
            return {"observations": [dict(date=d, value=str(v), realtime_start=a, realtime_end="9999-12-31")
                                     for d, v, a in series[p["series_id"]][1]]}
        if path == "series":
            vd = series[p["series_id"]][0]
            return {"seriess": [{"last_updated": f"{max(vd)} 09:24:00-05"}]}
        raise AssertionError(path)
    return get


def build(release, series, release_dates, today):
    mf.RELEASES = [release]
    mf.TODAY = today
    mf._vint_cache.clear()
    mf.fred.get = fake_fred(series, release_dates)
    ev, rep = mf.fred_events()
    return {e["date"]: e for e in ev}, rep[release[0]], ev


# ---------------------------------------------------------------- payrolls on 6 Nov
NFP = ("nfp", "Employment Situation", 50, "08:30", "BLS", 1, "PAYEMS",
       [("nfp_change_k", "PAYEMS", "chg", "M"), ("unemployment_rate", "UNRATE", "lin", "M")])
OLD = {"PAYEMS": [("2026-08-01", 100000.0, "2026-09-05"), ("2026-09-01", 100029.0, "2026-10-02")],
       "UNRATE": [("2026-08-01", 4.2, "2026-09-05"), ("2026-09-01", 4.2, "2026-10-02")]}
DATES = ["2026-09-05", "2026-10-02", "2026-11-06", "2026-12-04"]


def nfp(payems_today=True, unrate_today=True, revision_only=False):
    s = {}
    for k, rows in OLD.items():
        posted = payems_today if k == "PAYEMS" else unrate_today
        new = []
        if posted:
            new = [("2026-09-01", 100010.0, "2026-11-06")] if revision_only else \
                  [("2026-10-01", 100194.0 if k == "PAYEMS" else 4.3, "2026-11-06")]
        s[k] = (["2026-09-05", "2026-10-02"] + (["2026-11-06"] if posted else []), rows + new)
    return build(NFP, s, DATES, date(2026, 11, 6))


ev, rep, rows = nfp()
e = ev["2026-11-06"]
check("payrolls posted today: released, new period", e["status"] == "released" and e["kind"] == "new_period")
check("  with October's values", e["first_release"] == {"nfp_change_k": 165.0, "unemployment_rate": 4.3})
check("  an earlier release unchanged", ev["2026-10-02"]["first_release"]["nfp_change_k"] == 29.0)
check("  a later release still scheduled", ev["2026-12-04"]["status"] == "scheduled")
check("  posted time recorded", e["time_check"].get("fred_posted_et") == "09:24")

ev, rep, rows = nfp(payems_today=False, unrate_today=False)
check("not posted: scheduled, not dropped", ev["2026-11-06"]["status"] == "scheduled"
      and "2026-11-06" not in rep["dropped_no_headline_update"])

ev, rep, rows = nfp(unrate_today=False)
check("partly posted (UNRATE still September): scheduled and reported", ev["2026-11-06"]["status"] == "scheduled"
      and rep.get("today_incomplete") == ["2026-11-06"])
check("  one row for the day, and the released count leaves it out",
      sum(1 for x in rows if x["date"] == "2026-11-06") == 1 and rep["released"] == 2)

ev, rep, rows = nfp(revision_only=True)
check("only a revision posted today: scheduled, not a dropped revision-only row",
      ev["2026-11-06"]["status"] == "scheduled" and ev["2026-11-06"].get("kind") is None
      and rep.get("today_incomplete") == ["2026-11-06"])

# ---------------------------------------------------------------- GDP: three series, one headline
GDP = ("gdp", "GDP", 53, "08:30", "BEA", 1, ("A191RL1Q225SBEA", "GDPC1", "GDP"),
       [("real_gdp_qoq_saar", "A191RL1Q225SBEA", "lin", "Q")])


def gdp(a191_today, others_today):
    t = "2026-10-29"
    a191 = (["2026-07-30"] + ([t] if a191_today else []),
            [("2026-04-01", 3.8, "2026-07-30")] + ([("2026-07-01", 2.1, t)] if a191_today else []))
    other = (["2026-07-30"] + ([t] if others_today else []), [("2026-04-01", 23000.0, "2026-07-30")])
    return build(GDP, {"A191RL1Q225SBEA": a191, "GDPC1": other, "GDP": other},
                 ["2026-07-30", t, "2026-11-25"], date(2026, 10, 29))


ev, rep, rows = gdp(a191_today=False, others_today=True)
check("GDP: GDPC1 posted today, growth rate not yet: scheduled (no Q2 value shown as Q3)",
      ev["2026-10-29"]["status"] == "scheduled" and ev["2026-10-29"]["first_release"] is None)
ev, rep, rows = gdp(a191_today=True, others_today=True)
e = ev["2026-10-29"]
check("GDP: growth rate posted today: released as Q3's advance estimate",
      e["status"] == "released" and e["kind"] == "advance" and e["reference_period"] == "2026-07-01"
      and e["first_release"]["real_gdp_qoq_saar"] == 2.1)
check("  Q2's row unchanged", ev["2026-07-30"]["first_release"]["real_gdp_qoq_saar"] == 3.8)

# ---------------------------------------------------------------- ISM and FOMC minutes on their own day
mf.TODAY = date(2026, 11, 2)
mf.NOW = datetime(2026, 11, 2, 9, 0, tzinfo=ET)
ism = {e["date"]: e for e in mf.ism_events() if e["key"] == "ism_mfg"}
check("ISM (Mon 2 Nov) in a 09:00 build: scheduled", ism["2026-11-02"]["status"] == "scheduled")
mf.NOW = datetime(2026, 11, 2, 16, 30, tzinfo=ET)
ism = {e["date"]: e for e in mf.ism_events() if e["key"] == "ism_mfg"}
check("ISM in that day's evening build: released", ism["2026-11-02"]["status"] == "released")
check("  October's ISM still released, December's scheduled", ism["2026-10-01"]["status"] == "released"
      and ism["2026-12-01"]["status"] == "scheduled")
check("minutes dated tomorrow: not out; an earlier day: out",
      not mf._out(date(2026, 11, 3), 14) and mf._out(date(2026, 10, 30), 14))

# ---------------------------------------------------------------- the calendar write and the build lock
tmp = Path(tempfile.mkdtemp())
calls = []
real_replace, real_sleep = mf.os.replace, mf.time.sleep


def flaky_replace(a, b):
    calls.append(1)
    if len(calls) <= 2:
        raise PermissionError("in use")
    real_replace(a, b)


mf.os.replace, mf.time.sleep = flaky_replace, (lambda s: None)
mf.write_calendar({"meta": {}, "events": [1]}, out=tmp)
check("write: retried while a reader held the file, then swapped in whole",
      len(calls) == 3 and json.loads((tmp / "calendar.json").read_text())["events"] == [1]
      and not list(tmp.glob("*.tmp")))
mf.os.replace = lambda a, b: (_ for _ in ()).throw(PermissionError("in use"))
try:
    mf.write_calendar({"meta": {}, "events": [2]}, out=tmp)
except PermissionError:
    pass
check("write: a swap that never succeeds leaves the old file and no temp file",
      json.loads((tmp / "calendar.json").read_text())["events"] == [1] and not list(tmp.glob("*.tmp")))
mf.os.replace = real_replace
(tmp / "calendar.json.99999.tmp").write_text("x")
mf.sweep_temps(tmp)
check("sweep: a killed build's temp file is removed", not list(tmp.glob("*.tmp")))

from ml_research import macro_feats  # noqa: E402
import os  # noqa: E402
kal = tmp / "feats_calendar.json"


def feats_cal(n):
    kal.write_text(json.dumps({"meta": {}, "events": [
        {"key": "claims", "date": f"2026-11-{5 + 7 * i:02d}", "time_et": "08:30", "status": "released",
         "datetime_et": f"2026-11-{5 + 7 * i:02d}T08:30:00-05:00", "first_release": {"initial_claims": 189000.0},
         "expectation": None} for i in range(n)]}), encoding="utf-8")


macro_feats.CAL = kal
feats_cal(1)
os.utime(kal, (1_700_000_000, 1_700_000_000))
n1 = len(macro_feats.load_calendar(path=kal))
feats_cal(2)
os.utime(kal, (1_700_000_100, 1_700_000_100))
n2 = len(macro_feats.load_calendar(path=kal))
check("load_calendar: a swapped-in calendar is read, not the cached one", (n1, n2) == (1, 2))

mf.OUT, mf.LOCK = tmp, tmp / "build.lock"
held = mf.build_lock()
check("lock: busy while a build holds it", live.macro_busy(tmp / "build.lock"))
try:
    mf.build_lock(wait_s=0)
    check("lock: a second build can't take it", False)
except RuntimeError:
    check("lock: a second build can't take it", True)
held.close()
check("lock: free once the build ends", not live.macro_busy(tmp / "build.lock"))
check("lock: free when there's no lock file", not live.macro_busy(tmp / "none.lock"))
mf.time.sleep = real_sleep


# ---------------------------------------------------------------- the loop's intraday rebuild
class FakeProc:
    def __init__(self):
        self.done, self.returncode, self.killed = False, None, False

    def poll(self):
        return self.returncode if self.done else None

    def kill(self):
        self.killed = True


cal = tmp / "calendar.json"
live.LIVE = tmp
live.macro_missing.__defaults__ = (cal,)
live.macro_busy = lambda *a: False
logs, procs = [], []
live.log = logs.append


def popen(*a, **k):
    procs.append(FakeProc())
    return procs[-1]


def finish(p, code=0):
    p.done, p.returncode = True, code


def write_cal(valued):
    cal.write_text(json.dumps({"meta": {}, "events": [
        {"key": "cpi", "date": "2026-11-10", "datetime_et": "2026-11-10T08:30:00-05:00", "status":
         "released" if valued else "scheduled", "first_release": {"cpi_mom_pct": 0.3} if valued else None},
        {"key": "ism_mfg", "date": "2026-11-10", "datetime_et": "2026-11-10T10:00:00-05:00", "status": "scheduled",
         "first_release": None}]}), encoding="utf-8")


def at(hm, day=10):
    h, m = map(int, hm.split(":"))
    return datetime(2026, 11, day, h, m, tzinfo=ET)


write_cal(False)
st = {}
check("missing before +40 min: none", live.macro_missing(at("09:09")) == [])
check("missing at +40 min: cpi only (ISM has no FRED value)", live.macro_missing(at("09:10")) == ["cpi"])
live.macro_intraday(st, at("09:05"), popen)
check("no try before +40 min", len(procs) == 0)
live.macro_intraday(st, at("09:12"), popen)
check("try 1 at 09:12", len(procs) == 1 and st["n"] == 1)
live.macro_intraday(st, at("09:15"), popen)
check("no second while one runs", len(procs) == 1)
finish(procs[0])
live.macro_intraday(st, at("09:22"), popen)
check("ended, value still missing, within 30 min: no new try", len(procs) == 1 and "still no value for cpi" in logs[-1])
live.macro_busy = lambda *a: True
live.macro_intraday(st, at("09:42"), popen)
check("another build holds the lock: no try, none counted", len(procs) == 1 and st["n"] == 1)
live.macro_busy = lambda *a: False
live.macro_intraday(st, at("09:43"), popen)
check("try 2 once it's free", len(procs) == 2)
write_cal(True)
finish(procs[1])
live.macro_intraday(st, at("09:55"), popen)
check("ended with the value in: logged", "values are in" in logs[-1])
live.macro_intraday(st, at("10:30"), popen)
check("value in: no more tries", len(procs) == 2)

write_cal(False)
st, procs[:] = {}, []
for i in range(12):
    live.macro_intraday(st, at(f"{9 + i // 2:02d}:{12 + 30 * (i % 2):02d}"), popen)
    if procs:
        finish(procs[-1])
check("at most 6 tries a day", len(procs) == 6)
st, procs[:] = {}, []
live.macro_intraday(st, at("15:00"), popen)
check("none started from 15:00", len(procs) == 0)

st, procs[:] = {}, []
live.macro_intraday(st, at("09:12"), popen)
st["t0"] -= 31 * 60
live.macro_intraday(st, at("09:44"), popen)
n_logs = len(logs)
live.macro_intraday(st, at("09:45"), popen)
check("hung: killed after 30 minutes, logged once", procs[0].killed and "killed" in logs[n_logs - 1]
      and len(logs) == n_logs)
finish(procs[0], 1)
live.macro_intraday(st, at("09:46"), popen)
check("  then reaped and logged, and a new try started (34 minutes on)", st["proc"] is procs[1] and len(procs) == 2
      and any("ended (exit 1" in x for x in logs[-2:]))

st, procs[:] = {}, []
live.macro_intraday(st, at("14:55"), popen)
st["t0"] -= 31 * 60
live.macro_intraday(st, at("09:00", day=11), popen)
check("hung past midnight: still killed", procs[0].killed and st["day"] == date(2026, 11, 10))
finish(procs[0], 1)
live.macro_intraday(st, at("09:01", day=11), popen)
check("  then the new day starts clean", st["day"] == date(2026, 11, 11) and st["n"] == 0 and st["proc"] is None)

st = {}
live.macro_intraday(st, at("09:12"), lambda *a, **k: (_ for _ in ()).throw(OSError("no python")))
check("a failed start is logged, not raised", "intraday macro rebuild error" in logs[-1] and st.get("proc") is None)
live.log = lambda m: (_ for _ in ()).throw(ValueError("log down"))
st, procs[:] = {}, []
live.macro_intraday(st, at("09:12"), popen)
check("a failing log doesn't raise or orphan the build", len(procs) == 1 and st["proc"] is procs[0])

print("ALL OK" if ok else "FAILURES")
