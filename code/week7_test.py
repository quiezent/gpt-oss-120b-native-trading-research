"""Week 7 text and cadence (20 Nov 2026), offline and $0: the schedule clause from cadence and close, NEXT_LOOK's
place, the range line (trader.book_range) and its place, the per-sleeve keys, and loop()'s quarter-hour looks."""

import sys
import tempfile
from datetime import date, time as dtime
from pathlib import Path

import live
import trader

FAILS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


# 1. The schedule clause
check("30 minutes, full day: today's sentence", trader.schedule_clause(30) == trader.SCHEDULE
      and trader.schedule_clause(30, dtime(16, 0)) == trader.SCHEDULE)
check("15 minutes, full day", trader.schedule_clause(15) == "You get a look every 15 minutes, from 09:00 ET (before the 09:30 open) to 15:45 ET.")
check("half day", trader.schedule_clause(30, dtime(13, 0)).endswith("to 12:30 ET.") and trader.schedule_clause(15, dtime(13, 0)).endswith("to 12:45 ET."))
for bad in (60, 0, "15", None):
    try:
        trader.schedule_clause(bad)
        check(f"cadence {bad!r} refused", False)
    except ValueError:
        check(f"cadence {bad!r} refused", True)

# 2. live_system: unchanged by default and on a full day; NEXT_LOOK right after the schedule sentence
kw = dict(ml=False, max_gross=None, own_limit=3.0)
base = trader.live_system(30000.0, date(2026, 11, 23), **kw)
check("full day, 30 minutes, no sentence: as before", trader.live_system(30000.0, date(2026, 11, 23), cadence=30,
                                                                         close=live.close_time(date(2026, 11, 23)), **kw) == base)
half = trader.live_system(30000.0, date(2026, 11, 27), close=live.close_time(date(2026, 11, 27)), **kw)
check("half day: only the last time changes", half == base.replace("to 15:30 ET.", "to 12:30 ET.") and half != base)
nl = trader.live_system(30000.0, date(2026, 11, 23), next_look=True, **kw)
check("NEXT_LOOK right after the schedule sentence, inside its paragraph",
      nl == base.replace(trader.SCHEDULE, trader.SCHEDULE + " " + trader.NEXT_LOOK)
      and (trader.SCHEDULE + " " + trader.NEXT_LOOK + " Each look shows") in nl)

# 3. book_range on made-up bars (trader._daily patched)
real_daily = trader._daily
cal = [f"2026{m:02d}{d:02d}" for m in range(1, 12) for d in range(1, 29)][:300]


def bars(n_days, start=0, drift=0.001, gap=0.0005, stop=None):
    out, c = {}, 100.0
    for k, d in enumerate(cal):
        if k < start or (stop is not None and k > stop):
            continue
        o = c * (1 + (gap if k % 2 else -gap))
        c = c * (1 + (drift if k % 3 else -2 * drift))
        out[d] = (o, c)
    return out


B = {"SPY": bars(300), "AAA": bars(300), "BBB": bars(300, drift=-0.002), "NEW": bars(300, start=270), "MID": bars(300, start=150),
     "OLD": bars(300, stop=200)}
trader._daily = lambda s: B.get(s, {})
before = cal[-1]
st = {"value": 30000.0, "positions": [{"symbol": "AAA", "quantity": 100, "close": 100.0}]}
line = trader.book_range(st, before)
check("one name: 250 sessions, both tails, signs", line and line.startswith("Held as it is, over the last 250 sessions this book's day ranged from -$")
      and " to +$" in line and "and its move from one close to the next open from -$" in line and line.endswith("."), line)
st2 = dict(st, positions=st["positions"] + [{"symbol": "NEW", "quantity": 10, "close": 50.0}, {"symbol": "OLD", "quantity": 10, "close": 50.0}])
line2 = trader.book_range(st2, before)
check("a short history and bars that stop early: both left out, one ending", line2.endswith(" (2 names without enough daily bars left out).")
      and "over the last 250 sessions" in line2, line2)
st3 = dict(st, positions=st["positions"] + [{"symbol": "NEW", "quantity": 10, "close": 50.0}])
check("one left out: singular", trader.book_range(st3, before).endswith(" (1 name without enough daily bars left out)."))
st4 = dict(st, positions=st["positions"] + [{"symbol": "MID", "quantity": 10, "close": 50.0}])
B["YNG"] = bars(300, start=190)              # about 110 sessions in the window: under the floor of 120
check("a name with under 120 sessions is left out", trader.book_range(dict(st, positions=st["positions"] + [
    {"symbol": "YNG", "quantity": 10, "close": 50.0}]), before).endswith("(1 name without enough daily bars left out)."))
GAP = bars(300)
del GAP[cal[-10]]                           # one missing bar 10 sessions back (HLT on 4 Nov, for one)
B["GAP"] = GAP
gl = trader.book_range(dict(st, positions=st["positions"] + [{"symbol": "GAP", "quantity": 10, "close": 50.0}]), before)
check("a name with a recent gap is left out, and the window stays whole", gl.endswith("(1 name without enough daily bars left out).")
      and "over the last 250 sessions" in gl, gl)
check("N is the sessions the names share", "over the last 148 sessions" in trader.book_range(st4, before), trader.book_range(st4, before))
check("nothing held: no line", trader.book_range(dict(st, positions=[]), before) is None)
check("every name left out: no line", trader.book_range(dict(st, positions=[{"symbol": "NEW", "quantity": 5, "close": 9.0}]), before) is None)
check("no value: no line", trader.book_range(dict(st, value=0.0), before) is None)
short = {"value": 30000.0, "positions": [{"symbol": "BBB", "quantity": -100, "close": 100.0}]}
check("a short book's day range is signed by its own moves", trader.book_range(short, before).count("$") == 4)
trader._daily = real_daily

# 4. render_account: the range line after the open orders, before "Since your last decision"
rs = {"value": 30000.0, "cash": 30000.0, "gross": 0.0, "net": 0.0, "beta_net": 0.0, "positions": [], "orders": [], "events": []}
try:
    txt = trader.render_account(rs, {}, "HEADER", range_line="RANGE LINE.")
    i, j, k = txt.find("Open orders: none."), txt.find("RANGE LINE."), txt.find("Since your last decision")
    check("range line between the open orders and 'Since your last decision'", 0 <= i < j < k, txt[-300:])
    check("no range line: as before", "RANGE LINE" not in trader.render_account(rs, {}, "HEADER"))
except Exception as e:                      # render_account needs feats for the risk lines; a fake may not have them
    check(f"render_account with a range line ({type(e).__name__}: {e})", False)

# 5. The per-sleeve keys
b = {"model": "base", "start_cash": 30000.0}
check("next_look / range_line: per sleeve, text B, not Claude",
      live.next_look_of(dict(b, next_look=True)) and not live.next_look_of(b) and not live.next_look_of(dict(b, next_look="yes"))
      and not live.next_look_of({"model": "claude-fable-5-2", "next_look": True}) and not live.next_look_of(dict(b, next_look=True, text="A"))
      and live.range_line_of(dict(b, range_line=True)) and not live.range_line_of(b)
      and live.next_look_of({"model": "thinkingmachines/Inkling-2-Small", "next_look": True}))
check("switch_bad: a key that is not used is reported", any("next_look" in m for m in live.switch_bad({"model": "claude-fable-5-2", "next_look": True}))
      and any("range_line" in m for m in live.switch_bad(dict(b, range_line="on"))) and live.switch_bad(dict(b, next_look=True, range_line=True, cadence=15)) == [])
check("cadence_of", live.cadence_of(dict(b, cadence=15)) == 15 and live.cadence_of(b) == 30 and live.cadence_of(dict(b, cadence=60)) == 30
      and live.cadence_of({"model": "claude-fable-5-2", "cadence": 15}) == 30 and live.cadence_of(dict(b, cadence=True)) == 30)
check("quarter_sleeves", live.quarter_sleeves({"q1": dict(b, cadence=15), "h": b, "q2": dict(b, cadence=15)}) == ["q1", "q2"])
names = [n for n, _ in live.JOBS]
check("JOBS: half-hour and quarter-hour looks in time order", names[:4] == ["refresh", "look-0900", "look-0915", "look-0930"]
      and names[-3:] == ["look-1530", "look-1545", "record"] and len(names) == 30)

# 6. loop(): a quarter-hour look runs only the 15-minute sleeves, in its own 12-minute window; none without such sleeves
tmp = Path(tempfile.mkdtemp())
runs = []


def drive(at, sleeves, look_minutes=None, ticks=4, sleeves_fn=None):
    """loop() on a fake clock from `at`: each look takes look_minutes(o) minutes (o: its only list), each sleep 20 s; STOP
    after `ticks` sleeps. Returns [(HH:MM the look started, its only list)]."""
    clock = [live.datetime(at.year, at.month, at.day, at.hour, at.minute, tzinfo=live.ET)]

    class FakeDT(live.datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0]
    stop = tmp / "STOP"
    stop.unlink(missing_ok=True)
    saved = {k: getattr(live, k) for k in ("datetime", "load_done", "mark_done", "_look", "STOP", "LOOP", "HOLD", "code_line",
                                            "marker_files", "begin_marker", "end_marker", "macro_intraday", "compute_retry",
                                            "close_only_guard", "warm_poll", "own_marker", "refresh", "record", "log")}
    saved_sleeves, saved_sleep = live.paper.sleeves, live.time.sleep
    runs.clear()
    n = [0]

    def fake_look(d, o):
        runs.append((f"{clock[0]:%H:%M}", o))
        clock[0] = clock[0] + live.timedelta(minutes=(look_minutes(o) if look_minutes else 0))

    def fake_sleep(sec):
        clock[0] = clock[0] + live.timedelta(seconds=20)
        n[0] += 1
        if n[0] >= ticks:
            stop.write_text("", encoding="utf-8")
    try:
        live.datetime, live.load_done, live.mark_done = FakeDT, (lambda: set()), (lambda done, key: done.add(key))
        live._look = fake_look
        live.STOP, live.LOOP, live.HOLD = stop, tmp / "LOOP.json", tmp / "HOLD"
        live.code_line, live.marker_files = (lambda: "test"), (lambda: [])
        live.begin_marker = live.end_marker = live.macro_intraday = live.compute_retry = live.close_only_guard = live.warm_poll = (lambda *a: None)
        live.own_marker = lambda: tmp / "none"
        live.refresh = live.record = lambda *a: None
        live.log = lambda m: logged.append(m)  # never live.log (runs/live/live.log is a tracked ledger)
        live.paper.sleeves = sleeves_fn or (lambda: sleeves)
        live.time.sleep = fake_sleep
        live.loop()
    finally:
        for k, v in saved.items():
            setattr(live, k, v)
        live.paper.sleeves, live.time.sleep = saved_sleeves, saved_sleep
    return list(runs)


logged = []
S = {"h": b, "q": dict(b, cadence=15)}
only = lambda r: [o for _, o in r]
check("09:17: the 09:00 look for all, then the 09:15 look for the 15-minute sleeve",
      only(drive(live.datetime(2026, 11, 23, 9, 17), S)) == [None, ["q"]], runs)
check("09:17 with no 15-minute sleeve: the 09:00 look only", only(drive(live.datetime(2026, 11, 23, 9, 17), {"h": b})) == [None], runs)
check("09:22: the 09:15 look is still within its 8 minutes", only(drive(live.datetime(2026, 11, 23, 9, 22), S)) == [None, ["q"]], runs)
check("09:24: the 09:15 look's 8 minutes have passed (the 09:00 look's 25 have not)",
      only(drive(live.datetime(2026, 11, 23, 9, 24), S)) == [None], runs)
check("09:28: the 09:00 look's window has passed too", only(drive(live.datetime(2026, 11, 23, 9, 28), S)) == [], runs)
check("half day 12:50: the 12:30 look, then the 12:45 quarter look", only(drive(live.datetime(2026, 11, 27, 12, 50), S)) == [None, ["q"]], runs)
check("half day 12:53: under 8 minutes to the 13:00 close, no look starts", drive(live.datetime(2026, 11, 27, 12, 53), S) == [], runs)
check("half day 13:05: no look at or after the close", drive(live.datetime(2026, 11, 27, 13, 5), S) == [], runs)
check("full day 15:50: the 15:30 look, then the 15:45 quarter look", only(drive(live.datetime(2026, 11, 23, 15, 50), S)) == [None, ["q"]], runs)
check("full day 15:53: under 8 minutes to the close, no look starts", drive(live.datetime(2026, 11, 23, 15, 53), S) == [], runs)
r = drive(live.datetime(2026, 11, 23, 9, 16), S, look_minutes=lambda o: 13 if o is None else 6)
check("a late 09:00 look (13 minutes from 09:16): the 09:15 look is judged on a fresh clock and not started at 09:29; the "
      "09:30 look runs on time", r == [("09:16", None), ("09:30", None)], r)
r = drive(live.datetime(2026, 11, 23, 9, 0), S, look_minutes=lambda o: 6 if o is None else 5, ticks=150)
check("a normal morning: 09:00 all, 09:15 quarter, 09:30 all, 09:45 quarter, each on time",
      [t for t, _ in r][:4] == ["09:00", "09:15", "09:30", "09:45"] and only(r)[:4] == [None, ["q"], None, ["q"]], r[:4])
bad = iter([ValueError("half-written sleeves.json"), ValueError("again")])


def flaky():
    e = next(bad, None)
    if e:
        raise e
    return S


logged.clear()
r = drive(live.datetime(2026, 11, 23, 9, 17), S, sleeves_fn=flaky, ticks=6)
check("an unreadable sleeves.json never stops the loop; the quarter look runs once the file reads",
      only(r) == [None, ["q"]] and any("sleeves.json not read" in m for m in logged), (r, logged[-3:]))

print("week7_test.py:", "ALL OK" if not FAILS else f"FAILED ({len(FAILS)})")
sys.exit(1 if FAILS else 0)
