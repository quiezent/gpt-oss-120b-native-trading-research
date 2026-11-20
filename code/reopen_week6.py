"""Re-open Base + ML2's carried book and re-enter its open orders after the $1,000,000 paper reset (Clayton, 14 Nov 2026).

What happened: Clayton reset the paper account's cash to $1,000,000 on Sat 14 Nov (ET). IBKR removes every position held
at a reset with no execution (17 Oct, 25 Oct, 9 Nov), and the Gateway may go on listing the removed positions ("stale")
for a session or two, sometimes dropping only some of them first (27 Oct). A resting stop that triggers after a reset
opens a reverse position (11 did on 9 Nov), so all 59 open orders of client 10 were cancelled on Saturday after a
snapshot (runs/live/reset_20261116_open_orders.json: every order's terms and every sleeve's book). One sleeve, base-ml,
carries its book into Week 6: its 24 positions are re-opened at IBKR so the account again holds what base-ml's ledger
says, and its 11 cancelled orders (the model's own decisions) are re-entered with the same terms.

Clients. The re-open's MKT orders, and every read about them, go on client 18 ("trader-reopen"), so this script never
holds client 10 when the live loop connects for a look (a look that gets 326 is lost for every sleeve). find(ref) and
orders() cover only the caller's own client, so a re-open order is judged working, filled or gone only from client 18
(or from all-client reads: executions(all_clients=True) and all_open_orders()), never from a client-10 read. --orders runs
on client 10 ("trader-training"): it places base-ml's own orders through paper.place and reads them (orders(), find());
the re-open's state it takes from executions(all_clients=True) and all_open_orders() only. --hold and --release need
no Gateway.

Re-open refs are reopen-20261116-<SYM>, and reopen-20261116-<SYM>-<n> (n = 2, 3, ...) for a remainder. paper.sleeve_of
maps those to no sleeve, so their fills are not booked to base-ml, whose ledger already holds the positions at its own
cost basis; the ledger is not touched. Every re-open order sent is logged in runs/live/reopen_20261116.jsonl (with the
find() checks and what the model was told), so a ref is never used twice.

Modes (each prints what it would do and changes nothing without --go; the last line is "RESULT: ..."; exit code 0 only
when the step fully succeeded):

    python reopen_week6.py --hold [--go]      hold base-ml: its sleeves.json model becomes "hold" (live.HOLD_MODEL: no
                                              look), the real one kept in "model_held". No Gateway; any hour on 16 Nov
    python reopen_week6.py                    status (client 18): the preconditions, the plan, per name what is filled,
                                              working and remaining, and base-ml's hold state
    python reopen_week6.py --go [--stale-ok]  (client 18) send each name's remainder: the snapshot's quantity less what
                                              has filled and what is working under its re-open refs. Refuses unless
                                              base-ml is held
    python reopen_week6.py --orders [--go]    (client 10) per saved order: whether its name's re-open is verified (filled
                                              in full per executions(all_clients=True), nothing working; never from
                                              positions()). With --go re-enter those whose name is verified (orders on names
                                              base-ml does not hold, CRWV, LSCC and USDE, at any time), leave the model one
                                              note of the actual results, mark the dealt-with originals as told, and release
                                              base-ml's hold once every name is verified and every order is confirmed
                                              working at IBKR. Refuses unless base-ml is held
    python reopen_week6.py --release [--go]   restore base-ml's model without the rest (the owner's judgement)

The checks. The ET date must be 2026-11-16 to send or write (--date is for tests only); nothing connects between 04:55
and 05:10 ET (the Gateway's restart). runs/live/HOLD must exist for --hold --go, and for --go before 09:00 ET (the loop
must run no look on the old lineup); from 09:00 ET --go does not need it. A mode on client 10 (--orders) first reads
runs/live/LOOP.json, which the live loop writes at its start: missing, or without "looking" 1 or 2, it refuses. Whether
the loop's pid is alive or not (a loop restarted inside a job's window runs that job at once), it refuses while any live
look marker exists: runs/live/LOOKING.<pid> (each process that runs a look writes its own, "<ISO time> <pid>", from the
look's start until after its post-look sync) or the plain runs/live/LOOKING of a loop started before per-process markers.
A marker whose pid is dead is stale; so is one more than 40 minutes old, unless it names the live loop's pid in LOOP.json
(a look of the loop is live whatever its age, unless written over 2 minutes before that loop started); stale ones are said so and
ignored. A pid counts as its writer's only while Windows says its process was created before the marker (or, for the loop,
before LOOP.json's "started"), with 2 minutes' margin: a process created later means the pid was reused, so that marker is
stale and that loop is reported as not running. When the creation time can't be read the pid alone decides, and the
message says a reused pid can't be ruled out. It also refuses while a job of the loop
is due but not yet in runs/live/done.json (a look from its time to 25 minutes after, the refresh 07:28-09:30, the record
16:15-18:15), and while runs/live/done.json can't be read (the loop rewrites it in place); --force overrides none of
these, nor the 3 minutes before each look. Unless --force it also refuses from 3
minutes before each look (:00 and :30, 09:00-15:30 ET) to 12 minutes after it, 07:28-08:15 (the refresh), 09:29-09:42
and 16:13-16:45 (the record), and at its start when one of those windows opens within 5 minutes (the reads before the
first place() take time). These are checked before connecting, again before each place() (a run stops at once, closes
the connection and says what was and wasn't re-entered; a rerun continues) and before the note is written: a Claude
sleeve's lane can run 25 minutes and then connect on client 10 to place and sync. Before each re-entry the script logs
a 'sending' row; a send with no runs/orders.jsonl row after it (a run killed during place(): a slow ack, a timeout,
Ctrl-C) is uncertain: an order of base-ml with its terms and an unrecorded ref at IBKR (orders() and all_open_orders()
on client 10) is adopted as it (the note gives its stop and trail as the 'sending' row has them); with none, it waits as
an 'Unknown' does (below) before it may be sent again, and the note tells it as unconfirmed. Ctrl-C at any point after
connecting closes the connection and prints a RESULT line of what was sent and written (exit code 2). The note is
recorded (a 'queuing' row) before it is queued in base-ml's pending notes and its 'noted' row comes after; a rerun (or
--release --go, which refuses while client 10 is blocked) finishes a recorded note with no 'noted' row, queuing it only
if it is not already pending. runs/sleeves.json must already be Week 6's (reset_week.py cancels every open
client-10 order whose ref maps to no carried sleeve). base-ml's ledger as of the snapshot's time must equal the
snapshot. Executions are read, then positions and account, then executions again; if the two differ the reads are
repeated, up to 3 times. Positions, per symbol: the account must hold exactly what it should (the re-open fills so far,
plus every fill since the snapshot under a current sleeve's ref, from executions(all_clients=True)), or, with
--stale-ok, that plus the snapshot's net for the symbol (a stale name still listed); the names that have dropped are
printed. Cash, less the net cash of every fill since the snapshot, must be at least --min-cash. A name to be shorted is
sent only when IBKR's shortable_shares cover it (stats([symbol]), one at a time; a row with an error is not sent). The time in force is chosen per order at
send time: MKT OPG before 09:25:00 ET, MKT DAY from 09:31:00 ET to 16:00, nothing in between. An order counts as placed
only when IBKR answers PreSubmitted, Submitted or Filled with no coded message outside the framework's harmless set
(its NOTICES, 2100-2199, 1101, and 399, 2109, 202); a code-less text doesn't decide, and a re-entered order is judged
again from a fresh orders() read 3 seconds after the last place(). An 'Unknown' is never resent in that run; it counts
as gone only when it was sent at least 5 minutes ago, two find() calls at least 3 minutes apart (two runs) were both
empty and all_open_orders() does not list it; until then that name or order waits. Any Gateway error ends the run with
a RESULT line and a non-zero exit code.

A saved TRAIL order would restart its anchor from Monday's price: it is re-entered with trailStopPrice set where the
trail would have stood at Friday's close (for a SELL, the trailing percent below the highest regular-hours trade from
the original order's placement, from 30-minute bars; for a BUY, above the lowest); IBKR's own stop (orders()'
"trail_stop") is read back, logged and told to the model.

Monday, in this order:
  1. Create runs/live/HOLD and run python reopen_week6.py --hold --go (base-ml held, so it has no look until its book is
     verified and its orders are back).
  2. Run reset_week.py with the Week 6 lineup (base-ml carried; its entry, hold included, is kept).
  3. On the TWS Gateway Owner's go (the broker framework's owner gives it once the Gateway shows the new cash): python
     reopen_week6.py, then --go (with --stale-ok if the owner says so) on client 18, before 09:25 ET for the opening
     auction; a rerun from 09:31 sends what did not fill.
  4. Once the fills are verified: python reopen_week6.py --orders, then --orders --go on client 10, outside the look
     windows; rerun until RESULT says done.
  5. The operator removes runs/live/HOLD by 09:00 ET at the latest, as soon as base-ml shows held (step 1), whether or
     not the re-open orders are sent (the other sleeves must get their 09:00 look; every RESULT line of --hold --go and
     --go says so); from 09:00 ET --go does not need HOLD. The script releases base-ml's hold when everything is
     confirmed.
"""
import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import paper

ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
DATE = "2026-11-16"
SLEEVE = "base-ml"
PREFIX = "reopen-20261116-"
SNAP = ROOT / "runs" / "live" / "reset_20261116_open_orders.json"
WEEK6 = ROOT / "runs" / "week6_sleeves.json"
HOLD_FILE = ROOT / "runs" / "live" / "HOLD"         # live.HOLD
LOOKING = ROOT / "runs" / "live" / "LOOKING"        # live.LOOKING: it and LOOKING.<pid> beside it mark a running look
LOOP = ROOT / "runs" / "live" / "LOOP.json"         # live.LOOP: {pid, code, looking, started} of the running loop
DONE_FILE = ROOT / "runs" / "live" / "done.json"    # live.DONE: [[date, job], ...] of jobs started or done
LOG = ROOT / "runs" / "live" / "reopen_20261116.jsonl"
HOLD_MODEL = "hold"                                 # live.HOLD_MODEL: a sleeve with this model gets no look
REOPEN_CLIENT = "trader-reopen"                     # client 18: the re-open's orders and reads
ORDERS_CLIENT = "trader-training"                   # client 10: base-ml's own orders (the live loop's client)
ORDERS_CLIENT_ID = 10
OLD_NAMED = {"c1", "c1-ml", "f", "f-ml", "h", "h-ml", "opus-ml", "sonnet-ml", "haiku-ml", "base-ml3"}
OK_STATUS = {"PreSubmitted", "Submitted", "Filled"}
NOTICES = {2104, 2106, 2107, 2108, 2158, 2119, 2100, 1102, 10167, 165, 492}   # tws.gateway.NOTICES
ORDER_NOTICES = {399, 2109, 202}                                               # tws.gateway.ORDER_NOTICES
DONE = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
DEAD = {"Cancelled", "ApiCancelled", "Inactive"}
LOOK_TIMES = [(h, m) for h in range(9, 16) for m in (0, 30)]
OPG_UNTIL, DAY_FROM, CLOSE = dtime(9, 25, 0), dtime(9, 31, 0), dtime(16, 0, 0)
GONE_AGE = timedelta(minutes=5)          # an 'Unknown' is gone only when sent at least this long ago ...
GONE_GAP = timedelta(minutes=3)          # ... and two empty find()s are at least this far apart
LOOK_WINDOW = timedelta(minutes=25)      # live.WINDOW: a look may start up to 25 minutes after its time
PRE_LOOK = timedelta(minutes=3)          # before each look: never used, even with --force
LEAD = timedelta(minutes=5)              # at start: refuse if a window opens within this
LOOKING_STALE = timedelta(minutes=40)    # a LOOKING older than this is stale (unless it is the live loop's)
STARTED_MARGIN = timedelta(minutes=2)    # the loop's marker is stale only when written this long before LOOP.json's "started"
SETTLE = 3.0                             # seconds after the last place() before orders() is re-read (status, trail_stop)
REMIND = ("remove runs/live/HOLD by 09:00 ET at the latest, as soon as base-ml shows held, whether or not the re-open "
          "orders are sent (the other sleeves must get their 09:00 look); from 09:00 ET --go does not need HOLD")


def now_et():
    return datetime.now(ET)


def connect(client):
    from tws import Gateway
    return Gateway(client, timeout=30)


def pid_alive(pid):
    """True if process `pid` exists (Windows: OpenProcess + GetExitCodeProcess == STILL_ACTIVE); never touches it."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True
        except OSError:
            return False
    import ctypes
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return k32.GetLastError() == 5                 # access denied: it exists
    try:
        code = ctypes.c_ulong()
        return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259   # STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


def pid_created(pid):
    """When process `pid` was created, as an aware datetime in ET (Windows: OpenProcess + GetProcessTimes), or None if it
    can't be read (no such process, access denied, not Windows); never touches it, never raises."""
    try:
        pid = int(pid)
        if pid <= 0 or os.name != "nt":
            return None
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)        # its own instance: pid_alive's prototypes stay as they are
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        k32.GetProcessTimes.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        k32.CloseHandle.restype = wintypes.BOOL
        h = k32.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return None
        try:
            made, ended, kern, user = (wintypes.FILETIME() for _ in range(4))
            if not k32.GetProcessTimes(h, ctypes.byref(made), ctypes.byref(ended), ctypes.byref(kern), ctypes.byref(user)):
                return None
            ticks = (made.dwHighDateTime << 32) | made.dwLowDateTime     # 100-ns ticks since 1601-01-01 UTC
            if not ticks:
                return None
            return (datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=ticks // 10)).astimezone(ET)
        finally:
            k32.CloseHandle(h)
    except Exception:                               # noqa: BLE001
        return None


NOW = now_et          # the test replaces these
GATEWAY = connect
PID_ALIVE = pid_alive
PID_CREATED = pid_created
SLEEP = time.sleep
SUFFIX = ""           # appended to every RESULT line (--hold --go and --go: the HOLD reminder)


class Refuse(Exception):
    pass


def out(*a):
    if a and SUFFIX and str(a[0]).startswith("RESULT:"):
        a = (" ".join(map(str, a)) + f" | {SUFFIX}",)
    print(*a, flush=True)


def short(text, n=400):
    return text if len(text) <= n else text[:n] + " ... (in full above)"


def iso_now():
    return NOW().astimezone(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- the plan

def load_snap():
    return json.loads(SNAP.read_text(encoding="utf-8"))


def fill_time(t):
    """IBKR execution time -> aware datetime (as paper.et_date reads it: a named zone, else the Gateway's old UTC+8)."""
    s = " ".join(str(t).split())
    parts = s.split(" ")
    if len(parts) >= 3:
        return datetime.strptime(parts[0] + " " + parts[1], "%Y%m%d %H:%M:%S").replace(tzinfo=ZoneInfo(parts[2]))
    return datetime.strptime(s, "%Y%m%d %H:%M:%S").replace(tzinfo=timezone(timedelta(hours=8)))


def book_at(sleeve, when):
    """{symbol: quantity} the sleeve's ledger held at `when` (aware): its fills up to then, with splits up to then."""
    day = when.astimezone(ET).strftime("%Y%m%d")
    splits = paper.read(paper.SPLITS)
    pos = {}
    for f in paper.fills():
        if f.get("sleeve") != sleeve or fill_time(f["time"]) > when:
            continue
        fd = paper.et_date(f["time"])
        k = 1
        for r in splits:
            if r["symbol"] == f["symbol"] and fd < r["ex_date"] <= day:
                k *= r["ratio"]
        pos[f["symbol"]] = pos.get(f["symbol"], 0) + k * (f["shares"] if f["side"] == "BOT" else -f["shares"])
    return {s: q for s, q in pos.items() if q}


def plan(snap):
    """The snapshot's base-ml quantities (always re-opened as saved) and its saved orders; refuses if base-ml's ledger as
    of the snapshot's time differs from the snapshot (fills after it, e.g. once base-ml trades again, don't matter)."""
    want = {s: int(q) for s, q in snap["books"][SLEEVE].items() if q}
    taken = datetime.fromisoformat(snap["taken"])
    ledger = {s: int(q) for s, q in book_at(SLEEVE, taken).items()}
    if ledger != want:
        diff = {s: (want.get(s, 0), ledger.get(s, 0)) for s in sorted(set(want) | set(ledger)) if want.get(s, 0) != ledger.get(s, 0)}
        raise Refuse(f"the snapshot and {SLEEVE}'s ledger at {snap['taken']} differ (snapshot, ledger): {diff}")
    return want, [o for o in snap["orders"] if o.get("sleeve") == SLEEVE]


def terms(o):
    """paper.place's arguments for a saved order row: the same type, prices, time in force and trailing percent."""
    typ = o["order_type"]
    fields = {k: v for k, v in (o.get("fields") or {}).items() if k == "trailingPercent"}
    limit = o.get("limit") if typ in ("LMT", "STP LMT", "LOC", "TRAIL LIMIT") else None   # IBKR reports a lmtPrice on stops
    aux = o.get("aux") if typ in ("STP", "STP LMT") or (typ.startswith("TRAIL") and "trailingPercent" not in fields) else None
    return dict(symbol=o["symbol"], action=o["action"], quantity=int(o["quantity"]), order_type=typ, limit=limit, aux=aux,
                tif=o.get("tif") or "GTC", **fields)


def show(o):
    t = terms(o)
    return (f"{t['action']} {t['quantity']} {t['symbol']} {t['order_type']}"
            + (f" limit {t['limit']:g}" if t["limit"] else "") + (f" stop {t['aux']:g}" if t["aux"] else "")
            + (f" trail {t['trailingPercent']:g}%" if "trailingPercent" in t else "") + f" {t['tif']}")


# ---------------------------------------------------------------- clock and preconditions

def at_time(now, h, m, s=0):
    return now.replace(hour=h, minute=m, second=s, microsecond=0)


def loop_state():
    """('alive' | 'dead' | 'missing' | 'old' | 'unreadable', LOOP.json's contents) of the live loop."""
    if not LOOP.exists():
        return "missing", None
    try:
        d = json.loads(LOOP.read_text(encoding="utf-8"))
    except Exception:                               # noqa: BLE001
        return "unreadable", None
    if d.get("looking") not in (1, 2):              # 1: the plain LOOKING (up to 23a7075); 2: LOOKING.<pid> per process
        return "old", d
    return ("alive" if PID_ALIVE(d.get("pid")) and not loop_reused(d) else "dead"), d


def loop_started(d):
    """LOOP.json's "started" as an aware datetime (ET if it names no zone), or None."""
    try:
        started = datetime.fromisoformat(str(d.get("started")))
    except ValueError:
        return None
    return started.replace(tzinfo=ET) if started.tzinfo is None else started


def loop_reused(d):
    """The creation time of the process that now has LOOP.json's pid, if it was created more than 2 minutes after the
    loop's "started" (the loop is gone and Windows gave its pid to another process), else None. None too when the time
    can't be read: then the pid alone decides, as before."""
    started = loop_started(d)
    made = PID_CREATED(d.get("pid")) if started is not None else None
    return made if made is not None and made > started + STARTED_MARGIN else None


def pending_jobs(now):
    """The loop's jobs whose window is open and that are not yet in done.json (live.py marks a look done as it starts, so
    a running look shows as LOOKING instead): looks from their time to 25 minutes after, the refresh 07:28-09:30, the
    record from 16:15 to 18:15."""
    done = {tuple(x) for x in json.loads(DONE_FILE.read_text(encoding="utf-8-sig"))} if DONE_FILE.exists() else set()   # -sig: a
    # hand edit saved by Windows PowerShell 5.1 carries a BOM (the TWS Gateway Owner, 15 Nov)
    day = now.date().isoformat()
    due = []
    for h, m in LOOK_TIMES:
        t = at_time(now, h, m)
        if t <= now < t + LOOK_WINDOW and (day, f"look-{h:02d}{m:02d}") not in done:
            due.append(f"look-{h:02d}{m:02d}")
    if at_time(now, 7, 28) <= now < at_time(now, 9, 30) and (day, "refresh") not in done:
        due.append("refresh")
    if at_time(now, 16, 15) <= now < at_time(now, 18, 15) and (day, "record") not in done:
        due.append("record")
    return due


def marker_files():
    """Every look marker beside LOOKING: the plain LOOKING (loops up to 23a7075 write it) and each LOOKING.<pid> (live.py
    since: each process its own file)."""
    try:
        return sorted(p for p in LOOKING.parent.iterdir() if p.name == LOOKING.name
                      or (p.name.startswith(LOOKING.name + ".") and p.name[len(LOOKING.name) + 1:].isdigit()))
    except OSError:
        return []


def marker_state(path, now, loop_pid, started):
    """None if the file is gone, else ('running' | 'stale', why) for one marker. Its content is "<ISO time> <pid>"
    (live.py). Stale when that pid is dead; a marker naming the live loop's pid (LOOP.json) is running whatever its age,
    unless its time is before the loop's "started" (a previous process with the same pid); any other marker more than 40
    minutes old is stale (a hard kill, a reboot and a reused pid). "Before" allows 2 minutes, for a clock set back after
    the loop started; the loop itself removes a marker of its own pid at its start and between looks. Stale too when its
    pid is alive but that process was created more than 2 minutes after the marker's time (PID_CREATED: the writer is
    gone and the pid was reused), and created no later than now (a creation time after now means the clock was set back
    since); when the creation time can't be read, or is after now, the other rules decide. Unreadable content: the
    file's own time, no pid."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except FileNotFoundError:                       # removed between the listing and the read: no marker
        return None
    except Exception:                               # noqa: BLE001
        text = ""
    parts = text.split()
    when = pid = None
    try:
        when = datetime.fromisoformat(parts[0])
        if when.tzinfo is None:
            when = when.replace(tzinfo=ET)
    except (IndexError, ValueError):
        try:
            when = datetime.fromtimestamp(path.stat().st_mtime, ET)     # unreadable content: the file's own time
        except FileNotFoundError:
            return None
        except Exception:                           # noqa: BLE001
            when = now
    try:
        pid = int(parts[1])
    except (IndexError, ValueError):
        pid = None
    age = now - when
    if pid is not None and not PID_ALIVE(pid):
        return "stale", f"{path} is stale: its pid {pid} is gone ({text[:60]!r}); ignored"
    made = PID_CREATED(pid) if pid is not None else None
    # A writer exists before its marker: the pid was reused. Not when the creation time is later than now: the clock was
    # set back after that process started, so its creation time and the marker's time can't be compared.
    if made is not None and when + STARTED_MARGIN < made <= now:
        return "stale", (f"{path} is stale: its pid {pid} now belongs to a process created at {made:%H:%M:%S}, after the "
                         f"marker was written ({text[:60]!r}): the writer is gone and the pid was reused; ignored")
    if pid is not None and pid == loop_pid:
        if started is not None and when < started - STARTED_MARGIN:     # the margin: a clock set back a little
            return "stale", (f"{path} is stale: written before the loop (pid {pid}) started at {started:%H:%M:%S} "
                             f"({text[:60]!r}): a previous process with the same pid; ignored")
    elif age > LOOKING_STALE:
        return "stale", f"{path} is stale: written {age.total_seconds() / 60:.0f} minutes ago ({text[:60]!r}); ignored"
    return "running", (f"{path} exists ({text[:60]!r}, pid {pid} alive"
                       + (", the live loop" if pid is not None and pid == loop_pid else "")
                       + f", {age.total_seconds() / 60:.0f} minutes old): a look is running (the loop removes it after "
                       "its post-look sync); rerun once it is gone (--force does not override this)"
                       + (f". The creation time of pid {pid} could not be read, so a reused pid can't be ruled out: if no "
                          "look is on, restart the loop (it removes stale markers)" if pid is not None and made is None else ""))


def looking_state(now):
    """None if there is no LOOKING marker, else ('running' | 'stale', why). Every marker file counts, the plain LOOKING and
    each LOOKING.<pid> alike: a look is running if any one is (marker_state); otherwise the stale ones are named, each
    with its reason, and ignored."""
    state, d = loop_state()
    loop_pid = started = None
    if state == "alive":
        try:
            loop_pid = int(d.get("pid"))
        except (TypeError, ValueError):
            loop_pid = None
        started = loop_started(d)
    found = [s for s in (marker_state(p, now, loop_pid, started) for p in marker_files()) if s]
    running = [why for st, why in found if st == "running"]
    if running:
        return "running", "; ".join(running)
    if found:
        return "stale", "; ".join(why for _, why in found)
    return None


def windows(now, force):
    """Client 10's clock windows as (start, end, what). With --force only the 3 minutes before each look."""
    w = [(at_time(now, h, m) - PRE_LOOK, at_time(now, h, m) if force else at_time(now, h, m) + timedelta(minutes=12),
          f"the {h:02d}:{m:02d} look" + (" (3 minutes before it; --force does not override this)" if force else
                                         " (3 minutes before to 12 after)")) for h, m in LOOK_TIMES]
    if not force:
        w += [(at_time(now, 7, 28), at_time(now, 8, 15), "the loop's 07:30 refresh (07:28-08:15)"),
              (at_time(now, 9, 29), at_time(now, 9, 42), "the 09:30 look (09:29-09:42)"),
              (at_time(now, 16, 13), at_time(now, 16, 45), "the 16:15 record job (16:13-16:45)")]
    return w


def client10_block(now, force, start=False):
    """(why client 10 must not be used now, or None; a note to print). Refuses unless runs/live/LOOP.json names a loop
    that writes LOOKING. Whether the loop's pid is alive or not (a loop restarted inside a job's window runs that job at
    once): never while a live LOOKING exists (a stale one, by its pid or age, is ignored and said so) or a job of the loop
    is due and not started; never in the 3 minutes before a look; and, unless --force, from 3 minutes before each look to
    12 after, 07:28-08:15, 09:29-09:42 and 16:13-16:45. At the start of a run (start=True) also when one of those windows
    opens within 5 minutes, since the reads before the first place() take time."""
    state, d = loop_state()
    if state == "missing":
        return (f"{LOOP} is missing: no way to tell whether a live loop that writes LOOKING is running; start the loop "
                "(it writes LOOP.json) or rerun once it has"), None
    if state == "unreadable":
        return f"{LOOP} can't be read; rerun in a moment", None
    if state == "old":
        return (f"{LOOP} has no \"looking\": that loop doesn't write LOOKING around its looks, so a look can't be seen; "
                "restart the loop on the current live.py"), None
    reused = loop_reused(d) if state == "dead" and PID_ALIVE(d.get("pid")) else None
    gone = (f"pid {d.get('pid')} now belongs to a process created at {reused:%H:%M:%S}, after the loop's start at "
            f"{loop_started(d):%H:%M:%S}" if reused else f"pid {d.get('pid')} is gone")
    unread = state == "alive" and PID_CREATED(d.get("pid")) is None
    note = (f"the live loop is running (pid {d.get('pid')}"
            + ("; its process's creation time could not be read, so a reused pid can't be ruled out" if unread else "") + ")"
            if state == "alive" else
            f"the live loop is not running ({gone}); its windows still apply, since a loop restarted "
            "inside a job's window runs that job at once")
    # The due jobs before the markers: the loop writes its marker and then marks the job done, so a look that starts
    # between these two reads is still due in the first, and one that started before it shows its marker in the second.
    try:
        due, due_err = pending_jobs(now), ""
    except (ValueError, OSError) as e:              # live.py's mark_done rewrites done.json in place: a half-written file
        due, due_err = None, f"{type(e).__name__}: {str(e)[:80]}"
    lk = looking_state(now)
    if lk and lk[0] == "running":
        return lk[1], note
    if lk:
        note += "; " + lk[1]
    if due is None:
        return (f"{DONE_FILE} can't be read ({due_err}): its due jobs can't be told. The loop may be writing it: rerun in a "
                f"moment; if it says this again, look at {DONE_FILE} itself (--force does not override this)"), note
    if due:
        return (f"the loop's {', '.join(due)} is due and has not started (not in done.json): it would connect on client 10 "
                "during this run; rerun once it has run (--force does not override this)"), note
    if force:
        note += "; --force: only the 3 minutes before each look apply of the clock windows"
    for st, en, what in windows(now, force):
        if st <= now < en:
            return f"it is {now:%H:%M} ET: {what} may hold client 10; rerun after {en:%H:%M}", note
    if start:
        nxt = sorted((st, en, what) for st, en, what in windows(now, force) if now < st <= now + LEAD)
        if nxt:
            st, en, what = nxt[0]
            return (f"it is {now:%H:%M} ET: {what} opens at {st:%H:%M}, within 5 minutes, before this run's reads and "
                    f"placements could finish; rerun after {en:%H:%M}"), note
    return None, note


def tif_at(now):
    """('OPG'|'DAY', None) or (None, why not now)."""
    t = now.time()
    if t < OPG_UNTIL:
        return "OPG", None
    if t < DAY_FROM:
        return None, (f"it is {now:%H:%M:%S} ET: too late for the opening auction (OPG until 09:25:00) and before 09:31:00, "
                      "when MKT DAY starts; rerun from 09:31:00 ET to send the rest")
    if t < CLOSE:
        return "DAY", None
    return None, f"it is {now:%H:%M:%S} ET, after the 16:00 close"


def in_restart(now):
    return (4, 55) <= (now.hour, now.minute) < (5, 10)


def wrong_date(now, date):
    if now.date().isoformat() != date:
        return f"the ET date is {now.date().isoformat()}, not {date}: this script runs on Monday {date} only"
    return None


def lineup_problem(snap):
    """None if runs/sleeves.json is Week 6's (base-ml plus the sleeves of runs/week6_sleeves.json, none of last week's)."""
    sl = paper.sleeves()
    w6 = json.loads(WEEK6.read_text(encoding="utf-8"))
    old = sorted(((set(snap["books"]) | OLD_NAMED) - set(w6) - {SLEEVE}) & set(sl))
    missing = [k for k in [SLEEVE, *w6] if k not in sl]
    early = [k for k in w6 if k in sl and str(sl[k].get("start") or "") < snap["taken"]]
    if not (old or missing or early):
        return None
    why = []
    if old:
        why.append(f"last week's sleeves are still there: {old}")
    if missing:
        why.append(f"missing: {missing}")
    if early:
        why.append(f"started before the snapshot (last week's entries): {early}")
    return ("runs/sleeves.json is not Week 6's yet (" + "; ".join(why) + "). reset_week.py must run first: it cancels every "
            "open client-10 order whose ref maps to no carried sleeve, so run after the re-open it would silently cancel "
            "the re-open orders")


def busy_message(now, client):
    if client != ORDERS_CLIENT:
        return (f"client {client} is in use (IBKR error 326), probably by another run of this script. Nothing was sent. "
                "Rerun once it has ended.")
    block, _ = client10_block(now, False)
    return ("client 10 is in use (IBKR error 326): the live loop (live.py) holds it during its 07:30 refresh (about 40 "
            "minutes) and during each look (every 30 minutes from 09:00 ET, about 5-10 minutes each). Nothing was sent. "
            + (f"Rerun later: {block}." if block else "Rerun in a few minutes."))


def open_gateway(now, client):
    try:
        return GATEWAY(client), None
    except Exception as e:                          # noqa: BLE001
        if getattr(e, "code", None) == 326 or str(e).startswith("326"):
            return None, busy_message(now, client)
        return None, f"could not connect to the Gateway as {client}: {type(e).__name__}: {e}"


def close_quietly(gw):
    try:
        getattr(gw, "close", lambda: None)()
    except Exception:                               # noqa: BLE001
        pass


# ---------------------------------------------------------------- what IBKR has under a ref

def code(m):
    head = str(m).split(":", 1)[0].strip()
    return int(head) if head.isdigit() else None


def harmless(c):
    """The framework's own rule (tws.gateway._harmless and ORDER_NOTICES): notices, 2100-2199, 1101, 399, 2109, 202."""
    return c in NOTICES or c in ORDER_NOTICES or 2100 <= c <= 2199 or c == 1101


def refusals(msgs):
    """Messages with an IBKR code that is not harmless. A code-less text (e.g. warningText) doesn't decide: the order's
    status does, and --orders re-reads it from orders() a few seconds later."""
    return [m for m in msgs if code(m) is not None and not harmless(code(m))]


def classify(res, error=None):
    """('placed' | 'uncertain' | 'failed', detail) for one place() answer."""
    if error:
        return "failed", f"not sent: {error}"
    res = res or {}
    st, msgs = res.get("status"), [str(m) for m in res.get("messages") or []]
    bad = refusals(msgs)
    if st == "Unknown" and not bad:
        return "uncertain", "no acknowledgement from IBKR (status Unknown)"
    if st in OK_STATUS and not bad:
        return "placed", st
    return "failed", f"IBKR answered {st}" + (f" ({'; '.join(msgs)})" if msgs else "")


def suffix(sym, ref):
    base = PREFIX + sym
    if ref == base:
        return 1
    if ref and ref.startswith(base + "-") and ref[len(base) + 1:].isdigit():
        return int(ref[len(base) + 1:])
    return None


def ref_for(sym, n):
    return PREFIX + sym if n == 1 else f"{PREFIX}{sym}-{n}"


def log_rows():
    return paper.read(LOG)


def log_append(row):
    paper.append(LOG, {"time": iso_now(), **row})


def signed(side_or_action, q):
    return q if side_or_action in ("BOT", "BUY") else -q


def okey(o):
    return o.get("perm_id") or (o.get("ref"), o.get("order_id"), o.get("client_id"))


def resolve(gw, ref, sent_at, now, record=True):
    """An uncertain ref, on the client that sent it: ('found', {orders, executions}) | ('waiting', why) | ('gone', why).
    Gone only when sent at least 5 minutes ago, two find()s at least 3 minutes apart (two runs) came back empty, and
    all_open_orders() doesn't list it. Matched by ref, never by order_id."""
    f = gw.find(ref) or {}
    if f.get("orders") or f.get("executions"):
        return "found", f
    listed = [o for o in gw.all_open_orders() if o.get("ref") == ref]
    if listed:
        return "found", {"orders": listed, "executions": []}
    firsts = [datetime.fromisoformat(r["time"]) for r in log_rows() if r.get("ref") == ref and r.get("find_empty")]
    if record:                                      # only a --go run records it (status changes nothing)
        log_append({"ref": ref, "find_empty": True})
    age = now - sent_at if sent_at else timedelta(0)
    if firsts and now - min(firsts) >= GONE_GAP and age >= GONE_AGE:
        return "gone", (f"sent {age.total_seconds() / 60:.0f} minutes ago with no acknowledgement; find() empty on two runs "
                        f"{(now - min(firsts)).total_seconds() / 60:.0f} minutes apart and not in all_open_orders()")
    need = []
    if age < GONE_AGE:
        need.append(f"sent {age.total_seconds() / 60:.1f} minutes ago (needs 5)")
    if not firsts:
        need.append("first empty find() " + ("recorded now; a second --go run at least 3 minutes later decides"
                                             if record else "(a --go run records it)"))
    elif now - min(firsts) < GONE_GAP:
        need.append(f"first empty find() {(now - min(firsts)).total_seconds() / 60:.1f} minutes ago (needs 3)")
    return "waiting", f"waiting on {ref}: " + "; ".join(need)


def name_state(gw, want, now, own, record=False):
    """Per name: done (signed shares filled under its re-open refs), working (signed unfilled shares of its open re-open
    orders), the next free suffix, the refs seen, and refs waiting on find(). Reads orders() first, then
    all_open_orders(), then executions(all_clients=True), so a fill landing between the reads is counted by one of them;
    per ref, done is the larger of its executions' sum and its order row's filled. `own` is True on the re-open's client
    (18): only there are its orders() and find() about these refs; on client 10 an uncertain ref is never judged gone,
    and its name waits. Returns (state per name, every execution since the snapshot's day, all clients)."""
    ords = {}
    if own:
        for o in gw.orders():
            if str(o.get("ref") or "").startswith(PREFIX):
                ords[okey(o)] = o
    for o in gw.all_open_orders():
        if str(o.get("ref") or "").startswith(PREFIX) and okey(o) not in ords:
            ords[okey(o)] = o
    all_execs = {e["exec_id"]: e for e in gw.executions(days=7, all_clients=True)}
    execs = {k: e for k, e in all_execs.items() if str(e.get("ref") or "").startswith(PREFIX)}
    last, sent_at = {}, {}
    for r in log_rows():
        if r.get("ref") and ("sending" in r or "outcome" in r):
            last[r["ref"]] = r
            if r.get("sending"):
                sent_at[r["ref"]] = datetime.fromisoformat(r["time"])
    seen = {o.get("ref") for o in ords.values()} | {e.get("ref") for e in execs.values()}
    waiting, gone = {}, {}
    for ref, r in last.items():
        if r.get("outcome") not in (None, "uncertain") or ref in seen:
            continue
        if not own:
            waiting[ref] = (f"waiting on {ref}: its answer was uncertain and client 10 cannot see client 18's orders; "
                            "the re-open's status run (python reopen_week6.py, client 18) decides")
            continue
        verdict, f = resolve(gw, ref, sent_at.get(ref), now, record)
        if verdict == "found":
            for e in f.get("executions") or []:
                execs[e["exec_id"]] = e
                all_execs[e["exec_id"]] = e
            for o in f.get("orders") or []:
                ords[okey(o)] = o
        elif verdict == "waiting":
            waiting[ref] = f
        else:
            gone[ref] = f
    st = {}
    for s in want:
        top = max([suffix(s, ref) or 0 for ref in last] + [0])
        refs = {}
        for o in ords.values():
            n = suffix(s, o.get("ref"))
            if n:
                top = max(top, n)
                refs.setdefault(o["ref"], {"orders": [], "exec": 0, "sign": signed(o.get("action"), 1)})["orders"].append(o)
        for e in execs.values():
            n = suffix(s, e.get("ref"))
            if n:
                top = max(top, n)
                r = refs.setdefault(e["ref"], {"orders": [], "exec": 0, "sign": signed(e.get("side"), 1)})
                r["exec"] += e.get("shares") or 0
        done = working = 0
        lines = []
        for ref, r in refs.items():
            filled = max([r["exec"]] + [o.get("filled") or 0 for o in r["orders"]])
            done += r["sign"] * filled
            status = [o.get("status") for o in r["orders"]]
            for o in r["orders"]:
                if o.get("status") not in DONE:
                    working += r["sign"] * max(0, (o.get("quantity") or 0) - filled)
            lines.append(f"{ref} {'/'.join(map(str, status)) or 'fills only'}")
        q, have = want[s], done + working
        st[s] = {"done": done, "working": working, "refs": lines, "next": top + 1, "remaining": q - have,
                 "anomaly": (have != 0 and (have > 0) != (q > 0)) or abs(have) > abs(q),
                 "waiting": [w for ref, w in waiting.items() if suffix(s, ref)],
                 "gone": [f"{ref}: {w}" for ref, w in gone.items() if suffix(s, ref)]}
    return st, list(all_execs.values())


def verified(st, want):
    return {s: st[s]["done"] == want[s] and st[s]["working"] == 0 and not st[s]["waiting"] for s in want}


def since(execs, taken):
    return [e for e in execs if fill_time(e["time"]) > taken]


def clean_positions(st, execs, taken):
    """What positions() must list without any stale name: the re-open fills so far, plus every fill since the snapshot under
    a current sleeve's ref on client 10 (base-ml's own tt- fills included), from IBKR's executions, not the ledgers."""
    exp = {s: v["done"] for s, v in st.items() if v["done"]}
    sleeves = paper.sleeves()
    for e in since(execs, taken):
        if paper.sleeve_of(e.get("ref")) in sleeves and e.get("client_id") in (None, ORDERS_CLIENT_ID):
            exp[e["symbol"]] = exp.get(e["symbol"], 0) + signed(e.get("side"), e.get("shares") or 0)
    return {s: q for s, q in exp.items() if q}


def stale_net(snap):
    net = {}
    for b in snap["books"].values():
        for s, q in b.items():
            net[s] = net.get(s, 0) + q
    return {s: q for s, q in net.items() if q}


def flat_check(have, clean, stale, stale_ok):
    """Per symbol: have == clean, or (a stale name still listed) have == clean + stale. -> (problems, dropped, listed)."""
    bad, dropped, listed, odd = [], [], [], {}
    for s in sorted(set(have) | set(clean) | set(stale)):
        h, c, x = have.get(s, 0), clean.get(s, 0), stale.get(s, 0)
        if h == c:
            if x:
                dropped.append(s)
        elif x and h == c + x:
            listed.append(s)
        else:
            odd[s] = (h, c) if not x else (h, c, c + x)
    if odd:
        bad.append(f"positions differ in {len(odd)} names (have, expected[, expected with the stale listing]): {odd}")
    if listed and not stale_ok:
        bad.append(f"{len(listed)} stale names are still listed ({listed[:12]}{' ...' if len(listed) > 12 else ''}): "
                   "rerun with --stale-ok only on the framework owner's word that IBKR removed them")
    return bad, dropped, listed


def cash_check(cash, execs, taken, min_cash):
    """Cash less the net cash of every fill since the snapshot (all clients): about the reset's cash."""
    flow = sum(-signed(e.get("side"), e.get("shares") or 0) * (e.get("price") or 0) for e in since(execs, taken))
    base = cash - flow
    out(f"cash {cash:,.2f}; less the net cash of {len(since(execs, taken))} fills since the snapshot: {base:,.2f}")
    return None if base >= min_cash else f"cash {base:,.0f} (fills taken out) is below {min_cash:,.0f}: the reset hasn't shown"


# ---------------------------------------------------------------- base-ml's hold (sleeves.json)

def write_json_atomic(path, data, indent=None):
    tmp = path.with_name(path.name + ".reopen.tmp")
    tmp.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    for i in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:                     # a reader has it open for a moment (Windows)
            if i == 19:
                raise
            time.sleep(0.25)


def hold_state():
    e = paper.sleeves().get(SLEEVE)
    if e is None:
        return "absent", None
    if e.get("model") == HOLD_MODEL:
        if "model_held" in e:
            return "held", e["model_held"]
        return "hold-unknown", None
    if "model_held" in e:
        return "inconsistent", e.get("model")
    return "free", e.get("model")


HOLD_SAYS = ("the loop's runs/live/HOLD comes off by 09:00 ET at the latest, as soon as base-ml shows held, whether or "
             "not the re-open orders are sent")


def set_hold():
    state, model = hold_state()
    if state == "held":
        return False, f"{SLEEVE} is already held (model '{HOLD_MODEL}', model_held {model!r}); sleeves.json unchanged"
    if state != "free":
        raise Refuse(f"{SLEEVE}'s sleeves.json entry is {state} (model {model!r}); not changing it")
    d = json.loads(paper.SLEEVES.read_text(encoding="utf-8"))
    e = d[SLEEVE]
    e["model_held"], e["model"] = e["model"], HOLD_MODEL
    write_json_atomic(paper.SLEEVES, d, indent=1)
    return True, (f"held {SLEEVE}: sleeves.json model {model!r} -> '{HOLD_MODEL}' (kept in model_held); it gets no look "
                  "until released; nothing else in sleeves.json changed")


def release_hold():
    state, model = hold_state()
    if state == "free":
        return False, f"{SLEEVE} is not held (model {model!r}); sleeves.json unchanged"
    if state != "held":
        return False, f"{SLEEVE}'s entry is {state}: not changed; set its model by hand"
    d = json.loads(paper.SLEEVES.read_text(encoding="utf-8"))
    e = d[SLEEVE]
    e["model"] = e.pop("model_held")
    write_json_atomic(paper.SLEEVES, d, indent=1)
    return True, (f"released {SLEEVE}: sleeves.json model '{HOLD_MODEL}' -> {e['model']!r} (model_held removed); its next "
                  "look runs as usual; nothing else in sleeves.json changed")


def hold_step(a):
    now = NOW()
    state, model = hold_state()
    out(f"{now:%a %Y-%m-%d %H:%M:%S} ET; {SLEEVE}'s hold: {state} (model {model!r}); {HOLD_SAYS}")
    if state == "held":
        out(f"RESULT: {SLEEVE} is held (model_held {model!r})")
        return 0
    if not a.go:
        out(f"RESULT: --hold --go would set {SLEEVE}'s model to '{HOLD_MODEL}'" if state == "free" else
            f"RESULT: --hold --go would refuse: the entry is {state}")
        return 0 if state == "free" else 1
    w = wrong_date(now, a.date)
    if w:
        out(f"RESULT: refused: {w}")
        return 1
    if not HOLD_FILE.exists():
        out(f"RESULT: refused: {HOLD_FILE} does not exist: create it first (step 1), so the loop runs no look while "
            "base-ml is being held and the lineup changes")
        return 1
    ok, msg = set_hold()
    out(msg)
    out(f"RESULT: {msg}")
    return 0


# ---------------------------------------------------------------- the re-open (client 18)

def preflight(a, now, sending, snap):
    bad = []
    if in_restart(now):
        bad.append(f"it is {now:%H:%M} ET, inside the Gateway's daily restart (04:55-05:10 ET); rerun after 05:10")
    if sending:
        w = wrong_date(now, a.date)
        if w:
            bad.append(w)
    lp = lineup_problem(snap)
    if lp:
        bad.append(lp)
    return bad


def guarded(fn, gw, *args):
    """Run one step against the Gateway: any error ends it with a RESULT line and exit code 1; a Ctrl-C the step does not
    report itself (its reads before any place()) with a RESULT line and exit code 2."""
    try:
        return fn(gw, *args)
    except KeyboardInterrupt:
        out(f"RESULT: incomplete: interrupted (Ctrl-C); connection closed; what was sent and written before it is in "
            f"{LOG.name} and orders.jsonl; a rerun continues (status first: python reopen_week6.py, or --orders)")
        return 2
    except Exception as e:                          # noqa: BLE001
        out(f"RESULT: stopped by an error ({type(e).__name__}: {str(e)[:300]}); what was sent before it is in "
            f"{LOG.name} and orders.jsonl; run the status (python reopen_week6.py, or --orders) before anything else")
        return 1
    finally:
        close_quietly(gw)


def reopen(a):
    now = NOW()
    snap = load_snap()
    want, _ = plan(snap)
    bad = preflight(a, now, a.go, snap)
    if a.go and now.time() < dtime(9, 0) and not HOLD_FILE.exists():
        bad.append(f"{HOLD_FILE} does not exist: before 09:00 ET the loop must be held, so it runs no look on the old lineup")
    if a.go:
        tif, why = tif_at(now)
        if tif is None:
            bad.append(why)
    hs = hold_state()
    out(f"{now:%a %Y-%m-%d %H:%M:%S} ET; client {REOPEN_CLIENT} (18); {SLEEVE}'s hold: {hs[0]} (model {hs[1]!r}); "
        f"{HOLD_SAYS}; snapshot taken {snap['taken']}")
    out(f"plan: re-open {len(want)} names as in the snapshot: {want}")
    if in_restart(now):
        out(f"RESULT: refused, not connecting: {bad[0]}")
        return 1
    gw, err = open_gateway(now, REOPEN_CLIENT)
    if gw is None:
        out(f"RESULT: refused: {err}")
        return 1
    return guarded(reopen_with, gw, a, snap, want, bad, hs[0])


def consistent_reads(gw, want, own, record, tries=3):
    """Executions, then positions and account, then executions again; if the two executions reads differ (fills landing
    during the reads, e.g. at 09:30), read again, up to 3 times. -> (state, executions, positions, cash, None | why)."""
    for i in range(tries):
        st, execs = name_state(gw, want, NOW(), own=own, record=record and i == 0)
        have = {p["symbol"]: p["quantity"] for p in gw.positions() if p.get("quantity")}
        cash = float((gw.account() or {}).get("TotalCashValue") or 0)
        again = {e["exec_id"] for e in gw.executions(days=7, all_clients=True)}
        if again == {e["exec_id"] for e in execs}:
            return st, execs, have, cash, None
        out(f"executions changed while positions and cash were read ({len(again)} against {len(execs)}); reading again")
    return st, execs, have, cash, f"executions changed during each of {tries} reads (fills are arriving); rerun in a minute"


def reopen_with(gw, a, snap, want, bad, held):
    taken = datetime.fromisoformat(snap["taken"])
    st, execs, have, cash, moving = consistent_reads(gw, want, own=True, record=a.go)
    if moving:
        bad.append(moving)
    ver = verified(st, want)
    out(f"{'name':6} {'want':>6} {'filled':>7} {'working':>8} {'remain':>7}  refs")
    for s, q in want.items():
        v = st[s]
        out(f"{s:6} {q:>6} {v['done']:>7} {v['working']:>8} {v['remaining']:>7}  {', '.join(v['refs']) or '-'}"
            + ("  VERIFIED" if ver[s] else "") + ("  OVERFILLED OR OPPOSITE: not sent" if v["anomaly"] else "")
            + "".join(f"  ({w})" for w in v["waiting"]) + "".join(f"  (gone: {g})" for g in v["gone"]))
    probs, dropped, listed = flat_check(have, clean_positions(st, execs, taken), stale_net(snap), a.stale_ok)
    bad += probs
    out(f"positions: {len(have)} listed; stale names still listed {len(listed)}; dropped {len(dropped)}"
        + (f": {dropped}" if dropped else ""))
    c = cash_check(cash, execs, taken, a.min_cash)
    if c:
        bad.append(c)
    complete = all(ver.values())
    if complete:
        out(f"the re-open is complete: all {len(want)} names filled in full under the re-open refs")
    elif held != "held":
        bad.append(f"{SLEEVE} is not held: run python reopen_week6.py --hold --go first (step 1), so its looks wait for "
                   "its book and orders")
    todo = {s: st[s]["remaining"] for s in want if st[s]["remaining"] and not st[s]["anomaly"] and not st[s]["waiting"]}
    blocked = {s: "; ".join(st[s]["waiting"]) for s in want if st[s]["waiting"]}
    blocked.update({s: f"filled plus working {st[s]['done'] + st[s]['working']} against {want[s]}" for s in want if st[s]["anomaly"]})
    shorts = [s for s, r in todo.items() if r < 0]
    no_borrow = {}
    if shorts:                                      # one symbol at a time: a look's quote chunks may hold up to 90 lines
        got = {}
        for sym in shorts:                          # a list of one: a bad name comes back in row["error"], not raised
            rows = gw.stats([sym])
            got[sym] = (rows if isinstance(rows, list) else [rows])[0] if rows else {}
        for s in shorts:
            sh = (got.get(s) or {}).get("shortable_shares")
            if (got.get(s) or {}).get("error"):
                no_borrow[s] = f"shortable shares unknown (IBKR: {str(got[s]['error'])[:120]})"
            elif sh is None or sh < -todo[s]:
                no_borrow[s] = f"shortable_shares {sh} < {-todo[s]}" if sh is not None else "shortable_shares unknown"
        out("shortable:", {s: (got.get(s) or {}).get("shortable_shares") for s in shorts})
    if not a.go:
        out("would send:", {s: f"{'BUY' if r > 0 else 'SELL'} {abs(r)} as {ref_for(s, st[s]['next'])}"
                            for s, r in todo.items() if s not in no_borrow} or "nothing")
        for s, why in {**no_borrow, **blocked}.items():
            out(f"  would not send {s}: {why}")
        if bad:
            out("NOT READY:", " | ".join(bad))
            out(f"RESULT: --go would refuse: {short(bad[0])}")
            return 1
        out(f"RESULT: checks pass; {len(todo) - len(no_borrow)} names to send, {sum(ver.values())} of {len(want)} verified; "
            + ("run --orders next" if complete else "rerun with --go to send" if todo else "wait for the working orders"))
        return 0 if not (no_borrow or blocked) else 2
    if complete:
        out("RESULT: nothing to send: the re-open is complete; run --orders (then --orders --go) on client 10")
        return 0
    if bad:
        out("NOT SENDING:", " | ".join(bad))
        out(f"RESULT: refused, nothing sent: {short(bad[0])}")
        return 1
    placed, failed, uncertain, unsent, answered = [], [], [], {}, set()
    stop = sending = cur = None
    try:                                            # a Ctrl-C ends as --orders' placing loop's does (N4)
        for s, r in todo.items():
            if s in no_borrow:
                unsent[s] = f"not sent: {no_borrow[s]}"
                continue
            tif, why = (None, stop) if stop else tif_at(NOW())
            if tif is None:
                stop = why
                unsent[s] = f"not sent: {why}"
                continue
            ref, act = ref_for(s, st[s]["next"]), "BUY" if r > 0 else "SELL"
            assert paper.sleeve_of(ref) is None
            log_append({"ref": ref, "symbol": s, "action": act, "quantity": abs(r), "tif": tif, "sending": True})
            sending, cur = f"{act} {abs(r)} {s} MKT {tif} [{ref}]", s
            try:
                res, error = gw.place(s, act, abs(r), order_type="MKT", tif=tif, ref=ref), None
            except Exception as e:                      # noqa: BLE001
                res, error = None, f"{type(e).__name__}: {str(e)[:200]}"
            outcome, detail = classify(res, error)
            line = f"{act} {abs(r)} {s} MKT {tif} [{ref}]"
            if outcome == "placed":                 # counted as soon as IBKR has answered, so a Ctrl-C after it counts it
                placed.append(line)
            elif outcome == "uncertain":
                uncertain.append(f"{line}: {detail}; not resent in this run")
            else:
                failed.append(f"{line}: {detail}")
            answered.add(s)
            log_append({"ref": ref, "symbol": s, "outcome": outcome, "detail": detail, "result": res})
            sending = cur = None
            out(f"  {outcome:9} {line} -> {detail}")
    except KeyboardInterrupt:
        close_quietly(gw)
        mid = sending if sending and cur not in answered else None
        unsent.update({s: "not sent: interrupted (Ctrl-C)" for s in todo if s not in answered and s not in unsent
                       and not (mid and s == cur)})
        unsent.update({s: f"not sent: {w}" for s, w in blocked.items()})
        out(f"STOPPED: interrupted (Ctrl-C)" + (f" during the send of {mid}: its send is logged with no answer, so a "
                                                "rerun finds it at IBKR by its ref, or waits on it as an 'Unknown'"
                                                if mid else "") + f"; connection closed. Placed {len(placed)}: "
            f"{placed}; failed {len(failed)}: {failed}; uncertain {len(uncertain)}: {uncertain}; not sent {len(unsent)}: "
            f"{unsent}")
        out(f"RESULT: incomplete: interrupted (Ctrl-C); {len(placed)} placed, {len(failed)} failed, {len(uncertain)} "
            f"uncertain" + (", 1 sent with no answer (uncertain)" if mid else "") + f", {len(unsent)} not "
            "sent; a rerun sends only each name's remainder (status first: python reopen_week6.py)")
        return 2
    unsent.update({s: f"not sent: {w}" for s, w in blocked.items()})
    out(f"placed {len(placed)}; failed {len(failed)}: {failed}; uncertain {len(uncertain)}: {uncertain}; "
        f"not sent {len(unsent)}: {unsent}")
    if failed or uncertain or unsent:
        out(f"RESULT: incomplete: {len(placed)} placed, {len(failed)} failed, {len(uncertain)} uncertain, {len(unsent)} not "
            "sent; a rerun sends only each name's remainder (status first: python reopen_week6.py)")
        return 2
    out(f"RESULT: {len(placed)} re-open orders placed on client 18. When they fill, run --orders, then --orders --go")
    return 0


# ---------------------------------------------------------------- the saved orders (client 10)

def same_terms(row, t):
    return all(row.get(k) == t[k] for k in ("symbol", "action", "quantity", "order_type", "limit", "aux", "tif"))


def attempts(saved, taken):
    """{perm_id of a saved order: [its re-entry rows in runs/orders.jsonl, oldest first]}: base-ml rows after the snapshot
    with the same terms (base-ml places nothing else while it is held)."""
    rows = [r for r in paper.read(paper.ORDERS) if r.get("sleeve") == SLEEVE and r.get("ref") and r.get("time", "") >= taken]
    return {o["perm_id"]: [r for r in rows if same_terms(r, terms(o))] for o in saved}


def price_terms(t):
    """The prices that define an order's terms at IBKR (a limit for limit types, a stop for stops)."""
    typ = t["order_type"]
    return (t.get("limit") if typ in ("LMT", "STP LMT", "LOC", "TRAIL LIMIT") else None,
            t.get("aux") if typ in ("STP", "STP LMT") else None)


def ibkr_matches(row, t):
    return (row.get("symbol") == t["symbol"] and row.get("action") == t["action"] and row.get("quantity") == t["quantity"]
            and row.get("order_type") == t["order_type"] and (row.get("tif") or "").upper() == t["tif"]
            and price_terms(row) == price_terms(t))


def orphan_sends(saved, att):
    """{perm_id: its last 'sending' row} for each saved order whose last send has no orders.jsonl row (nor adoption)
    at or after it: killed during place() (a slow ack, a timeout, Ctrl-C), so IBKR may hold an order of it whose ref
    nobody recorded."""
    rows = log_rows()
    out_ = {}
    for o in saved:
        sends = [r for r in rows if r.get("reenter") == o["perm_id"] and r.get("sending")]
        if not sends:
            continue
        last = sends[-1]
        covered = [r for r in att[o["perm_id"]] if r.get("time", "") >= last["time"]]
        covered += [r for r in rows if r.get("reenter") == o["perm_id"] and r.get("adopt") and r["time"] >= last["time"]]
        if not covered:
            out_[o["perm_id"]] = last
    return out_


def sent_fields(perm_id, until):
    """The trail fields (trailStopPrice, trailingPercent) of the saved order's last 'sending' row at or before `until`:
    what an order adopted from IBKR (sent with no record) was sent with; and the reason for its stop ("trail_why")."""
    sends = [r for r in log_rows() if r.get("reenter") == perm_id and r.get("sending") and r.get("time", "") <= until]
    t = (sends[-1].get("terms") or {}) if sends else {}
    out_ = {k: t[k] for k in ("trailStopPrice", "trailingPercent") if t.get(k) is not None}
    if sends and sends[-1].get("trail_why"):
        out_["trail_why"] = sends[-1]["trail_why"]
    return out_


def look_for_orphan(gw, o, send, known, now, record):
    """('adopted', row) | ('waiting', why) | ('gone', why) for a send with no record: IBKR's orders of base-ml on client 10
    (orders(): open and completed; all_open_orders()) with the same terms and a ref nobody recorded are the order it made;
    with none, the 'Unknown' rule: the send at least 5 minutes old and two empty look-ups at least 3 minutes apart."""
    t = terms(o)
    rows = [r for r in gw.orders() if paper.sleeve_of(r.get("ref")) == SLEEVE]
    rows += [r for r in gw.all_open_orders() if paper.sleeve_of(r.get("ref")) == SLEEVE
             and r.get("client_id") in (None, ORDERS_CLIENT_ID) and r.get("ref") not in {x.get("ref") for x in rows}]
    hits = [r for r in rows if r.get("ref") not in known and ibkr_matches(r, t)]
    if hits:
        r = hits[-1]
        if record:
            log_append({"reenter": o["perm_id"], "adopt": r["ref"], "status": r.get("status")})
        return "adopted", r
    sent = datetime.fromisoformat(send["time"])
    firsts = [datetime.fromisoformat(r["time"]) for r in log_rows() if r.get("reenter") == o["perm_id"]
              and r.get("lookup_empty") and r["time"] >= send["time"]]
    if record:
        log_append({"reenter": o["perm_id"], "lookup_empty": True})
    age = now - sent
    if firsts and now - min(firsts) >= GONE_GAP and age >= GONE_AGE:
        return "gone", (f"sent at {sent.astimezone(ET):%H:%M:%S} ET with no record; no order of these terms at IBKR on "
                        f"two look-ups {(now - min(firsts)).total_seconds() / 60:.0f} minutes apart")
    return "waiting", (f"waiting on the re-entry of {show(o)} sent at {sent.astimezone(ET):%H:%M:%S} ET with no record (the "
                       "run was stopped during place()): no order of these terms at IBKR yet; it is sent again only when "
                       "the send is 5 minutes old and two look-ups at least 3 minutes apart find nothing")


def judge(row):
    """An IBKR order row (client 10's own orders()) -> ('placed' | 'failed' | 'waiting', detail)."""
    st = row.get("status")
    bad = refusals([str(m) for m in row.get("messages") or []])
    if (row.get("filled") or 0) > 0 or (st in OK_STATUS and not bad):
        return "placed", f"IBKR lists it {st}" + (f", {row.get('filled')} filled" if row.get("filled") else "")
    if st in DEAD or bad:
        return "failed", f"IBKR later reported {st}" + (f" ({'; '.join(row.get('messages') or [])})" if row.get("messages") else "")
    return "waiting", f"waiting on {row.get('ref')}: IBKR lists it {st}"


def outcome_of(gw, row, ibkr, now, record):
    """One re-entry attempt (on client 10, so its own orders() and find() are the right reads): IBKR's current row decides
    when there is one (a cancelled or inactive one is not placed, whatever the first answer); an uncertain one with no
    row is resolved by the 'Unknown' rule."""
    oc, detail = classify(row.get("result"), row.get("error"))
    here = ibkr.get(row["ref"])
    if here:
        return judge(here)
    if oc != "uncertain":
        return oc, detail
    verdict, f = resolve(gw, row["ref"], datetime.fromisoformat(row["time"]), now, record)
    if verdict == "found":
        if f.get("executions"):
            return "placed", "filled per find()"
        return judge(f["orders"][0])
    return verdict, f


def trail_stop(gw, o, today):
    """(stop, detail) where a saved TRAIL would have stood at the last close, or (None, why not)."""
    t = terms(o)
    pct, amt = t.get("trailingPercent"), t.get("aux")
    rows = [r for r in paper.read(paper.ORDERS) if r.get("ref") == o.get("ref") and r.get("time")]
    if not rows:
        return None, f"its original {o.get('ref')} is not in runs/orders.jsonl, so its placement time is unknown"
    placed = datetime.fromisoformat(rows[0]["time"])
    try:
        bars = gw.bars(o["symbol"], duration="10 D", bar_size="30 mins")
    except Exception as e:                              # noqa: BLE001
        return None, f"bars failed: {type(e).__name__}: {str(e)[:120]}"
    use = []
    for b in bars or []:
        parts = str(b.get("date", "")).split()
        if len(parts) < 2:
            continue
        start = datetime.strptime(parts[0] + " " + parts[1], "%Y%m%d %H:%M:%S").replace(
            tzinfo=ZoneInfo(parts[2]) if len(parts) > 2 else ET)
        if start + timedelta(minutes=30) > placed and start.astimezone(ET).date() < today:
            use.append(b)
    if not use or (pct is None and amt is None):
        return None, "no regular-hours bars since its placement" if not use else "it has neither a percent nor an amount"
    if o["action"] == "SELL":
        ext = max(b["high"] for b in use)
        stop = ext * (1 - pct / 100) if pct is not None else ext - amt
        what = "highest"
    else:
        ext = min(b["low"] for b in use)
        stop = ext * (1 + pct / 100) if pct is not None else ext + amt
        what = "lowest"
    stop = round(stop, 2)
    side = "below" if o["action"] == "SELL" else "above"
    by = f"{pct:g}% {side}" if pct is not None else f"{amt:g} {side}"
    out(f"  {o['symbol']} TRAIL: {len(use)} 30-minute bars used, {use[0]['date']} to {use[-1]['date']}; {what} {ext:.2f}")
    return stop, (f"{by} {o['symbol']}'s {what} regular-hours trade ({ext:.2f}) from the original's placement "
                  f"({placed.astimezone(ET):%a %d %b %H:%M} ET) to the last close")


def trail_words(sent, why, ibkr_stop):
    if sent is None:
        return f"with its stop starting from the price when it was entered (the earlier stop level could not be restored: {why})"
    if ibkr_stop is None:
        return f"sent with its stop at {sent:.2f} ({why}); IBKR has not reported its stop level"
    return f"with IBKR's stop at {ibkr_stop:.2f} (sent as {sent:.2f}: {why})"


def note_text(results, waiting, ver_n, n_names, follow, n_saved):
    being = "were re-entered" if ver_n == n_names else "are being re-entered"
    head = (f"framework note, follow-up on your open orders after the weekend reset (Mon 16 Nov): now {ver_n} of {n_names} "
            f"names filled in full per IBKR; your positions {being} at the broker under refs reopen-20261116-<SYM>. "
            if follow else
            f"framework note (Mon 16 Nov): the paper account's cash was reset over the weekend, which removes the "
            f"positions held at the broker. Your {n_saved} open orders were cancelled after it, so that none could trigger "
            f"against a position the broker no longer held. Your sleeve's positions and cost basis are unchanged; your "
            f"positions {being} at the broker on Monday under refs reopen-20261116-<SYM>, which are not booked to "
            f"your sleeve: now {ver_n} of {n_names} names filled in full per IBKR. ")
    parts = []
    ok = [r for r in results if r["outcome"] == "placed"]
    if ok:
        parts.append("Your orders re-entered with the same terms under new refs: " + "; ".join(
            f"{show(r['o'])}" + (f" {r['trail']}" if r.get("trail") else "") + f" [{r['ref']}]" for r in ok) + ".")
    unc = [r for r in results if r["outcome"] in ("uncertain", "waiting") and r["ref"]]
    if unc:
        parts.append("Sent, unconfirmed (no acknowledgement from IBKR yet): " +
                     "; ".join(f"{show(r['o'])} [{r['ref']}]" for r in unc) + ".")
    cut = [r for r in results if r["outcome"] in ("uncertain", "waiting") and not r["ref"]]
    if cut:                                         # a send stopped inside place(): IBKR may or may not have received it
        parts.append("Sending interrupted, not known whether IBKR received it (its ref was not recorded): " +
                     "; ".join(show(r["o"]) for r in cut) + ".")
    lost = [r for r in results if r["outcome"] == "gone" and not r["ref"]]
    if lost:
        parts.append("Sending interrupted, not found at IBKR (its ref was not recorded): " +
                     "; ".join(f"{show(r['o'])}: {r['detail']}" for r in lost) + ".")
    ans = [r for r in results if r["outcome"] in ("failed", "gone") and r not in lost and not r["detail"].startswith("not sent")]
    if ans:
        parts.append("Sent, not working: " + "; ".join(f"{show(r['o'])} [{r['ref']}]: {r['detail']}" for r in ans) + ".")
    ns = [r for r in results if r["outcome"] == "failed" and r["detail"].startswith("not sent")]
    if ns:
        parts.append("Not sent: " + "; ".join(f"{show(r['o'])}: {r['detail'][len('not sent: '):]}" for r in ns) + ".")
    if waiting:
        parts.append("Not re-entered yet, because the re-open of that name is not yet filled in full: " +
                     "; ".join(show(o) for o in waiting) + ".")
    return head + " ".join(parts)


def orders_step(a):
    now = NOW()
    snap = load_snap()
    want, saved = plan(snap)
    bad = preflight(a, now, a.go, snap)
    hs = hold_state()
    if a.go and hs[0] != "held":
        bad.append(f"{SLEEVE} is not held: run python reopen_week6.py --hold --go first (step 1)")
    block, info = client10_block(now, a.force, start=True)
    if info:
        out(f"client 10: {info}")
    if block:
        bad.insert(0, block)
    out(f"{now:%a %Y-%m-%d %H:%M:%S} ET; client {ORDERS_CLIENT} (10); {SLEEVE}'s hold: {hs[0]} (model {hs[1]!r}); "
        f"{HOLD_SAYS}; {len(saved)} saved orders")
    if in_restart(now) or block or (a.go and bad):
        out("NOT DOING IT:", " | ".join(bad))
        out(f"RESULT: refused, not connecting on client 10, nothing sent or written: {short(bad[0])}")
        return 1
    gw, err = open_gateway(now, ORDERS_CLIENT)
    if gw is None:
        out(f"RESULT: refused: {err}")
        return 1
    return guarded(orders_with, gw, a, snap, want, saved, bad)


def orders_with(gw, a, snap, want, saved, bad):
    now = NOW()
    ibkr = {o.get("ref"): o for o in gw.orders() if paper.sleeve_of(o.get("ref")) == SLEEVE}   # client 10's own
    st, _ = name_state(gw, want, now, own=False)       # the re-open: all_open_orders and all-client executions only
    ver = verified(st, want)
    att = attempts(saved, snap["taken"])
    settled = []
    if a.go:
        settled = settle_notes()                    # a note recorded by a run stopped before its 'noted' row (N4)
    else:
        for q in unsettled_notes():
            out(f"a note recorded at {q['time']} has no 'noted' row (its run was stopped): --orders --go settles it, "
                "queuing it only if it is not already in base-ml's pending notes")
    noted = {}
    for r in log_rows():
        if isinstance(r.get("noted"), dict):
            noted.update(r["noted"])
        elif isinstance(r.get("noted"), list):
            noted.update({ref: "placed" for ref in r["noted"]})
    stops = {r["ref"]: r for r in log_rows() if r.get("ref") and "trail_stop" in r}
    today = now.astimezone(ET).date()
    known = {r.get("ref") for r in paper.read(paper.ORDERS) if r.get("ref")} | {r["adopt"] for r in log_rows() if r.get("adopt")}
    adopted = {}
    for r in log_rows():
        if r.get("adopt"):
            adopted.setdefault(r["reenter"], []).append(r)
    orphans = orphan_sends(saved, att)
    done, todo, waiting, prior = {}, [], [], []
    for o in saved:
        last = None
        rows = att[o["perm_id"]] + [{"ref": r["adopt"], "time": r["time"], "adopted": True,
                                     "result": {"status": r.get("status")}, "fields": sent_fields(o["perm_id"], r["time"])}
                                    for r in adopted.get(o["perm_id"], [])]
        for row in sorted(rows, key=lambda r: r["time"]):
            oc, detail = outcome_of(gw, row, ibkr, now, a.go)
            prior.append({"o": o, "ref": row["ref"], "outcome": oc, "detail": detail, "row": row})
            last = (oc, row["ref"], detail)
        if o["perm_id"] in orphans:
            send = orphans[o["perm_id"]]
            verdict, x = look_for_orphan(gw, o, send, known, now, a.go)
            if verdict == "adopted":            # its stop and trail as sent, from the 'sending' row (N3)
                row = {"ref": x["ref"], "time": send["time"], "adopted": True,
                       "result": {"status": x.get("status")}, "fields": sent_fields(o["perm_id"], send["time"])}
                ibkr.setdefault(x["ref"], x)
                oc, detail = judge(x)
                prior.append({"o": o, "ref": x["ref"], "outcome": oc, "detail": f"{detail} (adopted: sent with no record)",
                              "row": row, "seen": x})
                last = (oc, x["ref"], detail)
                out(f"  adopted {x['ref']} for {show(o)}: sent with no record, found at IBKR by its terms")
            elif verdict == "waiting":          # told as unconfirmed, as other waiting orders are (N8)
                prior.append({"o": o, "ref": None, "key": f"unrecorded {o['perm_id']} {send['time']}", "outcome": "waiting",
                              "detail": x, "row": {"time": send["time"], "fields": sent_fields(o["perm_id"], send["time"])}})
                last = ("waiting", None, x)
            else:                               # told as not found, under the same key as its unconfirmed telling
                prior.append({"o": o, "ref": None, "key": f"unrecorded {o['perm_id']} {send['time']}", "outcome": "gone",
                              "detail": x, "row": {"time": send["time"], "fields": sent_fields(o["perm_id"], send["time"])}})
                last = ("gone", None, x)
                out(f"  {show(o)}: {x}; it may be sent again")
        held_name = o["symbol"] in want
        if last and last[0] in ("placed", "waiting"):
            done[o["perm_id"]] = last
            state = f"re-entered as {last[1]}" if last[0] == "placed" else last[2]
            if last[1] is None:
                done[o["perm_id"]] = ("waiting", None, last[2])
        elif held_name and not ver[o["symbol"]]:
            waiting.append(o)
            v = st[o["symbol"]]
            state = (f"waiting: the re-open of {o['symbol']} is not verified (filled {v['done']} of {want[o['symbol']]}"
                     + (f", working {v['working']}" if v["working"] else "") + ")" + "".join(f"; {w}" for w in v["waiting"]))
        else:
            todo.append(o)
            state = ("to re-enter: its name's re-open is verified" if held_name else
                     f"to re-enter: {SLEEVE} holds no {o['symbol']} (an entry order)") + (f" (last try {last[0]})" if last else "")
        out(f"  {o.get('ref')} {show(o):40} {state}")
    out(f"verified {sum(ver.values())} of {len(want)} names: not yet {[s for s in want if not ver[s]]}")
    trails = {}
    for o in todo:
        if o["order_type"] == "TRAIL":
            trails[o["perm_id"]] = trail_stop(gw, o, today)
            sp, why = trails[o["perm_id"]]
            out(f"  {o['symbol']} TRAIL: " + (f"stop {sp:.2f}: {why}" if sp is not None else
                                              f"stop not restored ({why}); it would start from today's price"))
    if not a.go:
        if bad:
            out("NOT READY:", " | ".join(bad))
            out(f"RESULT: --orders --go would refuse: {short(bad[0])}")
            return 1
        out(f"RESULT: {len(todo)} to re-enter now, {len(waiting)} waiting for their re-open, {len(done)} already sent; "
            + ("rerun with --orders --go" if todo or [p for p in prior if noted.get(nkey(p)) != p["outcome"]] else "nothing to do"))
        return 0
    now_results, stopped, sending = [], None, None
    try:
        for o in todo:
            blk, _ = client10_block(NOW(), a.force)     # before every place: LOOKING, LOOP.json, due jobs, the windows
            if blk:
                stopped = blk
                break
            t = terms(o)
            sym, act, qty, typ = t.pop("symbol"), t.pop("action"), t.pop("quantity"), t.pop("order_type")
            lim, aux, tif = t.pop("limit"), t.pop("aux"), t.pop("tif")
            sent = why = None
            if typ == "TRAIL":
                sent, why = trails[o["perm_id"]]
                if sent is not None:
                    t["trailStopPrice"] = sent
            log_append({"reenter": o["perm_id"], "terms": {**terms(o), **t}, "sending": True,      # before the send (G3)
                        **({"trail_why": why} if typ == "TRAIL" else {})})
            sending = o
            row = paper.place(gw, SLEEVE, sym, act, qty, typ, lim, aux, tif, **t)
            sending = None
            oc, detail = classify(row.get("result"), row.get("error"))
            r = {"o": o, "ref": row.get("ref"), "outcome": oc, "detail": detail, "sent": sent, "why": why,
                 "result": row.get("result") or {}}
            now_results.append(r)
            out(f"  {oc:9} {show(o)} [{row.get('ref')}] -> {detail}")
    except KeyboardInterrupt:
        stopped = "interrupted (Ctrl-C)" + (f" during the place() of {show(sending)}: its send is recorded, so a rerun "
                                            "looks for it at IBKR (by its terms) before sending it again" if sending else "")
    if stopped:
        close_quietly(gw)
        sent = [f"{show(r['o'])} [{r['ref']}]: {r['outcome']}" for r in now_results]
        left = [show(o) + (" (sent, no record: uncertain)" if o is sending else "") for o in todo[len(now_results):]]
        wrote = ("; ".join(settled) + "; this run's own note and marks were not written (they wait for a rerun)" if settled
                 else "nothing written (the note and marks wait for a rerun)")
        out(f"STOPPED before the next place(): {stopped}; connection closed; {wrote}. Sent in this run: {sent or 'nothing'}. "
            f"Not sent: {left}")
        out(f"RESULT: incomplete: {stopped}; {len(sent)} re-entered in this run, {len(left)} not sent; "
            + (f"{'; '.join(settled)}; " if settled else "") + f"{SLEEVE} stays held; "
            "rerun --orders --go once client 10 is free (it continues)")
        return 2
    note = {}                                       # this run's note, once recorded: what a Ctrl-C's RESULT line reads back
    try:                                            # a Ctrl-C from here on ends as one in the placements does (N4)
        if now_results:                             # IBKR's later openOrder (status, trail_stop) needs a moment
            SLEEP(SETTLE)
        # re-read client 10's own orders: a cancelled or inactive one is not placed (N8); IBKR's trail stop is read back (N9)
        fresh = {o.get("ref"): o for o in gw.orders() if paper.sleeve_of(o.get("ref")) == SLEEVE}
        for r in now_results:
            here = fresh.get(r["ref"])
            if here:                                # IBKR's current row decides, whatever the first answer said
                oc2, d2 = judge(here)
                if not (oc2 == "failed" and r["outcome"] == "failed"):   # both failed: keep the answer's own messages
                    r["outcome"], r["detail"] = oc2, d2
            if r["o"]["order_type"] == "TRAIL":
                ts = (here or {}).get("trail_stop", r["result"].get("trail_stop"))
                log_append({"ref": r["ref"], "trail_stop": ts, "sent_trail_stop": r["sent"]})
                r["trail"] = trail_words(r["sent"], r["why"], ts)
                out(f"  {r['o']['symbol']} TRAIL [{r['ref']}]: IBKR's stop {ts}; sent {r['sent']}")
        for p in prior:
            here = fresh.get(p["ref"]) if p["ref"] else None
            if here:
                p["outcome"], p["detail"] = judge(here)
            if p["o"]["order_type"] == "TRAIL":
                fs = p["row"].get("fields") or {}
                why = fs.get("trail_why") or sent_fields(p["o"]["perm_id"], p["row"]["time"]).get("trail_why")
                ts = (here or p.get("seen") or {}).get("trail_stop", (stops.get(p["ref"]) or {}).get("trail_stop"))
                p["trail"] = trail_words(fs.get("trailStopPrice"), why or "as sent earlier", ts)
        told_as = lambda oc: "uncertain" if oc == "waiting" else oc
        told = [p for p in prior if told_as(noted.get(nkey(p), "")) != told_as(p["outcome"])]
        results = told + now_results
        follow = bool(noted)
        wrote = None
        if results:
            block, _ = client10_block(NOW(), a.force)
            if block:
                out(f"NOT WRITTEN: {block}; the note and marks wait for a rerun of --orders --go")
            else:
                text = note_text(results, waiting, sum(ver.values()), len(want), follow, len(saved))
                dealt = {str(r["o"]["perm_id"]) for r in results if r["outcome"] in ("placed", "failed", "gone")}
                note.update({"note_id": uuid.uuid4().hex, "queuing": True, "note_text": text,
                             "note_noted": {nkey(r): r["outcome"] for r in results if nkey(r)}, "note_dealt": sorted(dealt)})
                log_append(note)                    # recorded before it is queued: a rerun settles it (settle_notes)
                marked = finish_note(note, already=False)
                wrote = text
                out(f"marked as told (runs/looks.json reported_orders): {marked}")
                out("note for the model:", text)
        final = {}
        for o in saved:
            mine = [r for r in prior + now_results if r["o"]["perm_id"] == o["perm_id"]]
            final[o["perm_id"]] = mine[-1] if mine else None
        confirmed = {pid: bool(r and r["outcome"] == "placed" and r["ref"] in fresh and judge(fresh[r["ref"]])[0] == "placed")
                     for pid, r in final.items()}
        missing = [show(o) + ": " + (final[o["perm_id"]]["detail"] if final[o["perm_id"]] else "waiting for its re-open")
                   for o in saved if not confirmed[o["perm_id"]]]
        unver = [s for s in want if not ver[s]]
        if not missing and not unver and (wrote is not None or not results):
            ok, msg = release_hold()
            out(msg)
            out(f"RESULT: done: all {len(want)} names verified and all {len(saved)} orders confirmed working at IBKR; " +
                (f"{SLEEVE} released" if ok else msg))
            return 0
        out(f"still missing: names not verified {unver}; orders not confirmed: {missing}")
        out(f"RESULT: incomplete, {SLEEVE} stays held: {len(unver)} names not verified, {len(missing)} orders not confirmed"
            + ("; the note was not written (look window)" if results and wrote is None else "")
            + "; rerun --orders --go outside the look windows (or --release --go on the owner's judgement)")
        return 2
    except KeyboardInterrupt:
        close_quietly(gw)
        return stopped_after(now_results, note, settled)


def nkey(r):
    """What a 'noted' row keys an order's telling by: its ref, or for a send with no record yet its perm_id and send time."""
    return r.get("key") or r.get("ref")


def mark_told(dealt):
    """Add the dealt-with originals' perm_ids to base-ml's reported_orders in runs/looks.json (each once); -> those added."""
    state = json.loads(paper.LOOKS.read_text(encoding="utf-8")) if paper.LOOKS.exists() else {}
    s = state.setdefault(SLEEVE, {"reported_exec": [], "reported_orders": []})
    s.setdefault("reported_orders", [])
    new = sorted(set(map(str, dealt)) - set(map(str, s["reported_orders"])))
    s["reported_orders"] += new
    write_json_atomic(paper.LOOKS, state)
    return new


def finish_note(q, already):
    """A recorded note (its 'queuing' row, q): queue it unless `already` in base-ml's pending notes, mark its originals as
    told, then write its 'noted' row. -> the perm_ids newly marked."""
    if not already:
        paper.add_notes(SLEEVE, [q["note_text"]])
    marked = mark_told(q.get("note_dealt") or [])
    log_append({"noted": q.get("note_noted") or {}, "marked": marked, "note_id": q["note_id"]})
    return marked


def unsettled_notes():
    """'queuing' rows with no 'noted' row of the same note_id after them: a run stopped between recording a note and
    finishing it."""
    rows = log_rows()
    settled = {r.get("note_id") for r in rows if "noted" in r and r.get("note_id")}
    return [r for r in rows if r.get("queuing") and r.get("note_id") and r["note_id"] not in settled]


def settle_notes():
    """Finish each recorded note a stopped run left: queued now if it is not in base-ml's pending notes (paper.NOTES),
    left as it is if it is. base-ml is held throughout (--orders --go and --release --go settle first), so no look has
    taken its notes: a recorded note not pending was never queued."""
    finished = []
    for q in unsettled_notes():
        already = q["note_text"] in paper.take_notes(SLEEVE, commit=False)
        marked = finish_note(q, already)
        finished.append(f"the note recorded at {q['time']} by an earlier stopped run was finished at this run's start ("
                        + ("it was already pending" if already else "queued now") + ")")
        out(f"a note recorded at {q['time']} by a run that was stopped: " + (
            "already in base-ml's pending notes, not queued again" if already else
            "not in base-ml's pending notes, queued now") + f"; its record is complete (marked as told: {marked})")
    return finished


def stopped_after(now_results, note, settled=()):
    """The RESULT of a Ctrl-C after the placements (the connection already closed): what this run sent, and what it wrote,
    read back from the files (and an earlier run's note it finished at its start, settled)."""
    sent = [f"{show(r['o'])} [{r['ref']}]: {r['outcome']}" for r in now_results]
    rows = [r for r in log_rows() if note.get("note_id") and r.get("note_id") == note["note_id"]]
    if any("noted" in r for r in rows):
        wrote = "the note was queued and recorded"
    elif rows:
        queued = note["note_text"] in paper.take_notes(SLEEVE, commit=False)
        wrote = ("the note was recorded and " + ("queued" if queued else "not queued") + "; the rerun completes its record"
                 + (" without queuing it again" if queued else " and queues it"))
    else:
        wrote = "no note was written"
    if settled:
        wrote = "; ".join(settled) + "; this run's own note: " + wrote
    held = hold_state()[0]
    after = (f"{SLEEVE} stays held; rerun --orders --go once client 10 is free (it continues)" if held == "held" else
             f"{SLEEVE}'s hold: {held}")
    out(f"STOPPED after the placements: interrupted (Ctrl-C); connection closed. Sent in this run: {sent or 'nothing'}; "
        f"{wrote}")
    out(f"RESULT: incomplete: interrupted (Ctrl-C) after the placements; {len(sent)} re-entered in this run; {wrote}; "
        f"{after}")
    return 2


def release_step(a):
    state, model = hold_state()
    out(f"{SLEEVE}'s hold: {state} (model {model!r})")
    if not a.go:
        out("RESULT: " + (f"--release --go would restore model {model!r}" if state == "held" else "nothing to release"
                          if state == "free" else f"--release --go would refuse: the entry is {state}"))
        return 0 if state in ("held", "free") else 1
    if state == "held" and unsettled_notes():       # a note a stopped --orders --go recorded is finished while held (N4)
        block, _ = client10_block(NOW(), a.force)   # it writes notes.json and looks.json, which a look's lanes rewrite
        if block:
            out(f"RESULT: refused, {SLEEVE} stays held: a note recorded by a stopped --orders --go must be finished first, "
                f"and it can't be now: {short(block)}; rerun --release --go outside the look windows")
            return 1
        try:
            settle_notes()
        except KeyboardInterrupt:
            out(f"RESULT: incomplete: interrupted (Ctrl-C) while finishing a recorded note; {SLEEVE} stays held; rerun "
                "--release --go (it finishes the note without queuing it twice)")
            return 2
        except Exception as e:                      # noqa: BLE001
            out(f"RESULT: stopped by an error while finishing a recorded note ({type(e).__name__}: {str(e)[:300]}); "
                f"{SLEEVE} stays held; run python reopen_week6.py --orders before anything else")
            return 1
    ok, msg = release_hold()
    out(msg)
    out("RESULT: " + msg)
    return 0 if ok or state == "free" else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--go", action="store_true", help="do it; without it every mode only prints")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--hold", action="store_true", help="hold base-ml (no Gateway): step 1 on Monday")
    g.add_argument("--orders", action="store_true", help="verify the re-open per name and re-enter base-ml's saved orders")
    g.add_argument("--release", action="store_true", help="restore base-ml's model (the owner's judgement)")
    ap.add_argument("--stale-ok", action="store_true", help="names IBKR removed may still be listed (per symbol)")
    ap.add_argument("--force", action="store_true", help="client 10: connect inside the clock windows (never while a look "
                    "marker, runs/live/LOOKING or LOOKING.<pid>, is live, a job of the loop is due, or runs/live/done.json "
                    "can't be read, nor in the 3 minutes before a look)")
    ap.add_argument("--min-cash", type=float, default=990000.0)
    ap.add_argument("--date", default=DATE, help="for tests only: the ET date the script may send on")
    a = ap.parse_args(argv)
    global SUFFIX
    SUFFIX = REMIND if a.go and not (a.orders or a.release) else ""
    try:
        if a.hold:
            return hold_step(a)
        if a.release:
            return release_step(a)
        return orders_step(a) if a.orders else reopen(a)
    except Refuse as e:
        out(f"RESULT: refused: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
