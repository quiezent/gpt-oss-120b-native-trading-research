"""Score the forecast tool's outside names before the first look (Clayton, 9 Nov 2026: "a major issue"; at 13:00 UTC+8
the same day: a fix for every ML sleeve, Base + ML2.2 included). In week 4 the ML sleeves asked about 30-49 names a day
outside ML2's cached universe, which the tool scores on demand inside its time cap, and 13 of 280 ML looks hit that cap,
so 41 requests got no risk columns. This runs the look's own functions ahead of time, forecast._ondemand (ML2's rows)
and forecast3._outside_rows (ML3's gap1 inputs), for the close the day's looks use, so a look reads each name's file at
once. Same functions, same as_of, same files and ONDEMAND_V: nothing in ML2 or ML3 changes, and the values are what a
look would have scored.

    python warm_outside.py [--client trader-warm] [--until 08:55] [--max N] [--dry]

Names, in order, without ML2's cached universe or names already cached for the close:
  (a) names the sleeves' looks asked the forecast tool about in the last 10 sessions (their records in
      runs/live/<date>/*.json, gpt-oss and Claude alike, not dry runs), the most recently asked first (by session), then
      the most often asked. Not the tool's cache folders: this script writes those itself, and a name cut by the time
      cap was never saved;
  (b) the broad universe (data/bars_broad, broad_fetch.py; used to choose and order names only, the look's functions
      fetch fresh bars) with a real bar (volume and price above zero; IBKR pads thin names with zero-volume bars) at
      most 10 sessions before the close, a last price of at least $3 and a 20-session dollar volume of at least $10M,
      largest first.
No name starts after --until (ET) or after --max names; --dry lists the names and touches no gateway. One JSON line per
name in runs/live/warm_outside.jsonl (seconds: its batch's time over its names; kept: a look reads ML2's file at once),
then a summary line whose "stopped" says why it ended: done, max, deadline, "gateway down" (a batch the gateway failed
whole, still failed after 3 retries), "connect failed: ...", another warm-up holding the lock, or "not warming: ..."
(ML2's newest daily file isn't for the latest completed session). Exit 0, or 2 when the gateway failed it (connect
failed, gateway down). live.refresh() starts it once the forecasts are computed; the loop logs its summary when it ends.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import forecast
import forecast3

ROOT = Path(__file__).resolve().parent
BROAD = ROOT / "data" / "bars_broad"
LOOKS = ROOT / "runs" / "live"  # <YYYY-MM-DD>/<sleeve>-<HHMM>.json: each look's record, its forecast calls in "trace"
LOG = ROOT / "runs" / "live" / "warm_outside.jsonl"
LOCK = ROOT / "runs" / "live" / "warm_outside.lock"
ET = ZoneInfo("America/New_York")
RECENT = 10                     # sessions whose looks' forecast asks are warmed first
STALE = 10                      # a broad name's last real bar may be this many sessions before the close
BATCH = 10
ML2_S, ML3_S = 120.0, 60.0      # each batch's time for ML2's rows, then ML3's (the memo serves ML3 the bars ML2 fetched)
MIN_PRICE, MIN_ADV = 3.0, 10e6
RETRIES, WAIT_S = 3, 60.0       # a batch the gateway failed whole: tried again up to 3 times, about 60 s apart


def _now():
    return datetime.now(ET)


def _file(s):
    return f"{s.replace(' ', '_')}.json"


def _log(rec):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"t": f"{_now():%Y-%m-%d %H:%M:%S}", **rec}) + "\n")


def _err(e):
    return f"{type(e).__name__}: {str(e)[:120]}"


def latest_session(now):
    """The last completed session, YYYYMMDD: today after its close, else the session before today."""
    import live                                     # inside, as forecast._prev_session: live imports the tools
    if live.trading_day(now.date()) and now.time() >= live.close_time(now.date()):
        return now.strftime("%Y%m%d")
    return forecast._prev_session(now.date())


def sessions(as_of, n):
    """The n sessions through as_of (YYYYMMDD), oldest first."""
    import live
    d, out = datetime.strptime(as_of, "%Y%m%d").date(), []
    while len(out) < n:
        if live.trading_day(d):
            out.append(d.strftime("%Y%m%d"))
        d -= timedelta(days=1)
    return out[::-1]


def ml2_file():
    """(as_of, cached names) of ML2's newest daily file, the one forecast.live_provider serves; None without one."""
    files = sorted(forecast.OUT.glob("fc_*.json"))
    if not files:
        return None
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    return d["as_of"], set(d["forecasts"])


def kept(s, as_of):
    """ML2's file for this close at its ONDEMAND_V: a look reads it at once (forecast._ondemand)."""
    try:
        doc = json.loads((forecast.ONDEMAND / as_of / _file(s)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(doc, dict) and doc.get("v") == forecast.ONDEMAND_V


def cached(s, as_of):
    """Both versions have the name for this close: ML2's file (kept) and ML3's."""
    return kept(s, as_of) and (forecast3.ONDEMAND / as_of / _file(s)).exists()


def asked_in(step):
    """The names one forecast call looked up, read from its args as forecast.run reads them."""
    a = step.get("args")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except ValueError:
            return []
    syms = a.get("symbols") if isinstance(a, dict) else None
    syms = [syms] if isinstance(syms, str) else syms
    if not isinstance(syms, (list, tuple)):
        return []
    syms = list(dict.fromkeys(str(s).upper().strip() for s in syms if str(s).strip()))[:forecast.MAX_SYMBOLS]
    return [s for s in syms if forecast.SYM_OK.fullmatch(s)]


def recent(as_of):
    """Names the looks of the last RECENT sessions through as_of asked the forecast tool about: the most recently
    asked first (by session), then the most often asked, then by name. Read from the look records' research steps
    (a step {"tool": "forecast", "args": {"symbols": [...]}}, as gpt-oss and Claude looks both store them); dry runs
    are left out."""
    n, last = Counter(), {}
    for day in sessions(as_of, RECENT):
        for p in sorted((LOOKS / f"{day[:4]}-{day[4:6]}-{day[6:]}").glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(rec, dict) or rec.get("dry_run") or not isinstance(rec.get("trace"), list):
                continue
            for step in rec["trace"]:
                if isinstance(step, dict) and step.get("tool") == "forecast":
                    for s in asked_in(step):
                        n[s] += 1
                        last[s] = day
    return sorted(sorted(n), key=lambda s: (last[s], n[s]), reverse=True)


def _price(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def broad(as_of):
    """The broad universe's names whose last real bar through as_of (volume and close above zero) is at most STALE
    sessions before it, with that close >= MIN_PRICE and the last 20 real bars' average dollar volume >= MIN_ADV,
    largest dollar volume first."""
    oldest, out = sessions(as_of, STALE + 1)[0], []
    for p in BROAD.glob("*.json"):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            b = [x for x in doc["bars"] if str(x["date"])[:8] <= as_of and _price(x.get("close")) and _price(x.get("volume"))][-20:]
            if not b:
                continue
            adv = sum(x["close"] * x["volume"] for x in b) / 20
            if str(b[-1]["date"])[:8] >= oldest and b[-1]["close"] >= MIN_PRICE and adv >= MIN_ADV:
                out.append((-adv, doc.get("symbol") or p.stem.replace("_", " ")))
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
            continue                               # candidates.json, the fetch log, a part-written file
    return [s for _, s in sorted(out)]


def names(as_of, universe):
    """The names to warm, in order: recent asks, then the broad universe; none in ML2's universe or cached for as_of."""
    return [s for s in dict.fromkeys(recent(as_of) + broad(as_of)) if s not in universe and not cached(s, as_of)]


def _guarded(call, batch):
    """call(batch); if it raises, call([s]) for each name, so one name's error costs that name only.
    Returns [(names, result or the exception)]."""
    try:
        return [(batch, call(batch))]
    except Exception:                               # noqa: BLE001
        out = []
        for s in batch:
            try:
                out.append(([s], call([s])))
            except Exception as e:                  # noqa: BLE001
                out.append(([s], e))
        return out


class _Watched:
    """The gateway, keeping what each of its calls raised as (the call's symbol or None, the exception): the look's
    functions catch gateway errors per name and report them as "not covered"."""

    def __init__(self, gw):
        self.gw, self.errors = gw, []

    def __getattr__(self, name):
        f = getattr(self.gw, name)
        if not callable(f):
            return f

        def call(*a, **k):
            try:
                return f(*a, **k)
            except Exception as e:                  # noqa: BLE001 - kept, then raised as before
                s = a[0] if a else k.get("symbol", k.get("symbols"))
                self.errors.append((s if isinstance(s, str) else None, e))
                raise
        return call


def _down(e):
    """The gateway's failure rather than the name's: its connection (326 client ID in use, 502, 504, 1100), pacing (162
    'pacing', 420) or no reply in time."""
    c, m = getattr(e, "code", None), str(e).lower()
    return (isinstance(e, ConnectionError) or c in (326, 502, 504, 1100, 420) or (c == 162 and "pacing" in m)
            or "no reply from ib gateway" in m)


def _pause(e):
    """Errors that need a wait before the next try: 1100 (the Gateway lost IBKR), pacing (162 'pacing', 420)."""
    c = getattr(e, "code", None)
    return c in (1100, 420) or (c == 162 and "pacing" in str(e).lower())


def _try(gw, batch, as_of, end):
    """One try at a batch: ({name: ML2's result}, {names with an ML3 row}, [(symbol, error) the gateway raised])."""
    w, ml2, ml3 = _Watched(gw), {}, set()
    for b, res in _guarded(lambda b: forecast._ondemand(w, b, as_of, deadline=min(time.monotonic() + ML2_S, end)), batch):
        if isinstance(res, Exception):
            ml2.update({s: f"error ({type(res).__name__}: {str(res)[:80]})" for s in b})
        else:
            ml2.update({s: "ok" if s in res[0] else res[1].get(s, "no row") for s in b})
    for b, res in _guarded(lambda b: forecast3._outside_rows(w, b, as_of, budget=max(0.0, min(ML3_S, end - time.monotonic()))),
                           batch):
        ml3 |= set() if isinstance(res, Exception) else set(res)
    return ml2, ml3, w.errors


def warm(gw, todo, as_of, end):
    """Score todo in batches, none started at or after the monotonic `end`, which also caps every gateway wait. A batch
    the gateway failed whole (no ML2 row, and the connection dropped or every error was the gateway's) is tried again
    after a reconnect or a wait, up to RETRIES times, then the run stops ("gateway down").
    Returns (names done, ML2 rows, ML2 rows kept, ML3 rows, "done", "deadline" or "gateway down")."""
    n = ok2 = k2 = ok3 = 0
    for i in range(0, len(todo), BATCH):
        if end - time.monotonic() < 5:              # ML2 starts no name with under 5 s left
            return n, ok2, k2, ok3, "deadline"
        batch, t0 = todo[i:i + BATCH], time.monotonic()
        for k in range(RETRIES + 1):
            ml2, ml3, errs = _try(gw, batch, as_of, end)
            down = [e for _, e in errs if _down(e)]
            lost = not getattr(gw, "connected", True)
            outage = "ok" not in ml2.values() and (lost or (bool(down) and len(down) == len(errs)))
            if not outage or k == RETRIES or time.monotonic() >= end:
                break
            print(f"names {i + 1}-{i + len(batch)}: the gateway failed the whole batch "
                  f"({_err(down[-1]) if down else 'disconnected'}); try {k + 2} of {RETRIES + 1}", flush=True)
            back = False
            if lost:
                try:
                    gw.connect()
                    back = True
                except Exception as e:              # noqa: BLE001
                    print(f"reconnect failed: {_err(e)}", flush=True)
            if not back or any(_pause(e) for e in down):
                time.sleep(max(0.0, min(WAIT_S, end - time.monotonic())))
            if end - time.monotonic() < 5:          # no time left for a real retry: the failed try's verdict stands
                break
        own = {s: e for s, e in errs if s is not None and _down(e)}
        for s in batch:                             # the gateway's error, not ML2's "not covered"
            if ml2[s] != "ok" and (s in own or outage):
                e = own.get(s) or (down[-1] if down else None)
                ml2[s] = f"gateway error ({_err(e) if e else 'disconnected'})"
        secs, keep = (time.monotonic() - t0) / len(batch), {s: kept(s, as_of) for s in batch}
        for s in batch:
            _log({"symbol": s, "as_of": as_of, "ml2": ml2[s], "kept": keep[s], "ml3": s in ml3, "seconds": round(secs, 1)})
        n, ok2, k2, ok3 = (n + len(batch), ok2 + sum(v == "ok" for v in ml2.values()), k2 + sum(keep.values()),
                           ok3 + len(ml3))
        print(f"{n}/{len(todo)} names, {ok2} ML2 rows ({k2} kept), {ok3} ML3 rows, {_now():%H:%M:%S} ET", flush=True)
        if outage:
            return n, ok2, k2, ok3, "gateway down"
    return n, ok2, k2, ok3, "done"


def _connect(client):
    """The gateway on a client the TWS Gateway Owner registered (its clients.json); ValueError if it isn't one."""
    from tws import Gateway, load_clients
    if client not in load_clients():
        raise ValueError(f"client {client!r} is not registered in the TWS Gateway Owner's clients.json "
                         f"({', '.join(sorted(load_clients()))}); ask it for one")
    return Gateway(client, timeout=30)


def _lock():
    """An exclusive OS lock on LOCK, held until _unlock or the process ends (even if killed), as
    ml_research/macro_fetch.build_lock; None when another warm-up holds it."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    f = open(LOCK, "a+")
    try:
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except OSError:
        f.close()
        return None


def _unlock(f):
    try:
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    f.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--client", default="trader-warm")
    ap.add_argument("--until", default="08:55", type=dtime.fromisoformat, help="ET, HH:MM: no name starts after it")
    ap.add_argument("--max", type=int, help="names at most")
    ap.add_argument("--dry", action="store_true", help="list the names and touch no gateway")
    a = ap.parse_args(argv)
    now, t0 = _now(), time.monotonic()
    got, want = ml2_file(), latest_session(now)
    as_of, universe = got or (None, set())

    def summary(why, todo=None, n=0, ok2=0, k2=0, ok3=0):
        _log({"summary": True, "as_of": as_of, "client": a.client, "names_to_warm": None if todo is None else len(todo),
              "names": n, "ml2_rows": ok2, "ml2_kept": k2, "ml3_rows": ok3, "stopped": why,
              "seconds": round(time.monotonic() - t0)})
        print(f"stopped ({why}): {n} names, {ok2} ML2 rows ({k2} kept), {ok3} ML3 rows", flush=True)

    if as_of != want:
        why = (f"not warming: ML2's newest daily file is {'fc_' + as_of if as_of else 'missing'}, and the latest completed "
               f"session is {want}")
        print(f"{why}; the looks would not read these names for that close")
        if not a.dry:
            summary(why)
        return 0
    if a.dry:
        todo = names(as_of, universe)
        planned = todo[:a.max] if a.max is not None else todo
        print(f"{len(todo)} names to warm for the close of {as_of}" + (f", {len(planned)} with --max" if a.max is not None else ""))
        print("\n".join(planned))
        return 0
    lock = _lock()
    if lock is None:
        summary(f"another warm-up holds {LOCK.name}")
        return 0
    try:
        todo = names(as_of, universe)
        planned = todo[:a.max] if a.max is not None else todo
        end = t0 + (datetime.combine(now.date(), a.until, tzinfo=ET) - now).total_seconds()
        print(f"{len(todo)} names to warm for the close of {as_of}" + (f", {len(planned)} with --max" if a.max is not None else "")
              + f"; until {a.until:%H:%M} ET", flush=True)
        if time.monotonic() >= end or not planned:
            n, ok2, k2, ok3, why = 0, 0, 0, 0, "deadline" if planned else "done"
        else:
            try:
                gw = _connect(a.client)
            except Exception as e:                  # noqa: BLE001 - unregistered, 326 in use, 502, a handshake timeout
                print(f"connect failed: {type(e).__name__}: {e}", flush=True)
                summary(f"connect failed: {type(e).__name__}: {str(e)[:300]}", todo)
                return 2
            with gw:
                n, ok2, k2, ok3, why = warm(gw, planned, as_of, end)
        why = "max" if why == "done" and len(planned) < len(todo) else why
        summary(why, todo, n, ok2, k2, ok3)
        return 2 if why == "gateway down" else 0
    finally:
        _unlock(lock)


if __name__ == "__main__":
    sys.exit(main())
