"""Offline tests of warm_outside.py and of live.py's hooks for it: no gateway, no Tinker. ML2's and ML3's caches, the
look records, the broad bars, the lock and the log go to a temporary folder; vol2.live and the cache load are canned,
so forecast._ondemand and forecast3._outside_rows themselves run on a fake gateway.

    .venv\\Scripts\\python.exe warm_outside_test.py          # prints ALL OK and exits 0, else exits 1

Covers: the as_of check; the asked-about names read from the looks (gpt-oss and Claude records, the last 10 sessions,
not dry runs, not the cache folders), their order; the broad names (filler bars and bad prices dropped, a 10-session
staleness window, the floors); skipping ML2's universe and cached names; --dry, --max, --until; connect failures
(unregistered, 326, a handshake timeout) and the lock; one failing name; the batches' deadlines; a gateway that fails
one batch and recovers, drops and reconnects, or stays down; two consecutive closes, where warmed-only names don't
climb; that the files are the ones forecast._ondemand and forecast3._outside_rows write (and that a look then reads them
with no time left); and live.py: refresh's hook (after compute_versions only, never two at once, a Popen error
swallowed), the loop's poll of the warm-up and its summary, the look record's kelly_prompt and shock_cap, and the
loop's start line.
"""

import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
import time
import traceback
import types
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import forecast             # noqa: E402
import forecast3            # noqa: E402
import live                 # noqa: E402
import ml_tools             # noqa: E402
import paper                # noqa: E402
import warm_outside as W    # noqa: E402
from ml_research import vol2    # noqa: E402
from tws import GatewayError    # noqa: E402

ET = ZoneInfo("America/New_York")
AS_OF, NEXT = "20261106", "20261109"
MON = datetime(2026, 11, 9, 8, 20, tzinfo=ET)
TUE = datetime(2026, 11, 10, 8, 20, tzinfo=ET)
V = forecast.ONDEMAND_V
ROW = {"close": 50.0, "range1_pct": 3.0, "range1_lo": 2.0, "range1_hi": 4.5, "absr1_pct": 1.8, "vol20_pct": 40.0,
       "iv_pct": 41.0, "note": ""}
UNIVERSE = {"AAPL", "MSFT"}
ASKED = ["CRDO", "TCAP", "ARM", "ALAB", "BRK B", "APP"]
BROAD = ["BIG", "LAST1", "ETF1", "ZEROPX", "MID", "OLD10", "LATER", "BF B"]
WANT = ASKED + BROAD


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    print(f"ok  {name}")


def daily(n, end=AS_OF, price=50.0, vol=1e6):
    """n weekday bars through `end`."""
    d, out = date(int(end[:4]), int(end[4:6]), int(end[6:])), []
    while len(out) < n:
        if d.weekday() < 5:
            i = len(out)
            out.append({"date": d.strftime("%Y%m%d"), "open": price * (1 + 0.01 * math.sin(i)), "high": price * 1.02,
                        "low": price * 0.98, "close": price * (1 + 0.01 * math.cos(i)), "volume": vol})
        d -= timedelta(days=1)
    return out[::-1]


class GW:
    """The gateway calls forecast._ondemand and forecast3._outside_rows make, recorded."""

    def __init__(self, end=AS_OF):
        self.calls, self.end = [], end

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def details(self, s):
        self.calls.append(("details", s))
        return {"long_name": f"{s} INC", "stock_type": "COMMON"}

    def bars(self, symbol, duration="1 Y", bar_size="1 day", what="TRADES", rth=True, **kw):
        self.calls.append(("bars", symbol, duration, bar_size, what))
        return daily(300, self.end)


class DeadGW:
    def __getattr__(self, name):
        raise AssertionError(f"gateway touched: {name}")


def no_connect(client):
    raise AssertionError("connected")


def fake_vol2(gw, symbols=None, data=None, **kw):
    """vol2.live for one outside name: the daily bars request it makes, then a canned row (notes None crashes
    forecast._ondemand outside its own try, for the failing-name test)."""
    s = symbols[0]
    gw.bars(s, duration="1 Y", bar_size="1 day", what="ADJUSTED_LAST")
    return {"rows": {s: dict(ROW)}, "notes": None if s == "APP" else []}


def put(p, doc):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")


def ml2_doc(day, s, v=V):
    return {"v": v, "symbol": s, "as_of": day, "scored_et": "old", "long_name": f"{s} INC", "stock_type": "COMMON",
            "row": dict(ROW)}


def look(day, name, steps, claude=False, dry=False):
    """A look record as live.record_look writes it: a gpt-oss trace's steps carry step, tokens and said and end with
    submit; a Claude trace's carry tool, args and result and end with the CLI's record."""
    trace = []
    for i, (tool, args) in enumerate(steps):
        trace.append({"tool": tool, "args": args, "result": "..."} if claude else
                     {"step": i, "tool": tool, "args": args, "result": "...", "tokens": 10, "said": ""})
    trace.append({"cli_returncode": 0, "turns": len(steps)} if claude else {"step": len(steps), "tool": "submit", "tokens": 5, "said": ""})
    hhmm = name.rsplit("-", 1)[1]
    put(W.LOOKS / f"{day[:4]}-{day[4:6]}-{day[6:]}" / f"{name}.json",
        {"time": f"{day[:4]}-{day[4:6]}-{day[6:]}T{hhmm[:2]}:{hhmm[2:]}:00-05:00", "sleeve": name.rsplit("-", 1)[0],
         "model": "claude-sonnet-5" if claude else "base", "trace": trace, "dry_run": dry, "ml_tool": "ML3.0"})


def broad_file(s, bars, etf=False):
    put(W.BROAD / W._file(s), {"symbol": s, "meta": {"symbol": s, "etf": etf}, "bars": bars})


def setup(tmp):
    forecast.OUT = tmp / "forecast"
    forecast.ONDEMAND = forecast.OUT / "ondemand"
    forecast3.ONDEMAND = tmp / "forecast3" / "ondemand"
    forecast3.BARS = tmp / "bars"                   # empty: ML3 fetches outside names' bars, as for a broad name
    W.BROAD, W.LOG, W.LOOKS, W.LOCK = tmp / "bars_broad", tmp / "warm_outside.jsonl", tmp / "live", tmp / "warm.lock"
    W._now = lambda: MON
    forecast._data = lambda as_of: object()         # vol2.live is canned: no cache to load
    forecast._MEMO.clear()
    vol2.live = fake_vol2
    put(forecast.OUT / "fc_20261105.json", {"as_of": "20261105", "market": {}, "forecasts": {"AAPL": {}}})
    put(forecast.OUT / f"fc_{AS_OF}.json", {"as_of": AS_OF, "market": {}, "forecasts": {"AAPL": {}, "MSFT": {}}})
    # the looks: the last 10 sessions through 6 Nov are 26 Oct-6 Nov
    look("20261023", "base-ml-1000", [("forecast", {"symbols": ["OLDX"]})])             # the 11th session back
    look("20261101", "base-ml-1000", [("forecast", {"symbols": ["SUNX"]})])             # a Sunday: no session
    look("20261103", "base-ml-1000", [("forecast", {"symbols": "APP"})])                # a string, as forecast.run takes it
    look("20261104", "base-100k-ml-0930", [("stats", {"symbols": ["STATX"]}), ("forecast", {"symbols": ["alab ", "ARM", "AAPL"]}),
                                           ("forecast", {"symbols": 5})])
    look("20261105", "sonnet-ml-1000", [("tape", {}), ("forecast", {"symbols": ["ALAB", "ARM", "BRK B", "BRK/B"]}),
                                        ("forecast", '{"symbols": ["ARM"]}')], claude=True)
    look("20261106", "base-ml-0900", [("forecast", {"symbols": ["CRDO", "COIN", "TCAP"], "notional_usd": 30000})])
    look("20261106", "haiku-ml-0900", [("forecast", {"symbols": ["DRYX"]})], claude=True, dry=True)
    look(NEXT, "base-ml-0900", [("forecast", {"symbols": ["FUTR"]})])                   # after as_of
    put(W.LOOKS / "2026-11-06" / "broken-0930.json", "{")
    put(W.LOOKS / "2026-11-06" / "list-0930.json", [])
    put(W.LOOKS / "2026-11-06" / "notrace-0930.json", {"trace": "JUNK"})
    (W.LOOKS / "2026-11-06" / "haiku-ml-0900").mkdir(parents=True, exist_ok=True)      # a Claude look's CLI folder
    # the cache: TCAP was asked but cut by the time cap (no file); WARMX was only warmed, never asked
    put(forecast.ONDEMAND / "20261105" / "WARMX.json", ml2_doc("20261105", "WARMX"))
    put(forecast.ONDEMAND / AS_OF / "ALAB.json", ml2_doc(AS_OF, "ALAB"))           # ML2 kept, ML3 missing: warmed
    put(forecast.ONDEMAND / AS_OF / "COIN.json", ml2_doc(AS_OF, "COIN"))           # both kept: skipped
    put(forecast3.ONDEMAND / AS_OF / "COIN.json", {"symbol": "COIN", "as_of": AS_OF, "row": [0.1, 0.2]})
    put(forecast.ONDEMAND / AS_OF / "CRDO.json", ml2_doc(AS_OF, "CRDO", v=V - 1))  # another version's row: warmed
    put(forecast3.ONDEMAND / AS_OF / "CRDO.json", {"symbol": "CRDO", "as_of": AS_OF, "row": [0.1, 0.2]})
    # the broad universe
    for s, price, vol, end, etf in [("BIG", 50, 1e6, AS_OF, False), ("LAST1", 40, 1e6, "20261105", False),
                                    ("ETF1", 30, 1e6, AS_OF, True), ("MID", 20, 1e6, AS_OF, False),
                                    ("OLD10", 18, 1e6, "20261023", False), ("LATER", 15, 1e6, NEXT, False),
                                    ("BF B", 12, 1e6, AS_OF, False), ("ALAB", 100, 1e6, AS_OF, False),
                                    ("SMALL", 50, 1e5, AS_OF, False), ("PENNY", 2, 1e8, AS_OF, False),
                                    ("STALE", 40, 1e6, "20261022", False), ("MSFT", 400, 1e7, AS_OF, False)]:
        broad_file(s, daily(30, end, price, vol), etf)
    real = daily(30, "20261016", 30, 1e7)                      # real bars end 16 Oct; IBKR's filler bars run on to 6 Nov
    fill = [dict(b, open=30.0, high=30.0, low=30.0, close=30.0, volume=0) for b in daily(15, AS_OF)]
    broad_file("FILLER", real + fill)
    broad_file("ZEROPX", daily(30, "20261105", 25, 1e6) + [{"date": AS_OF, "open": 0, "high": 0, "low": 0, "close": 0,
                                                           "volume": 1e6}])
    neg = daily(30, AS_OF, 8, 1e6)                             # $8M a day, under the floor, but for one bad bar
    neg[-5] = dict(neg[-5], close=-1.0, volume=-5e8)           # whose price times volume would add $25M to the average
    broad_file("NEGPX", neg)
    put(W.BROAD / "candidates.json", {"rows": [{"symbol": "BIG"}]})
    put(W.BROAD / "BROKEN.json", "{")


def run_main(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = W.main(argv)
    return code, out.getvalue()


def logged():
    return [json.loads(x) for x in W.LOG.read_text(encoding="utf-8").splitlines()] if W.LOG.exists() else []


def strip_time(doc):
    return {k: v for k, v in doc.items() if k != "scored_et"}


def main():
    real = {p: sorted(str(x) for x in p.rglob("*")) if p.exists() else None     # not data/bars_broad: a fetch may
            for p in (forecast.OUT, forecast3.OUT)}                                # be adding to it
    tmp = Path(tempfile.mkdtemp(prefix="warm_outside_test_"))
    setup(tmp / "a")
    connect = W._connect

    # 1. the close: ML2's newest file, and the latest completed session
    check("ML2's newest daily file", W.ml2_file() == (AS_OF, UNIVERSE), W.ml2_file())
    check("Monday before the open: Friday's close", W.latest_session(MON) == AS_OF)
    check("Saturday: Friday's close", W.latest_session(datetime(2026, 11, 7, 10, 0, tzinfo=ET)) == AS_OF)
    check("Monday after the close: Monday's", W.latest_session(MON.replace(hour=16, minute=30)) == NEXT)
    check("sessions: the last 10 through the close", W.sessions(AS_OF, 10) == [
        "20261026", "20261027", "20261028", "20261029", "20261030", "20261102", "20261103", "20261104", "20261105", AS_OF])
    W._connect = no_connect
    W._now = lambda: TUE
    code, out = run_main(["--dry"])
    check("a stale file, --dry: refused, exit 0, no log", code == 0 and "not warming" in out and NEXT in out
          and not logged(), out)
    code, out = run_main([])
    last = logged()
    check("a stale file: refused, exit 0, no gateway, a summary line saying so", code == 0 and "not warming" in out
          and len(last) == 1 and last[0]["summary"] and last[0]["stopped"] == "not warming: ML2's newest daily file is "
          f"fc_{AS_OF}, and the latest completed session is {NEXT}" and last[0]["names"] == 0, (out, last))
    W.LOG.unlink()
    W._now = lambda: MON

    # 2. the names: asked ones from the looks (most recent session first, then most asked), then the broad universe
    rec = W.recent(AS_OF)
    check("asked: from the looks, most recent session first, then most asked",
          rec == ["COIN", "CRDO", "TCAP", "ARM", "ALAB", "BRK B", "AAPL", "APP"], rec)
    check("asked: gpt-oss and Claude records, a string symbol and JSON args read", {"APP", "BRK B", "CRDO"} <= set(rec))
    check("asked: not the cache folders, other tools, dry runs, a non-session, older or later looks, bad symbols",
          not {"WARMX", "STATX", "DRYX", "SUNX", "OLDX", "FUTR", "BRK/B", "JUNK"} & set(rec), rec)
    check("asked: a time-capped name (no file) still counts", "TCAP" in rec)
    b = W.broad(AS_OF)
    check("broad: dollar volume order, floors, a 10-session window", b == ["MSFT", "ALAB", "BIG", "LAST1", "ETF1", "ZEROPX",
                                                                          "MID", "OLD10", "LATER", "BF B"], b)
    check("broad: filler bars don't make a stale file fresh; a bad price is dropped, not the name",
          "FILLER" not in b and "ZEROPX" in b and "NEGPX" not in b and "STALE" not in b)
    got = W.names(AS_OF, UNIVERSE)
    check("order, filters and skips", got == WANT, got)
    check("cached: both versions' files at ONDEMAND_V", W.cached("COIN", AS_OF) and not W.cached("CRDO", AS_OF)
          and not W.cached("ALAB", AS_OF))

    # 3. --dry lists them and touches no gateway
    code, out = run_main(["--dry"])
    check("--dry: the names, no gateway, no log", code == 0 and out.split("\n")[1:-1] == WANT and not logged(), out)

    # 4. the real functions on a fake gateway, --max 4
    gw = GW()
    W._connect = lambda client: gw
    old_alab = json.loads((forecast.ONDEMAND / AS_OF / "ALAB.json").read_text(encoding="utf-8"))
    code, out = run_main(["--max", "4"])
    lines = logged()
    check("--max 4: four names, then a summary", code == 0 and [x.get("symbol") for x in lines[:-1]] == WANT[:4]
          and lines[-1]["summary"] and lines[-1]["stopped"] == "max" and lines[-1]["names"] == 4
          and lines[-1]["ml2_kept"] == 4 and lines[-1]["ml3_rows"] == 4, lines)
    check("each name: ML2 ok and kept, ML3 ok", all(x["ml2"] == "ok" and x["kept"] and x["ml3"] and x["as_of"] == AS_OF
                                                   and isinstance(x["seconds"], float) for x in lines[:-1]), lines)
    for s in WANT[:4]:
        doc = json.loads((forecast.ONDEMAND / AS_OF / W._file(s)).read_text(encoding="utf-8"))
        check(f"{s}: ML2's file at ONDEMAND_V, ML3's file", doc["v"] == V and doc["as_of"] == AS_OF
              and (forecast3.ONDEMAND / AS_OF / W._file(s)).exists())
    check("a kept ML2 row is read, not scored again", json.loads((forecast.ONDEMAND / AS_OF / "ALAB.json").read_text(
        encoding="utf-8")) == old_alab and ("details", "ALAB") not in gw.calls)
    yr = lambda s: sum(c == ("bars", s, "1 Y", "1 day", "ADJUSTED_LAST") for c in gw.calls)
    check("ML3 reuses the bars ML2 fetched (one request a name)", all(yr(s) == 1 for s in WANT[:4]), gw.calls)
    check("the next run skips them", W.names(AS_OF, UNIVERSE) == WANT[4:])
    check("the lock is released after a run", (lambda f: f is not None and (W._unlock(f) or True))(W._lock()))

    # the files are the ones the look's own functions write, and a look then reads them with no time left
    a2 = json.loads((forecast.ONDEMAND / AS_OF / "ARM.json").read_text(encoding="utf-8"))
    a3 = json.loads((forecast3.ONDEMAND / AS_OF / "ARM.json").read_text(encoding="utf-8"))
    rows, why = forecast._ondemand(DeadGW(), ["ARM"], AS_OF, time.monotonic() - 1)
    check("a look reads ML2's row at once", "ARM" in rows and not why and rows["ARM"]["close"] == ROW["close"], (rows, why))
    check("a look reads ML3's row at once", forecast3._outside_rows(DeadGW(), ["ARM"], AS_OF, 0) == {"ARM": a3["row"]})
    (forecast.ONDEMAND / AS_OF / "ARM.json").unlink()
    (forecast3.ONDEMAND / AS_OF / "ARM.json").unlink()
    forecast._MEMO.clear()
    forecast._ondemand(GW(), ["ARM"], AS_OF, time.monotonic() + 60)
    forecast3._outside_rows(GW(), ["ARM"], AS_OF, 30)
    b2 = json.loads((forecast.ONDEMAND / AS_OF / "ARM.json").read_text(encoding="utf-8"))
    b3 = json.loads((forecast3.ONDEMAND / AS_OF / "ARM.json").read_text(encoding="utf-8"))
    check("ML2's file is what forecast._ondemand writes", strip_time(a2) == strip_time(b2), (a2, b2))
    check("ML3's file is what forecast3._outside_rows writes", a3 == b3, (a3, b3))

    # 5. one failing name costs that name only
    W.LOG.unlink()
    code, out = run_main(["--max", "3"])
    lines = logged()
    by = {x["symbol"]: x for x in lines if "symbol" in x}
    check("APP fails, BRK B and BIG are scored", code == 0 and list(by) == ["BRK B", "APP", "BIG"]
          and by["APP"]["ml2"].startswith("error (TypeError") and not by["APP"]["kept"] and by["BRK B"]["ml2"] == "ok"
          and by["BIG"]["ml2"] == "ok" and by["BIG"]["kept"] and lines[-1]["ml2_rows"] == 2 and lines[-1]["ml2_kept"] == 2,
          lines)

    # 6. --until: nothing starts after it, and no gateway when it has passed
    W.LOG.unlink()
    W._connect = no_connect
    code, out = run_main(["--until", "08:20"])
    lines = logged()
    check("--until passed: no name, no gateway", code == 0 and len(lines) == 1 and lines[0]["stopped"] == "deadline"
          and lines[0]["names"] == 0, lines)

    # 7. connect failures: printed, a summary line, exit 2; the default client is registered
    W._connect = connect
    code, out = run_main(["--client", "nobody-registered"])
    last = logged()[-1]
    check("unknown client: exit 2, why, and a summary", code == 2 and "not registered" in out and "ask it for one" in out
          and last["summary"] and last["stopped"].startswith("connect failed: ValueError: client 'nobody-registered'")
          and last["names"] == 0 and last["names_to_warm"] == len(WANT) - 6, (out, last))
    for err in (GatewayError("Unable to connect as the client id is already in use. Retry with a unique client id.", 326),
                GatewayError("IB Gateway at 127.0.0.1:4002 did not finish the API handshake within 30s"),
                TimeoutError("timed out")):
        W._connect = lambda client, e=err: (_ for _ in ()).throw(e)
        code, out = run_main([])
        last = logged()[-1]
        check(f"connect fails ({type(err).__name__}: {str(err)[:30]}): exit 2, the summary says so",
              code == 2 and last["stopped"] == f"connect failed: {type(err).__name__}: {err}" and str(err) in out, (out, last))
    from tws import load_clients
    check("the default client is registered", "trader-warm" in load_clients(), sorted(load_clients()))

    # the lock: a second warm-up exits cleanly with a summary line, and touches no gateway
    held = W._lock()
    W._connect = no_connect
    code, out = run_main(["--max", "1"])
    last = logged()[-1]
    check("a second warm-up: exit 0, a summary line, no gateway", held is not None and code == 0 and last["summary"]
          and last["stopped"] == f"another warm-up holds {W.LOCK.name}" and last["names"] == 0, (out, last))
    W._unlock(held)
    W._connect = lambda client: GW()
    code, out = run_main(["--max", "1"])
    check("after it ends, the next one runs", code == 0 and logged()[-1]["stopped"] == "max", logged()[-1])

    # 8. batches, deadlines, per-name errors and gateway outages, on a fake clock
    clock, sleeps = {"t": 0.0}, []
    calls2, calls3 = [], []

    def ondemand(gw, b, as_of, deadline):
        calls2.append((list(b), as_of, deadline))
        clock["t"] += 70
        if "BAD" in b:
            raise RuntimeError("one name's crash")
        return {s: dict(ROW) for s in b}, {}

    def outside(gw, b, as_of, budget):
        calls3.append((list(b), as_of, budget))
        if "BAD" in b:
            raise RuntimeError("one name's crash")
        return {s: [0.1, 0.2] for s in b}

    def sleep(s):
        sleeps.append(s)
        clock["t"] += s

    saved = (forecast._ondemand, forecast3._outside_rows, W.time)
    forecast._ondemand, forecast3._outside_rows = ondemand, outside
    W.time = types.SimpleNamespace(monotonic=lambda: clock["t"], sleep=sleep)
    try:
        todo = [f"N{i}" for i in range(25)]
        n, ok2, k2, ok3, why = W.warm(GW(), todo, AS_OF, 100.0)
        check("deadline: no batch starts at or after it", (n, why) == (20, "deadline") and len(calls2) == 2, (n, why, calls2))
        check("batches of 10, the close as_of", [c[0] for c in calls2] == [todo[:10], todo[10:20]]
              and all(c[1] == AS_OF for c in calls2 + calls3))
        check("ML2's deadline: 120 s, capped at the end", [c[2] for c in calls2] == [100.0, 100.0], calls2)
        check("ML3's budget: 60 s, capped at the end", [c[2] for c in calls3] == [30.0, 0.0], calls3)
        clock["t"], calls2[:], calls3[:] = 0.0, [], []
        n, ok2, k2, ok3, why = W.warm(GW(), ["A", "BAD", "C"], AS_OF, 1e9)
        check("a crash: the batch again name by name", (n, ok2, ok3, why) == (3, 2, 2, "done")
              and [c[0] for c in calls2] == [["A", "BAD", "C"], ["A"], ["BAD"], ["C"]] and not sleeps, calls2)

        # the look's functions catch a gateway error per name ("not covered"); the warm-up sees it through the gateway
        class Flaky:
            """bars() raises `err` for the next `fails` calls, and while disconnected; connect() reconnects."""

            def __init__(self, fails, err, connected=True, per_name=None):
                self.fails, self.err, self.connected, self.per = fails, err, connected, per_name or {}
                self.calls, self.connects = [], 0

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def bars(self, s, **kw):
                self.calls.append(s)
                clock["t"] += 1
                if s in self.per:
                    raise self.per[s]
                if self.fails > 0 or not self.connected:
                    self.fails -= 1
                    raise self.err
                return daily(70)

            def connect(self):
                self.connects += 1
                self.connected = True

        def ondemand_gw(gw, b, as_of, deadline):
            rows, why = {}, {}
            for s in b:
                if deadline - clock["t"] < 5:          # as the real one: no name started with under 5 s left
                    why[s] = "risk columns skipped (time cap)"
                    continue
                try:
                    gw.bars(s, duration="1 Y", bar_size="1 day", what="ADJUSTED_LAST")
                    rows[s] = dict(ROW)
                except Exception:                   # noqa: BLE001 - as vol2.live: a note, then "not covered"
                    why[s] = "not covered (not in the cache and no gateway bars)"
            return rows, why

        def outside_gw(gw, b, as_of, budget):
            out = {}
            for s in b:
                try:
                    gw.bars(s, duration="1 Y", bar_size="1 day", what="ADJUSTED_LAST")
                    out[s] = [0.1, 0.2]
                except Exception:                   # noqa: BLE001
                    pass
            return out

        forecast._ondemand, forecast3._outside_rows = ondemand_gw, outside_gw
        lost = GatewayError("Connectivity between IBKR and Trader Workstation has been lost.", 1100)
        todo = [f"N{i}" for i in range(15)]

        def run_warm(g, names=todo, end=1e9):
            W.LOG.unlink(missing_ok=True)
            clock["t"], sleeps[:] = 0.0, []
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                res = W.warm(g, names, AS_OF, end)
            return res, {x["symbol"]: x["ml2"] for x in logged()}, out.getvalue()

        g = Flaky(20, lost)                         # the first try of batch 1: its 10 ML2 and 10 ML3 requests
        (n, ok2, k2, ok3, why), ml2, out = run_warm(g)
        check("1100 on a whole batch: about 60 s, the same batch again, then on", (n, ok2, ok3, why) == (15, 15, 15, "done")
              and sleeps == [60.0] and g.calls[:20] == g.calls[20:40] == todo[:10] + todo[:10]
              and set(ml2.values()) == {"ok"} and "try 2 of 4" in out and "1100" in out, (n, ok2, ok3, why, sleeps, out))
        g = Flaky(0, GatewayError("Not connected to IB Gateway (it restarts daily at 06:00 ET); call connect() to reconnect",
                                  504), connected=False)
        (n, ok2, k2, ok3, why), ml2, out = run_warm(g)
        check("dropped: reconnect and the same batch again, no wait", (n, ok2, why) == (15, 15, "done") and g.connects == 1
              and not sleeps and set(ml2.values()) == {"ok"}, (n, ok2, why, g.connects, sleeps))
        for err, wait in ((GatewayError("Historical Market Data Service error message:API historical data query cancelled: "
                                        "pacing violation", 162), True),
                          (GatewayError("Invalid real-time query: pacing", 420), True)):
            g = Flaky(20, err)
            (n, ok2, k2, ok3, why), ml2, out = run_warm(g)
            check(f"{err.code} pacing: about 60 s, then the batch again", why == "done" and sleeps == [60.0] and ok2 == 15,
                  (why, sleeps, ok2))
        g = Flaky(10 ** 6, GatewayError("No reply from IB Gateway to the historical bars request within 20s"))
        (n, ok2, k2, ok3, why), ml2, out = run_warm(g)
        check("stays down: 3 more tries, then 'gateway down', not 'done'", (n, ok2, ok3, why) == (10, 0, 0, "gateway down")
              and len(g.calls) == 4 * 20 and set(g.calls) == set(todo[:10]) and sleeps == [60.0] * 3, (n, ok2, why, sleeps))
        check("stays down: the gateway's error logged, not 'not covered'", list(ml2) == todo[:10] and all(
            v == "gateway error (GatewayError: No reply from IB Gateway to the historical bars request within 20s)"
            for v in ml2.values()), ml2)
        g = Flaky(10 ** 6, lost)
        (n, ok2, k2, ok3, why), ml2, out = run_warm(g, end=100.0)
        check("down past --until: stops 'gateway down' at the deadline", why == "gateway down" and n == 10
              and sum(sleeps) <= 100.0, (why, n, sleeps))
        g = Flaky(10 ** 6, lost)
        (n, ok2, k2, ok3, why), ml2, out = run_warm(g, end=45.0)    # the wait runs into the deadline (the review's case)
        check("an outage into --until: still 'gateway down', gateway errors logged, not 'time cap'",
              why == "gateway down" and n == 10 and all(v.startswith("gateway error") for v in ml2.values()), (why, n, ml2))
        (n, ok2, k2, ok3, why), ml2, out = run_warm(Flaky(0, lost), end=4.0)
        check("under 5 s left: no batch started, 'deadline'", (n, why) == (0, "deadline") and not ml2, (n, why))
        nosec = GatewayError("No security definition has been found for the request", 200)
        slow = GatewayError("No reply from IB Gateway to the historical bars request within 20s")
        g = Flaky(0, lost, per_name={"NOSEC": nosec, "SLOW": slow})
        (n, ok2, k2, ok3, why), ml2, out = run_warm(g, names=["A", "NOSEC", "SLOW"])
        check("one name's errors in a batch with rows: no retry; its gateway error, or ML2's reason",
              why == "done" and not sleeps and len(g.calls) == 6 and ml2 == {
                  "A": "ok", "NOSEC": "not covered (not in the cache and no gateway bars)",
                  "SLOW": "gateway error (GatewayError: No reply from IB Gateway to the historical bars request within 20s)"},
              ml2)
        g = Flaky(0, lost, per_name={"X1": nosec, "X2": nosec})
        (n, ok2, k2, ok3, why), ml2, out = run_warm(g, names=["X1", "X2"])
        check("a batch of names with no data is not an outage", why == "done" and not sleeps and len(g.calls) == 4
              and set(ml2.values()) == {"not covered (not in the cache and no gateway bars)"}, (why, sleeps, ml2))
        W._connect = lambda client: Flaky(10 ** 6, GatewayError("Couldn't connect to IB Gateway", 502))
        clock["t"] = 0.0
        code, out = run_main(["--max", "12"])
        last = logged()[-1]
        check("main: a gateway that stays down ends 'gateway down', exit 2", code == 2 and last["stopped"] == "gateway down"
              and last["names"] == last["names_to_warm"] == 8 and last["ml2_rows"] == 0, (code, last))
    finally:
        forecast._ondemand, forecast3._outside_rows, W.time = saved

    # 9. two consecutive closes: the names warmed but never asked stay in the broad tier
    setup(tmp / "b")
    W._connect = lambda client: GW()
    code, out = run_main(["--max", "8"])
    check("Friday's close: asked names, then the broad ones", code == 0
          and [x.get("symbol") for x in logged()[:-1]] == WANT[:8], logged())
    put(forecast.OUT / f"fc_{NEXT}.json", {"as_of": NEXT, "market": {}, "forecasts": {"AAPL": {}, "MSFT": {}}})
    look(NEXT, "sonnet-ml-0930", [("forecast", {"symbols": ["NEWX", "ARM"]})], claude=True)
    W._now = lambda: TUE
    forecast._MEMO.clear()                          # a day apart in fact: the 15-minute memo is long gone
    asked_b = ["ARM", "FUTR", "NEWX", "COIN", "CRDO", "TCAP", "ALAB", "BRK B", "APP"]
    rec_b = W.recent(NEXT)
    check("Monday's close: the asks through Monday, no warmed-only name", rec_b == asked_b[:7] + ["BRK B", "AAPL", "APP"]
          and not {"BIG", "LAST1", "WARMX"} & set(rec_b), rec_b)
    nb = W.names(NEXT, UNIVERSE)
    check("Monday's close: every asked name before the broad ones; Friday's bars still count, OLD10 now too old",
          nb == asked_b + ["BIG", "LAST1", "ETF1", "ZEROPX", "MID", "LATER", "BF B"], nb)
    W.LOG.unlink()
    W._connect = lambda client: GW(end=NEXT)
    import bars
    completed, bars.completed = bars.completed, lambda rows: rows   # it reads the real clock; on Tue 08:20 Monday's bar is done
    try:
        code, out = run_main(["--max", "12"])
    finally:
        bars.completed = completed
    lines = logged()
    check("Monday's close, --max 12: in that order, ML2 and ML3 rows", code == 0 and [x.get("symbol") for x in lines[:-1]]
          == nb[:12] and lines[-1]["stopped"] == "max" and lines[-1]["ml3_rows"] == 12 and lines[-1]["ml2_kept"] == 11, lines)
    check("after it: warmed-only names still aren't asked, and the next names are the unscored ask, then the broad tier",
          not {"BIG", "LAST1", "ETF1"} & set(W.recent(NEXT)) and W.names(NEXT, UNIVERSE) == ["APP", "ZEROPX", "MID", "LATER", "BF B"],
          W.names(NEXT, UNIVERSE))
    W._now = lambda: MON

    # 10. live.py's hooks
    msgs = []
    live.log = msgs.append
    live.LIVE = tmp / "live"
    ml = {"base-ml": {"model": "base", "ml": True}, "base": {"model": "base", "ml": False}}
    paper.sleeves = lambda: ml

    class Proc:
        def __init__(self):
            self.pid, self.code = 4242, None

        def poll(self):
            return self.code

    started = []

    def popen(args, **kw):
        started.append((args, kw))
        return Proc()

    live.WARM = None
    live.warm_outside(now=MON, popen=popen)
    args, kw = started[-1]
    check("hook: started in the background", len(started) == 1 and args == [sys.executable, str(ROOT / "warm_outside.py")]
          and kw["cwd"] == live.ROOT and kw["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
          and kw["stderr"] == subprocess.STDOUT and "warm-up started (pid 4242)" in msgs[-1], (started, msgs))
    check("hook: its output log", (live.LIVE / "warm_outside.out.log").read_text(encoding="utf-8").startswith("=== 2026-11-09 08:20"))
    live.warm_outside(now=MON, popen=popen)
    check("hook: not two at once", len(started) == 1 and "still running" in msgs[-1], msgs)
    live.WARM.code = 0
    live.warm_outside(now=MON, popen=popen)
    check("hook: again once the last one ended, which is logged first", len(started) == 2
          and msgs[-2] == "outside-name warm-up ended (exit 0): no summary line from this run in runs/live/warm_outside.jsonl",
          msgs[-3:])

    def broken(args, **kw):
        raise OSError("no such file")

    live.WARM = None
    live.warm_outside(now=MON, popen=broken)
    check("hook: a Popen error is logged, not raised", "warm-up not started: OSError" in msgs[-1], msgs)
    live.warm_outside(now=datetime(2026, 11, 7, 8, 20, tzinfo=ET), popen=popen)
    live.warm_outside(now=datetime(2026, 11, 26, 8, 20, tzinfo=ET), popen=popen)
    paper.sleeves = lambda: {"base": {"model": "base", "ml": False}, "reserve": {"model": live.HOLD_MODEL, "ml": True}}
    live.warm_outside(now=MON, popen=popen)
    check("hook: not on a weekend or holiday, nor without an ML sleeve that looks", len(started) == 2, started)

    # the loop's poll: nothing while it runs; once it ends, its exit code and this run's summary line, once
    paper.sleeves = lambda: ml
    jl = live.LIVE / "warm_outside.jsonl"
    put(jl, "\n".join(json.dumps(x) for x in [
        {"t": "2026-11-06 08:55:01", "summary": True, "stopped": "deadline", "names": 1, "ml2_kept": 1},
        {"t": "2026-11-09 08:30:00", "symbol": "ARM", "ml2": "ok"},
        {"t": "2026-11-09 08:55:02", "summary": True, "as_of": AS_OF, "stopped": "deadline", "names": 312, "ml2_rows": 300,
         "ml2_kept": 298, "ml3_rows": 305}]) + "\n")
    live.WARM = None
    live.warm_outside(now=MON, popen=popen)
    n0 = len(msgs)
    live.warm_poll()
    check("poll: nothing while it runs", len(msgs) == n0 and live.WARM is not None)
    live.WARM.code = 0
    live.warm_poll()
    live.warm_poll()
    check("poll: once it ends, its exit code and summary, once", msgs[n0:] == [
        "outside-name warm-up ended (exit 0): stopped deadline, 312 names, 298 ML2 rows kept"] and live.WARM is None, msgs[n0:])
    live.warm_outside(now=MON.replace(hour=9, minute=0), popen=popen)
    live.WARM.code = 2
    live.warm_poll()
    check("poll: an older summary is not this run's", msgs[-1] == "outside-name warm-up ended (exit 2): no summary line "
          "from this run in runs/live/warm_outside.jsonl", msgs[-1])
    with jl.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"t": "2026-11-09 09:00:04", "summary": True, "stopped": "connect failed: GatewayError: 326: "
                            "client id is already in use", "names": 0, "ml2_kept": 0}) + "\n")
    live.warm_outside(now=MON.replace(hour=9, minute=0), popen=popen)
    live.WARM.code = 2
    live.warm_poll()
    check("poll: a connect failure", msgs[-1] == "outside-name warm-up ended (exit 2): stopped connect failed: GatewayError: "
          "326: client id is already in use, 0 names, 0 ML2 rows kept", msgs[-1])

    # the look record names the Kelly paragraph and the single-shock limit, next to the tool's version
    st = {"events": [], "value": 100000.0}
    live.record_look(MON, "base-100k-ml", {"model": "base", "ml": True, "ml_tool": "ML3", "kelly_prompt": True,
                                           "shock_cap": 0.15}, "text", st, None, [], [], [], True, secs=1.0)
    live.record_look(MON, "base", {"model": "base", "ml": False, "shock_cap": "15%"}, "text", st, None, [], [], [], True)
    r1, r0 = (json.loads((live.LIVE / "2026-11-09" / f"{k}-0820.json").read_text(encoding="utf-8")) for k in ("base-100k-ml", "base"))
    check("look record: kelly_prompt and shock_cap", (r1["ml_tool"], r1["kelly_prompt"], r1["shock_cap"]) == ("ML3.0", True, 0.15)
          and (r0["ml_tool"], r0["kelly_prompt"], r0["shock_cap"]) == (None, False, None), (r1, r0))

    # refresh() starts it after compute_versions succeeds, and not when it fails
    order = []
    import trader
    trader.universe = lambda: []
    paper.sleeves, paper.book = (lambda: ml), (lambda name, marks=None: {"positions": {}})
    live.gateway = lambda: contextlib.nullcontext(DeadGW())
    live.book_dividends = lambda gw=None: None
    live.rates.fetch = lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    sys.modules["fetch_extra"] = types.SimpleNamespace(update=lambda gw, syms, kind, **k: [])
    sys.modules["fetch_av"] = types.SimpleNamespace(daily=lambda n: (_ for _ in ()).throw(RuntimeError("offline")))
    live.compute_versions = lambda: order.append("compute")
    live.warm_outside = lambda: order.append("warm")
    live.ml2_guard = lambda: order.append("guard")
    live.refresh()
    check("refresh: the warm-up right after compute_versions", order == ["compute", "warm", "guard"], order)

    def fails():
        order.append("compute failed")
        raise RuntimeError("ML2 compute failed")

    order[:] = []
    live.compute_versions = fails
    try:
        live.refresh()
        raised = False
    except RuntimeError:
        raised = True
    check("refresh: no warm-up when compute_versions fails", raised and order == ["compute failed"], order)

    # the loop: its start line names the code it runs, and every pass polls the warm-up
    head = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    vs = ", ".join(f"{k} {ml_tools.module(k).VERSION}" for k in sorted(ml_tools.VERSIONS))
    check("start line: git describe (dirty shown) and every ML tool's VERSION",
          live.code_line() == f"code {head}; ML tools: {vs}" and head and "ML3 ML3.0" in vs, live.code_line())
    dirty = types.SimpleNamespace(returncode=0, stdout="2baf629-dirty\n")
    check("start line: a dirty tree is named", live.code_line(run=lambda *a, **k: dirty).startswith("code 2baf629-dirty; "))
    check("start line: git unavailable", live.code_line(run=lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("git")))
          .startswith("code unknown; "))
    polls = []
    live.JOBS, live.STOP, live.load_done = [], tmp / "STOP", lambda: set()
    live.macro_intraday = lambda st: None
    live.warm_poll = lambda: polls.append(1)
    live.time = types.SimpleNamespace(sleep=lambda s: live.STOP.touch(), time=time.time)
    live.loop()
    check("loop: the start line, and a poll each pass", msgs[-2] == f"live loop started: {live.code_line()}"
          and polls == [1] and "STOP file found" in msgs[-1], msgs[-2:])

    # nothing outside the temporary folder moved, and ML2 is as its manifest says
    now = {p: sorted(str(x) for x in p.rglob("*")) if p.exists() else None for p in real}
    check("ML2's and ML3's real caches untouched", now == real)
    check("ml_tools.changed_files() is empty", ml_tools.changed_files() == [], ml_tools.changed_files())


if __name__ == "__main__":
    try:
        main()
    except Exception:                                   # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
    print("ALL OK")
    sys.exit(0)
