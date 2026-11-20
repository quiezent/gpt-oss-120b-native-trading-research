"""live.look() with the gpt-oss sleeves side by side (GPTOSS_LANES, 11 Nov 2026), on fakes: no gateway, no Tinker, no
ledger or log writes (the gpt-oss tokenizer from the local cache).

Checks: every sleeve is recorded with its decision; the sampling overlaps; no two gateway calls, research calls,
placements or records run at once (they share client 10 and read-and-rewrite the ledgers); a read-modify-write ledger
loses no sleeve's update; each sleeve after the first re-syncs fills before its state, as before; a sleeve whose
sampling fails costs only its own look.

    python parallel_look_test.py
"""
import atexit
import contextlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

sys.dont_write_bytecode = True
import agent  # noqa: E402
import live  # noqa: E402
import paper  # noqa: E402
import tools  # noqa: E402
import trader  # noqa: E402

N, SAMPLE_S, TOOL_S = 8, 0.4, 0.03
SPEC = [{"name": "quote", "description": "q",
         "parameters": {"type": "object", "properties": {"symbols": {"type": "array", "items": {"type": "string"}}}}}]
THINK = "<|channel|>analysis<|message|>Reasoning.<|end|><|start|>assistant"
CALL = THINK + '<|channel|>commentary to=functions.quote <|constrain|>json<|message|>{"symbols": ["A"]}<|call|>'
SUBMIT = THINK + ('<|channel|>commentary to=functions.submit <|constrain|>json<|message|>'
                  '{"place": [{"symbol": "A", "action": "BUY", "quantity": 1}], "cancel": [], "notes": "n"}<|call|>')
TURNS = [CALL, CALL, SUBMIT]                       # 3 samples and 2 research calls a sleeve
ok, lines = [], []


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail else ""))


class Busy:
    """Counts how many threads are inside at once."""
    def __init__(self):
        self.now, self.max, self.n, self.lock = 0, 0, 0, threading.Lock()

    @contextlib.contextmanager
    def inside(self, hold=0.0):
        with self.lock:
            self.now += 1
            self.n += 1
            self.max = max(self.max, self.now)
        try:
            time.sleep(hold)
            yield
        finally:
            with self.lock:
                self.now -= 1


gw_busy, tool_busy, act_busy, rec_busy, sample_busy, sync_busy = Busy(), Busy(), Busy(), Busy(), Busy(), Busy()


GW_LOCK = threading.RLock()                        # tws.Gateway holds an RLock around each call


class FakeGW:
    def __getattr__(self, name):
        def call(*a, **k):
            with GW_LOCK, gw_busy.inside(0.005):
                return []
        return call


@contextlib.contextmanager
def fake_gateway():
    yield FakeGW()


class FakeData:
    dates = ["20261110"]

    def __init__(self, **k):
        pass

    def table_rows(self, d, extra=None):
        return ["A"], None


ledger_file = Path(tempfile.mkdtemp()) / "memo.json"
ledger_file.write_text("{}", encoding="utf-8")
recorded, act_calls, act_times, slow = {}, [], [], []
who = threading.local()          # the sleeve this thread samples for
FAIL = "s3"                                        # this sleeve's sampling raises


class FakeWorld:
    def __init__(self, gw, name, data):
        self.gw, self.name = gw, name

    def symbols_in_play(self):
        return []

    def act(self, call):
        with act_busy.inside(0.01):
            self.gw.place()                        # a gateway call inside the placement, as paper.place makes
            act_calls.append(self.name)
            act_times.append(time.time())
        return []


def fake_finish(now, name, meta, text, st, call, trace, notes, rows, dry_run, cost=None, secs=None):
    with rec_busy.inside():
        d = json.loads(ledger_file.read_text(encoding="utf-8"))   # read, wait, rewrite: loses updates if two overlap
        time.sleep(0.01)
        d[name] = {"call": call, "error": any("error" in t for t in trace)}
        ledger_file.write_text(json.dumps(d), encoding="utf-8")
        recorded[name] = (call, trace)


MARKER = Path(tempfile.mkdtemp()) / "LOOKING"     # live.LOOKING, patched (16 Nov 2026): the real one is never touched
atexit.register(shutil.rmtree, MARKER.parent, True)
OWN = MARKER.with_name(f"LOOKING.{os.getpid()}")   # this process's own marker (each process its own file)
OTHER_FILE = MARKER.with_name("LOOKING.99999")     # another process's
ALIVE = {99999}                                    # a dry-run look in another process; pid 1234 is dead


class FakeDT(live.datetime):                       # loop()'s clock: Mon 16 Nov 2026, 10:05 ET, inside the 10:00 look's window
    @classmethod
    def now(cls, tz=None):
        return live.datetime(2026, 11, 16, 10, 5, tzinfo=live.ET)
marker_seen = []


def fake_sync(gw):
    marker_seen.append(OWN.exists())               # the look's first sync, each sleeve's re-sync and the post-look sync
    with sync_busy.inside(0.005):
        gw.executions()


class FakeSC:
    def __init__(self, name):
        self.name, self.turns = name, iter(TURNS)


class FakeService:
    current = threading.local()

    made = 0

    def __init__(self, **k):
        FakeService.made += 1

    def create_sampling_client(self, base_model=None, model_path=None):
        return FakeSC(model_path)


span = []                                          # the first sample's start and the last one's end


def fake_sample_once(sc, renew, model_input, params, trace, step):
    who.sleeve = sc.name
    span.append(time.time())
    with sample_busy.inside(SAMPLE_S):
        if sc.name == f"m-{FAIL}":
            raise RuntimeError("sampling failed (test)")
        toks = list(trader.tokenizer().encode(next(sc.turns), add_special_tokens=False))
        span.append(time.time())
        return types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=toks, stop_reason="stop")]), sc


SLOW_S = 2.0


def fake_tool(gw, name, args, ml_tool=None, sleeve_value=None, look=None):    # 16 Nov 2026: the forecast keywords
    hold = TOOL_S
    if getattr(who, "sleeve", "") == "m-s0" and not getattr(who, "first_done", False):
        who.first_done = True
    elif getattr(who, "sleeve", "") == "m-s0":
        hold = SLOW_S                              # one slow research call, as an SEC retry or a stalled bars request
    t0 = time.time()
    with tool_busy.inside(hold):
        gw.quotes(["A"])
    if hold == SLOW_S:
        slow.append((t0, time.time()))
    return f"{name} result"


# The fakes. Nothing here writes a real ledger, the loop's log or a look record.
sleeves = {f"s{i}": {"model": f"m-s{i}", "start_cash": 30000} for i in range(N)}
patches = [
    (paper, "sleeves", lambda: sleeves), (paper, "sync_fills", fake_sync), (paper, "book", lambda *a, **k: {"positions": {}}),
    (live, "gateway", fake_gateway), (live, "reconcile", lambda gw: {}), (live, "ml2_guard", lambda: None),
    (live, "open_whatif", lambda: (None, False)),     # 16 Nov 2026: no what-if connection (check_orders_test covers it)
    (live, "LOOKING", MARKER), (live, "LOOP", MARKER.parent / "LOOP.json"),     # loop() below writes it: never the real one
    (live, "pid_alive", lambda pid: pid in ALIVE),    # 16 Nov 2026: which pids count as alive, for the marker's owner
    (live, "finish", fake_finish), (live, "log", lambda m: lines.append(m)), (live, "load_memo", lambda n: {}),
    (live, "sessions_left", lambda t: 5), (live.ml_tools, "unknown", lambda s: []), (live.rates, "line", lambda now: ""),
    (live.tinker, "ServiceClient", FakeService), (tools, "available", lambda gw: SPEC), (tools, "run", fake_tool),
    (agent, "sample_once", fake_sample_once),
    (trader, "Data", FakeData), (trader, "PaperWorld", FakeWorld),
    (trader, "state_text", lambda world, data, d, header=None, live=None, book=False, cash_clause=False:
        ("state", {"value": 30000, "events": []}, None)),
    (trader, "goal_line", lambda *a, **k: "goal"), (trader, "live_user", lambda *a, **k: "user"),
    (trader, "live_steps", lambda ml, **k: ""), (trader, "live_system", lambda *a, **k: "system"),
]
saved = [(m, a, getattr(m, a)) for m, a, _ in patches]
for m, a, v in patches:
    setattr(m, a, v)
try:
    t0 = time.time()
    live.look(dry_run=False)
    wall = time.time() - t0
    marker_after = OWN.exists() or MARKER.exists()
    # the marker after a look that raises (its first fill sync fails), and the loop's stale-marker removal at start
    real_sync, paper.sync_fills = paper.sync_fills, lambda gw: (_ for _ in ()).throw(RuntimeError("sync failed (test)"))
    try:
        live.look(dry_run=False)
        raised = False
    except RuntimeError:
        raised = True
    finally:
        paper.sync_fills = real_sync
    marker_after_raise = OWN.exists()
    MARKER.write_text("2026-11-16T09:00:00-05:00 1234\n", encoding="utf-8")   # the plain file a 23a7075 loop left, pid dead
    MARKER.with_name("LOOKING.1234").write_text("2026-11-16T09:00:00-05:00 1234\n", encoding="utf-8")   # dead
    OWN.write_text(f"2026-11-16T08:00:00-05:00 {os.getpid()}\n", encoding="utf-8")   # this pid's, from before a reboot
    MARKER.with_name("LOOKING.777").write_text("garbled\n", encoding="utf-8")            # names no pid, an hour old
    os.utime(MARKER.with_name("LOOKING.777"), (time.time() - 3600, time.time() - 3600))
    stop = MARKER.parent / "STOP"
    stop.write_text("", encoding="utf-8")
    real_stop, real_code = live.STOP, live.code_line
    live.STOP, live.code_line = stop, lambda: "code test"
    try:
        live.loop()
        removed = [m for m in lines if m.startswith("stale LOOKING marker removed (")]
        stale_gone = (not live.marker_files() and len(removed) == 4
                      and any(m.startswith("stale LOOKING marker removed (2026-11-16T09:00:00-05:00 1234; LOOKING)")
                              for m in removed))
        loop_json = json.loads((MARKER.parent / "LOOP.json").read_text(encoding="utf-8"))
        # a marker that names no pid but is under a minute old may be a begin_marker's write in progress: left; one that
        # can't be read is removed like one that names no pid, and logged
        FRESH, UNREAD = MARKER.with_name("LOOKING.778"), MARKER.with_name("LOOKING.888")
        FRESH.write_text("", encoding="utf-8")
        UNREAD.write_text("2026-11-16T09:00:00-05:00 888\n", encoding="utf-8")
        os.utime(UNREAD, (time.time() - 3600, time.time() - 3600))
        real_mpid = live.marker_pid
        live.marker_pid = lambda path=None: (None, None) if path == UNREAD else real_mpid(path)   # as on an OSError
        lines.clear()
        try:
            live.loop()
        finally:
            live.marker_pid = real_mpid
        fresh_kept = (FRESH.exists() and not UNREAD.exists()
                      and any(m.startswith("LOOKING left in place: it names no pid but was written under a minute ago "
                                           "(LOOKING.778") for m in lines)
                      and "stale LOOKING marker removed (not readable; LOOKING.888)" in lines)
        fresh_lines = list(lines)
        FRESH.unlink(missing_ok=True)
        UNREAD.unlink(missing_ok=True)
        # each process its own marker (the TWS Gateway Owner's review, 16 Nov): another live process's markers
        OTHER = "2026-11-16T10:00:00-05:00 99999"
        OTHER_FILE.write_text(OTHER + "\n", encoding="utf-8")
        MARKER.write_text(OTHER + "\n", encoding="utf-8")               # the plain file, naming the same live process
        lines.clear()
        live.loop()
        alive_kept = (OTHER_FILE.read_text(encoding="utf-8").strip() == OTHER
                      and MARKER.read_text(encoding="utf-8").strip() == OTHER
                      and len([m for m in lines if m.startswith("LOOKING left in place: process 99999 is alive and looking")]) == 2)
        OTHER_FILE.unlink()
        MARKER.unlink()
        real_look = live._look
        real_getpid = os.getpid

        def other_process(step):                    # the other process's own begin_marker() or end_marker(), as pid 99999
            os.getpid = lambda: 99999
            try:
                step()
            finally:
                os.getpid = real_getpid
        # the reviewer's example: a hand-started dry-run look (another process) starts and ends during this one's look
        during = []

        def overlapped(d, o):
            during.append(OWN.exists())
            other_process(live.begin_marker)
            during.append(OWN.exists() and OTHER_FILE.exists())
            other_process(live.end_marker)
            during.append(OWN.exists() and not OTHER_FILE.exists())
        lines.clear()
        live._look = overlapped
        live.look(dry_run=True)
        overlap_kept = during == [True, True, True] and not OWN.exists() and not live.marker_files()
        # another process's marker written mid-look is left at this look's end; this look removes only its own
        lines.clear()
        live._look = lambda d, o: other_process(live.begin_marker)
        live.look(dry_run=True)
        others_left = OTHER_FILE.exists() and not OWN.exists() and OTHER_FILE.read_text(encoding="utf-8").split()[-1] == "99999"
        inside = []
        live._look = lambda d, o: inside.append((OWN.read_text(encoding="utf-8").split()[-1],
                                                 OTHER_FILE.read_text(encoding="utf-8").split()[-1]))
        live.look(dry_run=True)                                       # starts with the other's marker there: writes its own
        own_beside = inside == [(str(os.getpid()), "99999")] and not OWN.exists() and OTHER_FILE.exists()
        OTHER_FILE.unlink()
        # loop(): the marker is written before the look is marked done; a look that never runs leaves none
        at_mark, ran = [], []
        live._look = lambda d, o: ran.append(OWN.read_text(encoding="utf-8").split()[-1])
        stop.unlink()
        sched = [(live, "JOBS", [("look-1000", (10, 0))]), (live, "datetime", FakeDT), (live, "load_done", lambda: set()),
                 (live, "HOLD", MARKER.parent / "HOLD"), (live, "macro_intraday", lambda st: None), (live, "warm_poll", lambda: None),
                 (live, "mark_done", lambda done, key: at_mark.append(OWN.exists() and OWN.read_text(
                     encoding="utf-8").split()[-1] == str(os.getpid()))),
                 (live.time, "sleep", lambda s: stop.write_text("", encoding="utf-8"))]
        kept = [(m, a, getattr(m, a)) for m, a, _ in sched]
        for m, a, v in sched:
            setattr(m, a, v)
        try:
            live.loop()
            order_ok = at_mark == [True] and ran == [str(os.getpid())] and not OWN.exists()
            stop.unlink()
            live.mark_done = lambda done, key: (_ for _ in ()).throw(OSError("done.json not written (test)"))
            ran.clear()
            try:
                live.loop()
            except OSError:
                pass
            never_ran = not ran and not OWN.exists()
            # the look's own end_marker can't remove the file (on Windows a file another handle holds open, such as
            # reopen_week6.py reading it, can't be deleted): the loop removes it once the handle is gone, before it sleeps
            held_open = []
            live.mark_done = sched[6][2]
            live._look = lambda d, o: held_open.append(open(OWN, encoding="utf-8"))
            live.warm_poll = lambda: [f.close() for f in held_open]
            stop.unlink(missing_ok=True)
            lines.clear()
            try:
                live.loop()
            finally:
                for f in held_open:
                    f.close()
            unlink_failed = any(m.startswith("LOOKING marker not removed") for m in lines)
            swept = bool(held_open) and not OWN.exists()
        finally:
            for m, a, v in kept:
                setattr(m, a, v)
            live._look = real_look
            os.getpid = real_getpid
    finally:
        live.STOP, live.code_line = real_stop, real_code
finally:
    for m, a, v in saved:
        setattr(m, a, v)
check("LOOKING (b): another live process's markers (its LOOKING.<pid> and the plain file) are left at loop start, each logged",
      alive_kept)
check("LOOKING (N1): another process's look starting and ending during this look leaves this look's marker the whole time",
      overlap_kept, during)
check("LOOKING: another process's marker written mid-look is left at this look's end; this look removes only its own",
      others_left)
check("LOOKING: a look that starts beside another live process's marker writes its own file and leaves the other's",
      own_beside, inside)
check("LOOKING (d): in loop(), the marker exists, naming this process, when the look is marked done; gone after the look", order_ok,
      (at_mark, ran))
check("LOOKING: a look that never runs (marking it done fails) leaves no marker", never_ran)
check("LOOKING: present through the look, from its first fill sync to its post-look sync, and gone after it",
      marker_seen and all(marker_seen) and len(marker_seen) == N + 1 and not marker_after, (marker_seen, marker_after))
check("LOOKING: gone after a look that raises (the exception still reaches the caller)", raised and not marker_after_raise)
check("LOOKING (c): at loop start a dead pid's marker (plain and LOOKING.<pid>), this pid's own leftover and one naming no "
      "pid are removed, each logged", stale_gone, removed)
check("LOOP.json says \"looking\": 2 (per-process markers)", loop_json.get("looking") == 2, loop_json)
check("LOOKING: at loop start a marker naming no pid but under a minute old (being written) is left, and one that can't be "
      "read is removed, each logged", fresh_kept, fresh_lines)
check("LOOKING: this process's marker that its look's end could not remove (held open) is removed by the loop before it "
      "sleeps", swept and (unlink_failed or os.name != "nt"), (held_open, OWN.exists(), unlink_failed))

good = [n for n in sleeves if n != FAIL]
seq = N * len(TURNS) * SAMPLE_S
check(f"all {N} sleeves recorded", sorted(recorded) == sorted(sleeves), sorted(recorded))
check("each healthy sleeve decided and placed", all(recorded[n][0] and recorded[n][0]["place"] for n in good)
      and sorted(act_calls) == sorted(good), sorted(act_calls))
check("the failing sleeve's look ended with its error only", recorded.get(FAIL, (1,))[0] is None
      and any("sampling failed" in str(t.get("error", "")) for t in recorded[FAIL][1]))
check("the sampling overlapped", sample_busy.max >= min(N, live.GPTOSS_LANES) - 1, f"max {sample_busy.max} at once")
phase = max(span) - min(span)
check(f"the sampling phase took far less than one after another ({phase:.1f}s against {seq:.1f}s)", phase < 0.5 * seq,
      f"whole look {wall:.1f}s, with the post-look 5 s wait")
check("no two gateway calls at once (the Gateway's own lock)", gw_busy.max == 1, f"max {gw_busy.max} of {gw_busy.n}")
check("one Tinker client for the look, shared by the sleeves", FakeService.made == 1, f"{FakeService.made} made")
during = [t for t in act_times if slow and slow[0][0] < t < slow[0][1]]
check("a slow research call held up no other sleeve's placement", slow and len(during) >= 1,
      f"{len(during)} of {len(act_times)} placements during s0's {SLOW_S:.0f} s call")
check("no two research calls at once", tool_busy.max == 1 and tool_busy.n == 2 * len(good), f"max {tool_busy.max} of {tool_busy.n}")
check("no two placements at once", act_busy.max == 1, f"max {act_busy.max}")
check("no two records at once", rec_busy.max == 1, f"max {rec_busy.max}")
d = json.loads(ledger_file.read_text(encoding="utf-8"))
check("a read-and-rewrite ledger kept every sleeve's update", sorted(d) == sorted(sleeves), sorted(d))
check(f"each sleeve after the first re-synced fills ({N - 1}), plus the look's first and last", sync_busy.n == N + 1,
      f"{sync_busy.n} syncs")
check("nothing logged as a thread failure", not any("thread failed" in m for m in lines), [m for m in lines if "failed" in m][:2])
print("ALL OK" if all(ok) else f"FAILED ({ok.count(False)})")
sys.exit(0 if all(ok) else 1)
