"""live.look() with the gpt-oss sleeves side by side (GPTOSS_LANES, 11 Nov 2026), on fakes: no gateway, no Tinker, no
ledger or log writes (the gpt-oss tokenizer from the local cache).

Checks: every sleeve is recorded with its decision; the sampling overlaps; no two gateway calls, research calls,
placements or records run at once (they share client 10 and read-and-rewrite the ledgers); a read-modify-write ledger
loses no sleeve's update; each sleeve after the first re-syncs fills before its state, as before; a sleeve whose
sampling fails costs only its own look.

    python parallel_look_test.py
"""
import contextlib
import json
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


def fake_sync(gw):
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


def fake_tool(gw, name, args, ml_tool=None):
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
    (live, "finish", fake_finish), (live, "log", lambda m: lines.append(m)), (live, "load_memo", lambda n: {}),
    (live, "sessions_left", lambda t: 5), (live.ml_tools, "unknown", lambda s: []), (live.rates, "line", lambda now: ""),
    (live.tinker, "ServiceClient", FakeService), (tools, "available", lambda gw: SPEC), (tools, "run", fake_tool),
    (agent, "sample_once", fake_sample_once),
    (trader, "Data", FakeData), (trader, "PaperWorld", FakeWorld),
    (trader, "state_text", lambda world, data, d, header=None, live=None, book=False: ("state", {"value": 30000, "events": []}, None)),
    (trader, "goal_line", lambda *a, **k: "goal"), (trader, "live_user", lambda *a, **k: "user"),
    (trader, "live_steps", lambda ml: ""), (trader, "live_system", lambda *a, **k: "system"),
]
saved = [(m, a, getattr(m, a)) for m, a, _ in patches]
for m, a, v in patches:
    setattr(m, a, v)
try:
    t0 = time.time()
    live.look(dry_run=False)
    wall = time.time() - t0
finally:
    for m, a, v in saved:
        setattr(m, a, v)

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
