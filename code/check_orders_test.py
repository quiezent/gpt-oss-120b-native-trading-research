"""check_orders and the whole-account line, text B from Mon 16 Nov 2026 (calculator_tools_spec_v2.md, frozen 15 Nov; Clayton,
14 Nov: 3 checks a look outside the 12 research calls). Offline: fake gateways built from the Gateway Owner's fixture of
15 Nov (C:/Claude/TWS-Gateway-handoff/whatif_fixture_20261115.json, one read at 01:04Z), no IB connection, no Tinker, no
Claude CLI; nothing under runs/ is written (the look records go to a temp folder).

- the three worked results, rebuilt from the fixture as the checker's corrections ask, byte for byte, through gptoss_look's
  counter; every failure text; position phrases; the all-orders line; trails;
- the mapping of each order type to whatif arguments (the same as PaperWorld.act's for paper.place), the model's own keys
  left out (checked against the framework's own Order-field test when tws imports offline);
- the cap of 3, outside the 12 research calls; a failed connection does not count; a what-if that raises costs that check;
  nothing is placed;
- the account line, byte for byte, and its missing pieces; text A gets neither tool nor line;
- the MCP path (its own client, its own cap), claude_look's env and sentence, and live.look's wiring on fakes;
- live.open_whatif without whatif_many: off, one log line;
- the reviews of 15 Nov: a look always reaches a submit (only checks, research and checks alternating, random mixes; the link
  working, failing, absent); an order with no type is not checked; the all-orders line when nothing was sent, a result has no
  figures, or a non-201 code has figures; a slow link waits 5 s once a look and its checks still run.

    python check_orders_test.py
"""
import json
import os
import sys
import tempfile
import threading
import types
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import agent  # noqa: E402
import checks  # noqa: E402
import live  # noqa: E402
import paper  # noqa: E402
import tools  # noqa: E402
import trader  # noqa: E402

ok = []
TMP = Path(tempfile.mkdtemp())


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail and not good else ""))


FIX = json.loads(Path("C:/Claude/TWS-Gateway-handoff/whatif_fixture_20261115.json").read_text(encoding="utf-8"))
R = {k: v["result"] for k, v in FIX["whatifs"].items()}
ACCT = FIX["account"]
NOW = datetime(2026, 11, 16, 10, 31, 7, tzinfo=checks.ET)
CTX = {"value": 90000.0, "positions": {}, "prices": {"SPY": [749.6, "delayed last"], "SOXX": [148.09, "delayed last"]},
       "sleeves": 9, "looking": 9}


class GatewayError(Exception):                              # as tws.GatewayError: "<code>: <message>"
    def __init__(self, message, code=None):
        super().__init__(f"{code}: {message}" if code is not None else message)
        self.code = code


class FakeGW:
    """A what-if gateway: account(), positions(), all_open_orders() and whatif_many() from scripted replies. place() raises."""

    def __init__(self, replies=(), held=None, resting=(), acct=None, fail=None):
        self.replies, self.held, self.resting = list(replies), {"SPY": 82} if held is None else held, list(resting)
        self.acct = dict(ACCT) if acct is None else acct
        self.fail, self.sent, self.connects, self.placed = dict(fail or {}), [], 0, []
        self.reads, self.closed = 0, False

    def close(self):
        self.closed = True

    def _maybe(self, what):
        e = self.fail.get(what)
        if e is not None:
            if isinstance(e, list):
                e = e.pop(0) if e else None
            if e is not None:
                raise e

    def account(self):
        self.reads += 1
        self._maybe("account")
        return dict(self.acct)

    def positions(self):
        self._maybe("positions")
        return [{"symbol": s, "quantity": q} for s, q in self.held.items()]

    def all_open_orders(self):
        return list(self.resting)

    def whatif_many(self, orders, timeout=10.0):
        self._maybe("whatif")
        for o in orders:
            extra = set(o) - {"symbol", "action", "quantity", "order_type", "limit", "aux", "tif", "primary", "ref"}
            if extra - {"trailingPercent", "outsideRth"}:
                raise ValueError(f"Order has no field {sorted(extra)[0]!r}")
            if not o.get("symbol"):
                raise ValueError(f"What-if order {o!r} has no symbol")
        self.sent.append([dict(o) for o in orders])
        out = []
        for o in orders:
            r = self.replies.pop(0) if self.replies else dict(R["R3"])
            out.append(dict(r))
        return out

    def connect(self):
        self.connects += 1
        self.fail.pop("account", None)
        self.fail.pop("whatif", None)

    def place(self, *a, **k):
        self.placed.append(a)
        raise AssertionError("place() called during a check")


def run(place, replies=(), ctx=CTX, **k):
    g = FakeGW(replies, **k)
    text, counted, raw = checks.check(checks.Link(g), {"place": place}, ctx, now=NOW)
    return text, counted, raw, g


HEAD = ("IBKR what-if at 10:31:07 ET; nothing was placed. Figures are for the whole paper account, which holds all 9 sleeves; 9 of "
        "them look in this round, and orders the others send after 10:31:07 are not in these figures.")
BEFORE = ("Account before (IBKR): net liquidation $1,371,230; equity with loan $1,370,011; initial margin $203,808; excess liquidity "
          "$1,167,302; SMA $1,145,555; gross $775,595.")
BAND = ("IBKR's price-band notice, sent with every limit-order what-if (not about this order's size): If your order is not "
        "immediately executable, our systems may, depending on market conditions, reject your order if its limit price is more "
        "than the allowed amount away from the reference price at that time.")
GOLD = {
    "R1": "\n".join([HEAD, BAND, BEFORE,
                     "1. BUY 1826 SPY LMT 750 DAY: value $1,369,500 at its limit = 15.22x your sleeve's value ($90,000). Your sleeve "
                     "holds no SPY; if it fills, it opens a 1826-share long. Account: net 82 SPY long, no resting SPY buys; to IBKR it "
                     "adds 1826 to that long. IBKR: initial margin after $546,014; equity with loan after $1,369,326; commission "
                     "$7.86-9.14.",
                     "All orders, computed here (not by IBKR): initial margin $546,014; equity with loan $1,369,326.",
                     "(order check 1 of 3)"]),
    "R2": "\n".join([HEAD, BEFORE,
                     "1. BUY 6348 SPY MKT DAY: value $4,758,461 at the delayed last 749.60 = 52.87x your sleeve's value ($90,000). Your "
                     "sleeve holds no SPY; if it fills, it opens a 6348-share long. Account: net 82 SPY long, no resting SPY buys; to "
                     "IBKR it adds 6348 to that long. IBKR: initial margin after $1,393,470; equity with loan after $1,355,731; no "
                     "commission given. IBKR error 201 (this order alone against the whole account's margin; what-if, nothing was "
                     "sent): Order rejected - reason:YOUR ORDER IS NOT ACCEPTED. IN ORDER TO OBTAIN THE DESIRED POSITION YOUR EQUITY "
                     "WITH LOAN VALUE [1355731.11 USD] MUST EXCEED THE INITIAL MARGIN [1393470.42 USD]",
                     "All orders, computed here (not by IBKR): initial margin $1,393,470; equity with loan $1,355,731.",
                     "(order check 2 of 3)"]),
    "R3": "\n".join([HEAD, BEFORE,
                     "1. SELL 100 SPY TRAIL 0.02% GTC: value $74,960 at the delayed last 749.60 = 0.83x your sleeve's value ($90,000); "
                     "the trail is $0.15 (749.60 x 0.02 / 100). Your sleeve holds no SPY; if it fills, it opens a 100-share short. "
                     "Account: net 82 SPY long, no resting SPY sells; to IBKR it closes that long and opens an 18-share short. IBKR: "
                     "initial margin after $192,488 (change -$11,319.42); equity with loan after $1,370,008; commission $2.54.",
                     "All orders, computed here (not by IBKR): no order raises initial margin, so the account's figures above stand.",
                     "(order check 3 of 3)"]),
}
O1 = {"symbol": "SPY", "action": "BUY", "quantity": 1826, "type": "LMT", "limit": 750, "tif": "DAY"}
O2 = {"symbol": "SPY", "action": "BUY", "quantity": 6348, "type": "MKT", "tif": "DAY"}
O3 = {"symbol": "SPY", "action": "SELL", "quantity": 100, "type": "TRAIL", "trail_percent": 0.02, "tif": "GTC"}


# ---------------------------------------------------------------- gptoss_look with fakes (no tokenizer, no sampler)
class FakeRenderer:
    seen = []

    def __init__(self, tok, **k):
        pass

    def create_conversation_prefix_with_tools(self, specs, system):
        FakeRenderer.seen.append((specs, system))
        return [{"role": "system", "content": system}]

    def build_generation_prompt(self, msgs):
        return types.SimpleNamespace(to_ints=lambda: [1, 2, 3])


fake_tok = types.SimpleNamespace(decode=lambda t: "", encode=lambda s, add_special_tokens=False: [7])


def run_look(script, *a, **k):
    it = iter(script)
    rmod = types.ModuleType("tinker_cookbook.renderers.gpt_oss")
    rmod.GptOssRenderer = FakeRenderer
    patches = [(agent, "parse_any", lambda toks, stop, names: next(it)), (trader, "tokenizer", lambda: fake_tok),
               (agent, "sample_once", lambda sc, renew, mi, params, trace, step:
                (types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=[5], stop_reason="stop")]), sc))]
    saved = [(m, n, getattr(m, n)) for m, n, _ in patches]
    saved_r = sys.modules.get("tinker_cookbook.renderers.gpt_oss")
    for m, n, v in patches:
        setattr(m, n, v)
    sys.modules["tinker_cookbook.renderers.gpt_oss"] = rmod
    try:
        return agent.gptoss_look(None, *a, **k)
    finally:
        for m, n, v in saved:
            setattr(m, n, v)
        if saved_r is None:
            sys.modules.pop("tinker_cookbook.renderers.gpt_oss", None)
        else:
            sys.modules["tinker_cookbook.renderers.gpt_oss"] = saved_r


SUB = ("TOOL", "submit", {"place": [], "cancel": []})
SPEC = [{"name": "quote", "description": "q", "parameters": {"type": "object", "properties": {}}}]
researched = []
real_run = tools.run
tools.run = lambda gw, name, args, **k: researched.append(name) or "quote result"
try:
    g = FakeGW([R["R1"], R["R2"], R["R3"]])
    link = checks.Link(g)
    checker = lambda args: checks.check(link, args, CTX, now=NOW)
    script = [("TOOL", "check_orders", {"place": [O1]}), ("TOOL", "check_orders", {"place": [O2]}),
              ("TOOL", "check_orders", {"place": [O3]})]
    FakeRenderer.seen = []
    call, trace, _, _ = run_look(script + [SUB], None, "sys", "state", "2026-11-16", SPEC, "high", checker=checker)
    res = [t["result"] for t in trace if t.get("tool") == "check_orders"]
    for k, r in zip(("R1", "R2", "R3"), res):
        check(f"golden {k} (fixture of 15 Nov), byte for byte, with the counter", r == GOLD[k], "\n" + r)
    check("the trace records each check like a research call, with IBKR's raw replies",
          all(t["args"] and t["ibkr"]["whatif"] and t["ibkr"]["account"]["NetLiquidation"] == ACCT["NetLiquidation"]
              for t in trace if t.get("tool") == "check_orders"))
    check("the tool list: research, check_orders, then submit", [s["name"] for s in FakeRenderer.seen[0][0]] == ["quote", "check_orders", "submit"])
    check("nothing placed", not g.placed)

    # 12 research calls and 3 checks are all served; the 4th check gets the limit text; checks do not move the research count
    g = FakeGW([R["R3"]] * 3)
    link = checks.Link(g)
    script = ([("TOOL", "quote", {})] * 6 + [("TOOL", "check_orders", {"place": [O3]})] + [("TOOL", "quote", {})] * 6
              + [("TOOL", "check_orders", {"place": [O3]})] * 3 + [SUB])
    researched.clear()
    call, trace, _, _ = run_look(script, None, "sys", "state", "2026-11-16", SPEC, "high",
                                 checker=lambda args: checks.check(link, args, CTX, now=NOW))
    rc = [t["result"] for t in trace if t.get("tool") == "quote"]
    cc = [t["result"] for t in trace if t.get("tool") == "check_orders"]
    check("12 research calls and 3 checks are all served, then the look submits", call is not None and len(researched) == 12
          and rc[-1].endswith("(research call 12 of 12)") and rc[6].endswith("(research call 7 of 12)"), rc[-1:])
    check("the counters: (order check n of 3), separate from the research count",
          [c.rsplit("\n", 1)[-1] for c in cc[:3]] == ["(order check 1 of 3)", "(order check 2 of 3)", "(order check 3 of 3)"])
    check("a 4th check gets the limit text and the look goes on to its submit", cc[3] == checks.CAP_TEXT == (
        "check limit reached (3 of 3); no further check is run this look.") and trace[-1]["tool"] == "submit")
    check("after the research limit a check is still served (no 'call submit now')", "research limit" not in cc[1] + cc[2])

    # a failed connection does not count; a what-if gateway that raises costs only that check
    g = FakeGW([R["R3"]] * 3, fail={"whatif": [GatewayError("Disconnected from IB Gateway", 504)] * 2})
    g.connect = lambda: None                                  # the reconnect does not bring it back
    link = checks.Link(g)
    script = [("TOOL", "check_orders", {"place": [O3]}), ("TOOL", "check_orders", {"place": [O3]}), SUB]
    call, trace, _, _ = run_look(script, None, "sys", "state", "2026-11-16", SPEC, "high",
                                 checker=lambda args: checks.check(link, args, CTX, now=NOW))
    cc = [t["result"] for t in trace if t.get("tool") == "check_orders"]
    check("a lost connection: the failure text, not counted", cc[0] == ("error: the what-if connection failed (GatewayError: "
          "Disconnected from IB Gateway); this check did not count."), cc[0])
    check("the next check, served, is check 1 of 3", cc[1].endswith("(order check 1 of 3)"), cc[1][-40:])
    boom = lambda args: (_ for _ in ()).throw(RuntimeError("what-if gateway broke"))
    call, trace, _, _ = run_look([("TOOL", "check_orders", {"place": [O3]}), ("TOOL", "quote", {}), SUB], None, "sys", "state",
                                 "2026-11-16", SPEC, "high", checker=boom)
    check("a checker that raises costs only that check: error text, research and submit go on",
          trace[0]["result"].startswith("error: the order check failed (RuntimeError") and trace[1]["tool"] == "quote" and call)
    g = FakeGW([R["R3"]])
    g.whatif_many = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("socket gone"))
    logged = []
    t, c, raw = checks.check(checks.Link(g), {"place": [O3]}, CTX, now=NOW, log=logged.append)
    check("checks.check never raises: a what-if that raises gives a text, not counted, and a log line",
          not c and t.startswith("error: the what-if connection failed (RuntimeError: socket gone)") and logged, t)
    FakeRenderer.seen = []
    run_look([SUB], None, "sys", "state", "2026-11-16", SPEC, "high")
    check("no checker (text A, or no connection): no check_orders tool", [s["name"] for s in FakeRenderer.seen[0][0]] == ["quote", "submit"])
finally:
    tools.run = real_run

# ---------------------------------------------------------------- a look always reaches a submit (review, 15 Nov, finding 1)
# A fake model that calls check_orders (or alternates research and checks) for ever, until the loop opens the submit call
# for it: the forced turn's tokens start with SUBMIT_OPEN()'s (the fake tokenizer encodes it as [7]; a sample is [5]).
import random  # noqa: E402


def endless(kinds, names_checked=False):
    """parse_any for a model that makes the calls in `kinds` (cycled) until its turn is forced into submit."""
    state = {"i": 0}

    def parse(toks, stop, names):
        if toks and toks[0] == 7:                           # the forced turn: the model writes its orders
            return SUB
        kind = kinds[state["i"] % len(kinds)]
        state["i"] += 1
        if kind == "bad":
            return ("MALFORMED", None, "bad JSON")
        if names_checked and kind not in names:
            return ("MALFORMED", None, f"call to undeclared tool {kind!r}")
        return ("TOOL", kind, {"place": [O3]} if kind == "check_orders" else {})
    return parse


def look_with(parse, checker):
    rmod = types.ModuleType("tinker_cookbook.renderers.gpt_oss")
    rmod.GptOssRenderer = FakeRenderer
    patches = [(agent, "parse_any", parse), (trader, "tokenizer", lambda: fake_tok), (tools, "run", lambda gw, n, a, **k: "quote result"),
               (agent, "sample_once", lambda sc, renew, mi, params, trace, step:
                (types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=[5], stop_reason="stop")]), sc))]
    saved = [(m, n, getattr(m, n)) for m, n, _ in patches]
    saved_r = sys.modules.get("tinker_cookbook.renderers.gpt_oss")
    for m, n, v in patches:
        setattr(m, n, v)
    sys.modules["tinker_cookbook.renderers.gpt_oss"] = rmod
    try:
        return agent.gptoss_look(None, None, "sys", "state", "2026-11-16", SPEC, "high", checker=checker)
    finally:
        for m, n, v in saved:
            setattr(m, n, v)
        if saved_r is None:
            sys.modules.pop("tinker_cookbook.renderers.gpt_oss", None)
        else:
            sys.modules["tinker_cookbook.renderers.gpt_oss"] = saved_r


def links():
    """(label, checker): the what-if link working, failing on every call (504, and the reconnect does not bring it back),
    and absent (no connection this look)."""
    up = checks.Link(FakeGW([R["R3"]] * 50))
    dead = FakeGW(fail={"whatif": [GatewayError("Disconnected from IB Gateway", 504) for _ in range(200)]})
    dead.connect = lambda: None
    down = checks.Link(dead)
    return [("working", lambda a: checks.check(up, a, CTX, now=NOW)), ("failing", lambda a: checks.check(down, a, CTX, now=NOW)),
            ("absent", lambda a: checks.check(None, a, CTX, now=NOW))]


ends = []
for kinds in (["check_orders"], ["quote", "check_orders"], ["check_orders", "quote"], ["quote"] * 12 + ["check_orders"]):
    for label, checker in links():
        call, trace, _, _ = look_with(endless(kinds), checker)
        ends.append((tuple(kinds[:2]), label, call is not None and trace[-1].get("tool") == "submit"
                     and not any("no submit within the step limit" in str(t.get("error")) for t in trace), len(trace)))
check("only checks, or research and checks alternating to the end, with the link working, failing and absent: every look "
      "ends with a submit", all(e[2] for e in ends), [e for e in ends if not e[2]])
call, trace, _, _ = look_with(endless(["check_orders"]), links()[1][1])
cc = [t["result"] for t in trace if t.get("tool") == "check_orders"]
check("a dead link: 4 calls run (each 'did not count'), then the call limit with its counts, then 'call submit now' (twice, "
      "as for research: the second call is answered and the submit opened after it)", len(cc) == 7
      and all("this check did not count" in c for c in cc[:4])
      and cc[4] == "check limit reached (4 calls this look, 0 of 3 counted); no further check is run this look."
      and cc[5] == cc[6] == "check limit reached (4 calls this look, 0 of 3 counted); no further check is run this look; call submit now"
      and trace[-1]["tool"] == "submit", cc)
call, trace, _, _ = look_with(endless(["check_orders"]), links()[0][1])
cc = [t["result"] for t in trace if t.get("tool") == "check_orders"]
check("a working link: 3 counted checks, the frozen limit text, 'call submit now', then the submit",
      len(cc) == 6 and cc[2].endswith("(order check 3 of 3)") and cc[3] == checks.CAP_TEXT
      and cc[4] == cc[5] == "check limit reached (3 of 3); no further check is run this look; call submit now" and call is not None,
      cc[3:])
call, trace, _, _ = look_with(endless(["check_orders"], names_checked=True), None)
check("no checker (text A): a check_orders call is undeclared; the second malformed turn opens the submit",
      call is not None and trace[-1]["tool"] == "submit" and len(trace) == 3, trace)
rng, worst, fails = random.Random(16), 0, []
for seed in range(400):
    kinds = [rng.choice(["quote", "check_orders"]) for _ in range(rng.randint(1, 30))]
    if rng.random() < 0.5:                                  # at most one malformed turn: a second opens the submit anyway
        kinds.insert(rng.randrange(len(kinds) + 1), "bad")
    label, checker = links()[seed % 3]
    call, trace, _, _ = look_with(endless(kinds), checker)
    worst = max(worst, len([t for t in trace if t.get("step") is not None]))
    if call is None:
        fails.append((label, kinds))
check(f"400 random mixes of research, checks and a malformed turn: all reach a submit (longest look {worst} steps of "
      f"{tools.MAX_CALLS + 5 + checks.MAX_TRIES + 1})", not fails, fails[:2])

# reconnect after the 05:00 restart: 504 on account(), one reconnect, then the check runs and counts
g = FakeGW([R["R3"]], fail={"account": GatewayError("Not connected to IB Gateway (it restarts daily at 05:00 ET); call connect()", 504)})
link = checks.Link(g)
t, c, raw = checks.check(link, {"place": [O3]}, CTX, now=NOW)
check("504 on account(): reconnected once, the check runs and counts", c and g.connects == 1 and link.gen == 1
      and t.split("\n")[1] == BEFORE, t[:200])
g = FakeGW([R["R3"]], fail={"whatif": GatewayError("Connectivity between IBKR and Trader Workstation has been lost.", 1100)})
t, c, raw = checks.check(checks.Link(g), {"place": [O3]}, CTX, now=NOW)
check("1100 on the what-ifs: reconnected, served", c and g.connects == 1)
g = FakeGW(fail={"whatif": GatewayError("Refusing to send: a what-if order lost its whatIf flag")})
t, c, raw = checks.check(checks.Link(g), {"place": [O3]}, CTX, now=NOW)
check("a what-if that lost its flag (GatewayError): not sent, not counted, never placed",
      t == "error: IBKR's what-if was not sent (GatewayError: Refusing to send: a what-if order lost its whatIf flag); this check did "
      "not count." and not c and not g.placed, t)

# ---------------------------------------------------------------- failure texts
TIMEOUT = dict(R["R3"], **{k: None for k in ("init_before", "init_change", "init_after", "maint_before", "maint_change", "maint_after",
                                             "ewl_before", "ewl_change", "ewl_after", "commission")}, error="timeout")
SOXX = {"symbol": "SOXX", "action": "SELL", "quantity": 60, "type": "TRAIL", "trail_percent": 0.02, "tif": "GTC"}
t, c, raw, g = run([O1, SOXX], [R["R1"], TIMEOUT])
lines = t.split("\n")
check("one order times out: its line, and the all-orders line says 1 order is not in it", lines[4] == (
    "2. SELL 60 SOXX TRAIL 0.02% GTC: value $8,885 at the delayed last 148.09 = 0.10x your sleeve's value ($90,000); the trail is "
    "$0.03 (148.09 x 0.02 / 100). IBKR did not answer within 10 s: no IBKR figures for this order, and it is not in the all-orders "
    "line.") and lines[5] == ("All orders, computed here (not by IBKR): initial margin $546,014; equity with loan $1,369,326. "
                              "(1 order without IBKR figures is not in this line)") and c, "\n" + t)
E200 = dict(TIMEOUT, error_code=200, error="No security definition has been found for the request", warning=None)
t, c, raw, g = run([{"symbol": "XYZQ", "action": "BUY", "quantity": 10, "type": "MKT", "tif": "DAY"}], [E200])
check("IBKR error other than 201: verbatim, one line", t.split("\n")[2] == (
    "1. BUY 10 XYZQ MKT DAY: IBKR error 200: No security definition has been found for the request"), t)
t, c, raw, g = run([O3], [R["R3"]], fail={"account": GatewayError("No reply from IB Gateway to the account summary request within 10s")})
check("account() fails: its line, the orders still checked, no all-orders line", t.split("\n")[1] == (
    "Account before: IBKR's account figures were not read (timeout); the all-orders line is not computed.")
    and t.split("\n")[2].startswith("1. SELL 100 SPY TRAIL") and len(t.split("\n")) == 3 and c, t)
t, c, raw, g = run([O1, {"symbol": "NVDA", "action": "BUY", "type": "MKT", "tif": "DAY"}, O3], [R["R1"], R["R3"]])
check("a malformed order: 'not checked' with clean_order's text, the others still sent", t.split("\n")[4] == (
    "2. not checked: BUY NVDA: quantity is required") and len(g.sent[0]) == 2 and t.split("\n")[5].startswith("3. SELL 100 SPY"), t)
t, c, raw, g = run([{"symbol": "NVDA", "action": "BUY", "quantity": 10, "tif": "DAY"}, O3], [R["R3"]])
check("an order with no type: 'not checked: type is required', never checked as MKT (review, 15 Nov)",
      t.split("\n")[2] == "1. not checked: BUY NVDA: type is required" and len(g.sent[0]) == 1
      and g.sent[0][0]["order_type"] == "TRAIL", t)
t, c, raw, g = run([{"symbol": "NVDA", "action": "BUY", "type": "MKT", "tif": "DAY"}])
check("no order sent (all malformed): no all-orders line, so nothing says the figures stand",
      "All orders" not in t and not g.sent and t.split("\n")[-1] == "1. not checked: BUY NVDA: quantity is required", t)
REJ = dict(TIMEOUT, error=None, error_code=None)                   # a refusal with rejectReason only: no figures, no code
t, c, raw, g = run([O1, O3], [R["R1"], REJ])
check("a result without figures (no code) is counted as missing in the all-orders line", t.split("\n")[-1] == (
    "All orders, computed here (not by IBKR): initial margin $546,014; equity with loan $1,369,326. (1 order without IBKR figures "
    "is not in this line)"), t.split("\n")[-1])
t, c, raw, g = run([O3, O3], [R["R3"], REJ])
check("only margin-lowering orders with figures, and one without: the figures-stand sentence is not used", t.split("\n")[-1] == (
    "All orders, computed here (not by IBKR): no order with IBKR figures raises initial margin. (1 order without IBKR figures is not "
    "in this line)"), t.split("\n")[-1])
t, c, raw, g = run([O1], [TIMEOUT])
check("every order without figures: the all-orders line is not computed",
      t.split("\n")[-1] == "All orders: not computed (no order came back with IBKR figures).", t.split("\n")[-1])
E202 = dict(R["R1"], error_code=202, error="Order cancelled")
t, c, raw, g = run([O1, O1, O3], [R["R1"], E202, TIMEOUT])
check("a non-201 code that came with figures is labelled by its code, not as 'without IBKR figures'", t.split("\n")[-1] == (
    "All orders, computed here (not by IBKR): initial margin $546,014; equity with loan $1,369,326. (1 order without IBKR figures "
    "and 1 order with IBKR error 202 are not in this line)") and "IBKR error 202: Order cancelled" in t, t.split("\n")[-1])


class SlowGW(FakeGW):
    """A connected link that does not answer account summaries: each read waits its timeout (scaled down 50x here) and then
    raises the framework's 'No reply'. It holds its own lock for the wait, as tws's Gateway does."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.timeout, self._lock, self.waits = 10.0, threading.RLock(), []

    def _slow(self, what):
        with self._lock:
            self.waits.append(self.timeout)
            import time
            time.sleep(self.timeout / 50)
            raise GatewayError(f"No reply from IB Gateway to the {what} request within {self.timeout:g}s")

    def account(self):
        self.reads += 1
        return self._slow("account summary")


g = SlowGW([R["R3"]] * 30)
link = checks.Link(g)
outs = []
ths = [threading.Thread(target=lambda: outs.append(checks.check(link, {"place": [O3]}, CTX, now=NOW))) for _ in range(6)]
for th in ths:
    th.start()
for th in ths:
    th.join()
check("a slow link: account() waits 5 s, not the connection's 10, and only one read waits in a look of 6 parallel checks",
      g.waits == [checks.READ_TIMEOUT] and g.timeout == 10.0 and link.slow and len(outs) == 6, (g.waits, link.slow))
check("the slow link's checks still run, counted, and say which account facts they lack",
      all(c for _, c, _ in outs) and sum("were not read (timeout);" in t for t, _, _ in outs) == 1
      and sum("were not read (skipped after a timeout earlier in this look)" in t for t, _, _ in outs) == 5
      and all("Account: IBKR's positions or resting orders were not read for this check." in t for t, _, _ in outs), outs[0][0][:400])
t, c, raw = checks.check(checks.Link(FakeGW([R["R3"]])), {"place": [O3]}, CTX, now=NOW)
check("a fresh link (the next look) reads the account again", c and t.split("\n")[1] == BEFORE)
t, c, raw, g = run([])
check("an empty place: header, account, the no-orders line; it counts", t == "\n".join([HEAD, BEFORE, checks.EMPTY_TEXT]) and c
      and checks.EMPTY_TEXT == "No orders given; the account's figures above are all this check returns." and not g.sent)
t, c, raw = checks.check(checks.Link(FakeGW()), {"place": "SPY"}, CTX, now=NOW)
check("place not a list: a text, not counted", not c and t.startswith("error: give place"))
# 16 Nov 2026 (the gpt-oss dry look): a call wrapped under the place field's own description is repaired; nothing else is
WRAPPED = {"Orders to check": {"place": [O3]}}
call, trace, _, _ = run_look([("TOOL", "check_orders", json.loads(json.dumps(WRAPPED))), SUB], None, "sys", "state", "2026-11-16",
                             SPEC, "high", checker=lambda args, l=checks.Link(FakeGW([R["R3"]])): checks.check(l, args, CTX, now=NOW))
check("a wrapped call {\"Orders to check\": {\"place\": [...]}} is served and counted as check 1; the trace keeps it as written",
      trace[0]["result"] == GOLD["R3"].replace("(order check 3 of 3)", "(order check 1 of 3)") and trace[0]["args"] == WRAPPED
      and call is not None, trace[0]["result"][-60:])
# 16 Nov 2026 (the ML3 dry look of 15 Nov): the orders as a list straight under that one key are the call's place too
LISTED = {"Orders to check": [O3]}
call, trace, _, _ = run_look([("TOOL", "check_orders", json.loads(json.dumps(LISTED))), SUB], None, "sys", "state", "2026-11-16",
                             SPEC, "high", checker=lambda args, l=checks.Link(FakeGW([R["R3"]])): checks.check(l, args, CTX, now=NOW))
check("a wrapped list {\"Orders to check\": [...]} is served and counted as check 1; the trace keeps it as written",
      trace[0]["result"] == GOLD["R3"].replace("(order check 3 of 3)", "(order check 1 of 3)") and trace[0]["args"] == LISTED
      and call is not None, trace[0]["result"][-60:])
check("unwrap: the wrapped list and the wrapped object give the same call as the plain one; a plain call is untouched",
      checks.unwrap(LISTED) == {"place": [O3]} and checks.unwrap(WRAPPED) == {"place": [O3]}
      and checks.unwrap({"place": [O3]}) == {"place": [O3]} and agent._estimate("check_orders", LISTED) == agent._estimate("check_orders", {"place": [O3]}))
bad_shapes = [{"Orders to check": []}, {"Orders to check": ["SPY", "QQQ"]}, {"Orders to check": [O3, "x"]}, {"Orders to check": [O3], "notes": "x"},
              {"Orders to check": {"place": [O3]}, "notes": "x"}, {"Orders to check": {"orders": [O3]}}, {"Orders to check": "SPY"}]
outs = [checks.check(checks.Link(FakeGW([R["R3"]])), a, CTX, now=NOW) for a in bad_shapes]
check("not repaired: an empty list, a list of anything but objects, two keys, an inner object without place: the error text, not counted",
      all(t == "error: give place, a list of orders as for submit; this check did not count." and not c for t, c, _ in outs), outs)
t, c, raw, g = run([dict(O3, p_fill=0.7, p_win=0.6), dict(O1, p_fill=0.5)], [R["R3"], R["R1"]])
check("the model's own keys (p_fill, p_win) are left out of what is sent", all("p_fill" not in o and "p_win" not in o for o in g.sent[0])
      and c and "not checked" not in t)

# ---------------------------------------------------------------- the mapping, as PaperWorld.act maps it for paper.place
FORMS = [{"symbol": "AAA", "action": "BUY", "quantity": 5, "type": "MKT", "tif": "DAY"},
         {"symbol": "AAA", "action": "BUY", "quantity": 5, "type": "MOO", "tif": "DAY", "limit": 9.0},
         {"symbol": "AAA", "action": "BUY", "quantity": 5, "type": "LMT", "limit": 10.5, "stop": 9.0, "tif": "GTC"},
         {"symbol": "AAA", "action": "SELL", "quantity": 5, "type": "STP", "stop": 9.5, "limit": 9.0, "tif": "GTC"},
         {"symbol": "AAA", "action": "SELL", "quantity": 5, "type": "STP LMT", "stop": 9.5, "limit": 9.4, "tif": "DAY"},
         {"symbol": "AAA", "action": "SELL", "quantity": 5, "type": "TRAIL", "trail_percent": 1.5, "stop": 9.0, "tif": "GTC"},
         {"symbol": "AAA", "action": "SELL", "quantity": 5, "type": "MOC", "tif": "DAY"},
         {"symbol": "AAA", "action": "BUY", "quantity": 5, "type": "LOC", "limit": 10.1, "tif": "DAY"}]
WANT = [("MKT", None, None, "DAY", {}), ("MKT", None, None, "OPG", {}), ("LMT", 10.5, None, "GTC", {}), ("STP", None, 9.5, "GTC", {}),
        ("STP LMT", 9.4, 9.5, "DAY", {}), ("TRAIL", None, None, "GTC", {"trailingPercent": 1.5}), ("MOC", None, None, "DAY", {}),
        ("LOC", 10.1, None, "DAY", {})]
got = [checks.whatif_args(trader.clean_order(f)[0]) for f in FORMS]
check("each order type maps to whatif arguments (MOO: MKT OPG; TRAIL: trailingPercent; limit and aux only where used)",
      [(a["order_type"], a["limit"], a["aux"], a["tif"], {k: v for k, v in a.items() if k == "trailingPercent"}) for a in got] == WANT, got)


class PlacePaper:
    def __init__(self):
        self.calls = []

    def book(self, sleeve, marks=None):
        return {"positions": {}, "cash": 1e9}

    def mark(self, q):
        return q["last"]

    def place(self, gw, sleeve, symbol, action, quantity, typ, limit, aux, tif, arrival=None, probs=None, **fields):
        self.calls.append({"symbol": symbol, "action": action, "quantity": quantity, "order_type": typ, "limit": limit, "aux": aux,
                           "tif": tif, **fields})
        return {"result": {"status": "Submitted"}}

    def add_notes(self, sleeve, notes):
        pass


real_sleeves = paper.sleeves
paper.sleeves = lambda: {"s": {"max_gross": None, "text": "A"}}
try:
    w = trader.PaperWorld(types.SimpleNamespace(quotes=lambda syms: [{"symbol": s, "last": 10.0} for s in syms]), "s", None)
finally:
    paper.sleeves = real_sleeves
w.paper = PlacePaper()
w.act({"place": [dict(f) for f in FORMS], "cancel": []})
check("the mapping is PaperWorld.act's, order by order", w.paper.calls == got, w.paper.calls)
try:
    from tws.gateway import Gateway as _G
except Exception as e:                                      # noqa: BLE001
    print(f"(tws not importable offline, the Order-field check is skipped: {type(e).__name__})")
else:
    bad = []
    for a in got:
        extra = {k: v for k, v in a.items() if k not in ("symbol", "action", "quantity", "order_type", "limit", "aux", "tif")}
        try:
            _G._build_order(a["action"], a["quantity"], a["order_type"], a["limit"], a["aux"], a["tif"], "t", extra)
        except ValueError as e:
            bad.append(str(e))
    check("every mapped order passes the framework's own Order-field test (nothing sent)", not bad, bad)
    try:
        _G._build_order("BUY", 1, "MKT", None, None, "DAY", "t", {"p_fill": 0.5})
        refused = False
    except ValueError:
        refused = True
    check("and the framework refuses p_fill, which is why it is left out", refused)

# ---------------------------------------------------------------- phrases, trails, the all-orders line
T = lambda o, held=None, resting=(), positions=None, reply=None: run(
    [o], [reply or R["R3"]], ctx=dict(CTX, positions=positions or {}), held=held, resting=resting)[0].split("\n")[2]
L = T(dict(O3), held={"SPY": 82}, resting=[{"symbol": "SPY", "action": "SELL", "quantity": 18, "status": "Submitted"}])
check("A 82, R 18 resting sells, SELL 100: IBKR sees 64 and a 36-share short",
      "Account: net 82 SPY long, 18 shares in resting SPY sells; to IBKR (the resting sells counted as filled: net 64 SPY long) it "
      "closes that long and opens a 36-share short." in L, L)
L = T({"symbol": "LRCX", "action": "BUY", "quantity": 20, "type": "MKT", "tif": "DAY"}, held={"LRCX": -10},
      resting=[{"symbol": "LRCX", "action": "BUY", "quantity": 15, "status": "PreSubmitted"},
               {"symbol": "LRCX", "action": "SELL", "quantity": 7, "status": "Submitted"},
               {"symbol": "LRCX", "action": "BUY", "quantity": 99, "status": "Cancelled"}])
check("LRCX -10 with 15 BUY STP resting: a BUY sees +5 (a sell and a cancelled buy not counted)",
      "Account: net 10 LRCX short, 15 shares in resting LRCX buys; to IBKR (the resting buys counted as filled: net 5 LRCX long) it adds "
      "20 to that long." in L, L)
L = T(dict(O3, quantity=10), held={"SPY": 82}, resting=[{"symbol": "SPY", "action": "SELL", "quantity": 100, "status": "Submitted"}])
check("R larger than A", "(the resting sells counted as filled: net 18 SPY short) it adds 10 to that short." in L, L)
L = T(dict(O3, quantity=30), held={})
check("no account position", "Account: no net SPY position, no resting SPY sells; to IBKR it opens a 30-share short." in L, L)
SL = [(0, "BUY", 50, "Your sleeve holds no SPY; if it fills, it opens a 50-share long."),
      (0, "SELL", 8, "Your sleeve holds no SPY; if it fills, it opens an 8-share short."),
      (40, "BUY", 10, "Your sleeve holds 40 SPY long; if it fills, it adds 10 to your 40-share long."),
      (-40, "SELL", 11, "Your sleeve holds 40 SPY short; if it fills, it adds 11 to your 40-share short."),
      (40, "SELL", 10, "Your sleeve holds 40 SPY long; if it fills, it sells 10 of your 40-share long."),
      (-40, "BUY", 10, "Your sleeve holds 40 SPY short; if it fills, it covers 10 of your 40-share short."),
      (40, "SELL", 40, "Your sleeve holds 40 SPY long; if it fills, it closes your 40-share long."),
      (-40, "BUY", 40, "Your sleeve holds 40 SPY short; if it fills, it closes your 40-share short."),
      (60, "SELL", 100, "Your sleeve holds 60 SPY long; if it fills, it closes your 60-share long and opens a 40-share short."),
      (-60, "BUY", 78, "Your sleeve holds 60 SPY short; if it fills, it closes your 60-share short and opens an 18-share long.")]
bad = [(q0, a, n) for q0, a, n, want in SL
       if want not in T({"symbol": "SPY", "action": a, "quantity": n, "type": "STP", "stop": 700, "tif": "GTC"}, positions={"SPY": q0})]
check("every sleeve phrase, including a stop larger than the position", not bad, bad)
trails = [checks._value(trader.clean_order(dict(O3, trail_percent=p))[0], CTX).split("the trail is ")[1].split(" ")[0]
          for p in (0.01, 0.02, 1, 2)]
check("trails at 749.60: 0.01, 0.02, 1 and 2 give $0.07, $0.15, $7.50, $14.99", trails == ["$0.07", "$0.15", "$7.50", "$14.99"], trails)
AA = dict(R["R2-3733-MKT"], symbol="AAPL")
t, c, raw, g = run([O1, dict(O2, symbol="AAPL", quantity=3733)], [R["R1"], AA])
check("all orders: two opening orders in different symbols add up", t.split("\n")[-1] == (
    "All orders, computed here (not by IBKR): initial margin $1,245,606; equity with loan $1,360,929."), t.split("\n")[-1])
t, c, raw, g = run([O1, {"symbol": "SPY", "action": "SELL", "quantity": 82, "type": "STP", "stop": 712.12, "tif": "GTC"}],
                   [R["R1"], R["STP-close"]])
check("an opening order and a margin-lowering exit: the exit is left out", t.split("\n")[-1] == (
    "All orders, computed here (not by IBKR): initial margin $546,014; equity with loan $1,369,326.")
    and "initial margin after $188,440 (change -$15,367.42)" in t, t.split("\n")[-1])
t, c, raw, g = run([O3, dict(O3, type="LMT", limit=749.6, tif="DAY")], [R["R3"], R["R3-LMT"]])
check("all lowering: the figures above stand; the band notice once", t.split("\n")[-1].endswith("so the account's figures above stand.")
      and t.count("price-band") == 1 and "reject your order if" in t.split("\n")[1] and "will not receive" not in t, t)
t, c, raw, g = run([O1, O1], [R["R1"], R["R1"]], acct={k: v for k, v in ACCT.items() if k != "SMA"})
check("SMA missing from account(): 'SMA not reported'", "excess liquidity $1,167,302; SMA not reported; gross $775,595." in t)

# ---------------------------------------------------------------- the account line
INFO = {"account": dict(ACCT), "sleeves": 9, "total": 690000.0}
ST = {"value": 90000.0, "cash": 90000.0, "gross": 0.0, "net": 0.0, "beta_net": 0.0, "positions": [], "orders": [], "events": []}
LINE = ("Whole paper account (shared by the 9 sleeves, whose values total $690,000; IBKR's figures at 20:04:25 ET): net liquidation "
        "$1,371,230; equity with loan $1,370,011; initial margin $203,808; excess liquidity $1,167,302; cushion 85%; SMA $1,145,555, "
        "never below equity with loan minus half of gross ($982,213 now); gross $775,595. Your sleeve's value (at delayed last "
        "prices) is 6.6% of the account's net liquidation (at IBKR's prices); its gross exposure is 0.0% of the account's gross.")
check("the account line from the fixture, byte for byte", trader.account_line(INFO, ST, live={"SPY": 749.6}) == LINE,
      trader.account_line(INFO, ST, live={"SPY": 749.6}))
lines = trader.render_account(ST, {}, "H", {"SPY": 749.6}, None, account=INFO).split("\n")
check("placed directly after the sleeve-value line", lines[2] == LINE and lines[1].startswith("Sleeve value now"))
st2 = dict(ST, positions=[{"symbol": "SPY", "quantity": 82, "avg_cost": 700.0, "close": 749.6, "unrealized": 1.0, "days": 1}])
check("the sleeve's gross share", "its gross exposure is 7.9% of the account's gross." in trader.account_line(INFO, st2),
      trader.account_line(INFO, st2)[-80:])
noSMA = trader.account_line(dict(INFO, account={k: v for k, v in ACCT.items() if k not in ("SMA", "RegTMargin")}), ST)
check("SMA missing; RegTMargin missing gives the floor from half of gross",
      "cushion 85%; SMA not reported, never below equity with loan minus half of gross ($982,213 now); gross" in noSMA
      and "(at last close)" in noSMA, noSMA)
check("account() failing", trader.account_line({"account": None, "sleeves": 9, "total": 1.0}, ST)
      == "Whole paper account: IBKR's figures were not read for this look.")
check("no BuyingPower figure", "uying" not in LINE and "4,664,812" not in LINE)

# 16 Nov 2026: the account's positions that no sleeve holds (live.unheld_gross), one clause after the gross figure
CLAUSE_LINE = LINE.replace("gross $775,595.", "gross $775,595; $76,933 of the gross is in positions no sleeve holds.")
check("the clause after the gross figure, only above $1; 0 or None or under $1: the line unchanged, byte for byte",
      trader.account_line(dict(INFO, unheld=76933.2), ST, live={"SPY": 749.6}) == CLAUSE_LINE
      and all(trader.account_line(dict(INFO, unheld=u), ST, live={"SPY": 749.6}) == LINE for u in (0.0, None, 0.4)),
      trader.account_line(dict(INFO, unheld=76933.2), ST, live={"SPY": 749.6}))


class PosGW:
    def __init__(self, rows, fail=None):
        self.rows, self.fail, self.reads = rows, fail, 0

    def positions(self):
        self.reads, self.seen = self.reads + 1, getattr(self, "timeout", None)    # the wait the framework would use
        if self.fail:
            raise self.fail
        return [dict(r) for r in self.rows]


class MarkData:
    def features(self, d):
        return {"SPY": {"close": 749.0}, "NVDA": {"close": 166.0}, "AMAT": {"close": 380.0}}, None


ACCT_POS = [{"symbol": "SPY", "quantity": 82, "avg_cost": 700.0},      # a sleeve holds 40: 42 unexplained at 749.60
            {"symbol": "NVDA", "quantity": -270, "avg_cost": 167.4},   # a short no sleeve holds: 270 at 160.00
            {"symbol": "XYZQ", "quantity": 10, "avg_cost": 25.0},      # no mark: its average cost
            {"symbol": "TSLA", "quantity": 5, "avg_cost": 390.0},      # the sleeves are short 20: all 5 unexplained, at 400
            {"symbol": "AMAT", "quantity": 50, "avg_cost": 370.0},     # held in full by the sleeves
            {"symbol": "LRCX", "quantity": 10, "avg_cost": 220.0}]     # the sleeves hold more (30): nothing unexplained
BOOKS = {"s1": {"SPY": 40, "AMAT": 30, "TSLA": -20}, "s2": {"AMAT": 20, "LRCX": 30}}
real = (paper.sleeves, paper.book, live.log)
logs2 = []
paper.sleeves = lambda: {k: {} for k in BOOKS}
paper.book = lambda name, marks=None: {"positions": {s: {"quantity": q, "avg_cost": 1.0} for s, q in BOOKS[name].items()}, "cash": 0.0}
live.log = logs2.append
try:
    LIVE_PX = {"SPY": 749.6, "NVDA": 160.0, "TSLA": 400.0}
    u = live.unheld_gross(checks.Link(PosGW(ACCT_POS)), MarkData(), 1, LIVE_PX)
    check("unheld: a name one sleeve partly holds, a short, a name with no mark, a long against the sleeves' short",
          abs(u - (42 * 749.6 + 270 * 160.0 + 10 * 25.0 + 5 * 400.0)) < 1e-6 and round(u) == 76933, u)
    explained = [{"symbol": "SPY", "quantity": 40, "avg_cost": 700.0}, {"symbol": "AMAT", "quantity": 50, "avg_cost": 370.0},
                 {"symbol": "TSLA", "quantity": -20, "avg_cost": 400.0}, {"symbol": "LRCX", "quantity": 30, "avg_cost": 1.0}]
    u0 = live.unheld_gross(checks.Link(PosGW(explained)), MarkData(), 1, LIVE_PX)
    check("the ledgers explain every position: 0, so the line is unchanged",
          u0 == 0 and trader.account_line(dict(INFO, unheld=u0), ST, live=LIVE_PX) == LINE, u0)
    logs2.clear()
    un = live.unheld_gross(checks.Link(PosGW(ACCT_POS, fail=GatewayError("No reply from IB Gateway to the positions request within 5s"))),
                           MarkData(), 1, LIVE_PX)
    check("positions() fails: None (no clause) and one log line", un is None and len(logs2) == 1
          and logs2[0].startswith("account line: positions no sleeve holds not computed (GatewayError: No reply")
          and trader.account_line(dict(INFO, unheld=un), ST, live=LIVE_PX) == LINE, logs2)
    slow_link = checks.Link(PosGW(ACCT_POS))
    slow_link.slow = "account() gave no reply within 5 s"
    logs2.clear()
    check("after the account() read timed out on that link, positions() is skipped: no clause, one log line",
          live.unheld_gross(slow_link, MarkData(), 1, LIVE_PX) is None and slow_link.gw.reads == 0 and len(logs2) == 1, logs2)
    g = PosGW(ACCT_POS)
    g.timeout, g._lock = 30.0, threading.RLock()
    live.unheld_gross(checks.Link(g), MarkData(), 1, LIVE_PX)
    check("positions() for the clause is read with the line's 5 s bound", g.seen == checks.READ_TIMEOUT and g.timeout == 30.0
          and g.reads == 1, (g.seen, g.timeout))
finally:
    paper.sleeves, paper.book, live.log = real


class W:
    MAX_GROSS, ACCOUNT = None, INFO

    def __init__(self, text):
        self.TEXT = text

    def state(self, d, feats, live=None):
        return dict(ST)

    def symbols_in_play(self):
        return []


class D:
    def iso(self, d):
        return "2026-11-16"


a_txt, a_st = trader.account_text(W("A"), D(), 1, {})
b_txt, b_st = trader.account_text(W("B"), D(), 1, {})
check("text A: no account line even with ACCOUNT set; the default render is unchanged",
      "Whole paper account" not in a_txt and "account" not in a_st and a_txt == trader.render_account(ST, {}, a_txt.split("\n")[0], None, None))
W.ACCOUNT = dict(INFO, unheld=76933.2)
a2_txt, _ = trader.account_text(W("A"), D(), 1, {})
b2_txt, _ = trader.account_text(W("B"), D(), 1, {})
W.ACCOUNT = INFO
check("text A with the unheld figure set: unchanged (no account line); text B: the clause",
      a2_txt == a_txt and "of the gross is in positions no sleeve holds" in b2_txt and "no sleeve holds" not in a2_txt)
check("text B: the line, and the info kept in the state for replays", "Whole paper account (shared" in b_txt and b_st["account"] == INFO)

# ---------------------------------------------------------------- the MCP path and claude_look
import importlib.util  # noqa: E402


class FakeServer:                                           # stands in for mcp.server.mcpserver.MCPServer
    def __init__(self, name=None):
        self.tools = {}

    def tool(self, description=None):
        def deco(f):
            self.tools[f.__name__] = (description, f)
            return f
        return deco

    def run(self, *a):
        pass


def mcp(text, n, **env):
    """mcp_server.py as a Claude look of this text would start it (fake MCP package, env set while it imports)."""
    e = {"TRADER_SUBMIT_FILE": str(TMP / f"submit{n}.json"), "TRADER_LOG_FILE": str(TMP / f"log{n}.jsonl"), "TRADER_HAS_NEWS": "0",
         "TRADER_HAS_SCANNER": "0", "TRADER_HAS_STATS": "0", "TRADER_HAS_FORECAST": "0", "TRADER_ML_TOOL": "", "TRADER_TEXT": text,
         "TRADER_SLEEVE_VALUE": "", "TRADER_MAX_GROSS": "none", "TRADER_HAS_CHECK": "0", "TRADER_CHECK_FILE": ""} | env
    fake = {"mcp": types.ModuleType("mcp"), "mcp.server": types.ModuleType("mcp.server"),
            "mcp.server.mcpserver": types.ModuleType("mcp.server.mcpserver")}
    fake["mcp.server.mcpserver"].MCPServer = FakeServer
    saved = {k: sys.modules.get(k) for k in fake}
    old_env = {k: os.environ.get(k) for k in e}
    sys.modules.update(fake)
    os.environ.update(e)
    try:
        spec = importlib.util.spec_from_file_location(f"mcp_{text}{n}", ROOT / "mcp_server.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return mod


(TMP / "check.json").write_text(json.dumps(CTX), encoding="utf-8")
mB = mcp("B", 21, TRADER_HAS_CHECK="1", TRADER_CHECK_FILE=str(TMP / "check.json"))
mA = mcp("A", 22)
mN = mcp("B", 23)
check("MCP: check_orders only with TRADER_HAS_CHECK=1, with the spec's description",
      "check_orders" in mB.server.tools and "check_orders" not in mA.server.tools and "check_orders" not in mN.server.tools
      and mB.server.tools["check_orders"][0] == checks.CHECK_DESC)
mB._gw = FakeGW([R["R1"], R["R2"], R["R3"], R["R3"]])
mB.checks_now = NOW
outs = [mB.server.tools["check_orders"][1]([O]) for O in (O1, O2, O3, O3)]
check("MCP: runs on its own research client's Gateway, 3 checks then the limit text", [o.rsplit("\n", 1)[-1] for o in outs[:3]] ==
      ["(order check 1 of 3)", "(order check 2 of 3)", "(order check 3 of 3)"] and outs[3] == checks.CAP_TEXT
      and len(mB._gw.sent) == 3, outs[3])
logrows = [json.loads(l) for l in mB.LOG.read_text(encoding="utf-8").splitlines()]
check("MCP: each check logged (tool, args, result, IBKR's replies); the research count untouched",
      [r["tool"] for r in logrows] == ["check_orders"] * 4 and logrows[0]["ibkr"]["whatif"] and mB._calls == 0)
seen = {}
real_sub, real_exe = agent.subprocess.run, agent.claude_exe
agent.subprocess.run = lambda cmd, **k: seen.setdefault("cmds", []).append((cmd, k["env"])) or types.SimpleNamespace(returncode=0, stdout="", stderr="")
agent.claude_exe = lambda: str(TMP / "cli" / "1.0" / "x" / "claude.exe")
try:
    agent.claude_look("claude-x", "sys", "state", [], TMP / "cl-a", "max", text="A")
    agent.claude_look("claude-x", "sys", "state", [], TMP / "cl-b", "max", text="B", check_ctx=CTX)
finally:
    agent.subprocess.run, agent.claude_exe = real_sub, real_exe
(ca, ea), (cb, eb) = seen["cmds"]
sys_a, sys_b = ca[ca.index("--system-prompt") + 1], cb[cb.index("--system-prompt") + 1]
check("claude_look text A: no check tool, sentence or file", ea["TRADER_HAS_CHECK"] == "0" and "check_orders" not in sys_a
      and "check_orders" not in ca[ca.index("--allowedTools") + 1] and not (TMP / "cl-a" / "check.json").exists())
check("claude_look text B: the sentence, the tool allowed, the context file and env",
      sys_b == sys_a + " check_orders: IBKR's what-if, up to 3 calls a look, not research calls." and eb["TRADER_HAS_CHECK"] == "1"
      and "mcp__trader__check_orders" in cb[cb.index("--allowedTools") + 1]
      and json.loads(Path(eb["TRADER_CHECK_FILE"]).read_text(encoding="utf-8")) == CTX
      and json.loads((TMP / "cl-b" / "mcp.json").read_text(encoding="utf-8"))["mcpServers"]["trader"]["env"]["TRADER_HAS_CHECK"] == "1")

# ---------------------------------------------------------------- live.py: the framework switch, and the look's wiring on fakes
logs = []
real_log = live.log
live.log = logs.append
fake_tws = types.ModuleType("tws")
fake_tws.Gateway = type("Gateway", (), {"account": lambda s: {}, "positions": lambda s: []})   # no whatif_many
saved_tws = sys.modules.get("tws")
sys.modules["tws"] = fake_tws
try:
    live._whatif_off.clear()
    r1, r2 = live.open_whatif(), live.open_whatif()
finally:
    if saved_tws is None:
        sys.modules.pop("tws", None)
    else:
        sys.modules["tws"] = saved_tws
check("a framework without whatif_many: the tool is off, with one log line", r1 == r2 == (None, False)
      and len([m for m in logs if "check_orders off" in m]) == 1, logs)

import contextlib  # noqa: E402

opened = {}


class LookGW:
    def account(self):
        opened["acct_reads"] = opened.get("acct_reads", 0) + 1
        return dict(ACCT)

    def __getattr__(self, name):
        return lambda *a, **k: []


@contextlib.contextmanager
def fake_gateway():
    yield LookGW()


class FakeData:
    dates = ["20261113"]

    def __init__(self, **k):
        pass

    def features(self, d):
        return {"SPY": {"close": 749.6}}, None

    def table_rows(self, d, extra=None):
        return ["SPY"], None


class FakeWorld:
    def __init__(self, gw, name, data):
        self.name = name

    def symbols_in_play(self):
        return []


worlds, gptoss, finished = {}, {}, {}


def fake_state_text(world, data, d, header=None, live=None, book=False, cash_clause=False):
    worlds[world.name] = getattr(world, "ACCOUNT", None)
    return "head\nstate", {"value": 90000.0, "positions": [], "events": []}, {"SPY": {"close": 749.6}}


def fake_gptoss(sc, gw, system, text, date, sp, effort, **k):
    gptoss[threading.current_thread().name + str(len(gptoss))] = k
    c = k.get("checker")
    k["out"] = c({"place": [O3]}) if c else None
    return None, [], 0, 0


link_gw = FakeGW([R["R3"]] * 4)
sl = {"base-90k": {"model": "base", "start_cash": 90000.0, "text": "A"}, "cand1": {"model": "base", "start_cash": 90000.0}}
patches = [
    (paper, "sleeves", lambda: sl), (paper, "sync_fills", lambda gw: None), (paper, "book", lambda *a, **k: {"positions": {}, "cash": 90000.0}),
    (live, "gateway", fake_gateway), (live, "reconcile", lambda gw: {}), (live, "ml2_guard", lambda: None),
    (live, "open_whatif", lambda: (checks.Link(link_gw), True)), (live, "LOOKING", TMP / "LOOKING"),   # never the real marker
    (live, "LOOP", TMP / "LOOP.json"),
    (live, "finish", lambda now, name, meta, text, st, call, trace, *a, **k: finished.update({name: trace})),
    (live, "load_memo", lambda n: {}), (live, "sessions_left", lambda t: 5), (live.ml_tools, "unknown", lambda s: []),
    (live.rates, "line", lambda now: ""), (live.tinker, "ServiceClient", lambda **k: types.SimpleNamespace(
        create_sampling_client=lambda **kk: object())), (tools, "available", lambda gw: []),
    (agent, "gptoss_look", fake_gptoss), (trader, "Data", FakeData), (trader, "PaperWorld", FakeWorld),
    (trader, "state_text", fake_state_text), (trader, "tokenizer", lambda: None),
]
saved = [(m, a, getattr(m, a)) for m, a, _ in patches]
for m, a, v in patches:
    setattr(m, a, v)
try:
    live.look(dry_run=True)
finally:
    for m, a, v in saved:
        setattr(m, a, v)
    live.log = real_log
ks = list(gptoss.values())
by = {("A" if k.get("submit_tool") is trader.TOOL_LIVE_PROBS else "B"): k for k in ks}
check("live.look: text A gets no checker and no account info; text B gets both", set(by) == {"A", "B"}
      and by["A"].get("checker") is None and worlds["base-90k"] is None
      and worlds["cand1"]["account"]["NetLiquidation"] == ACCT["NetLiquidation"] and worlds["cand1"]["sleeves"] == 2
      and worlds["cand1"]["total"] == 180000.0, (finished, logs[-3:]))
check("live.look: one account() read for the look's line, on the what-if connection (one more for the check itself)",
      opened.get("acct_reads") is None and link_gw.reads == 2 and by["B"]["out"][1]
      and "which holds all 2 sleeves; 2 of them look" in by["B"]["out"][0], (opened, link_gw.reads, by["B"].get("out")))
check("live.look: the what-if connection is closed after the look", link_gw.closed)

# ---------------------------------------------------------------- token counts (gpt-oss tokenizer, offline)
try:
    from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
    tok = trader.tokenizer()
except Exception as e:                                      # noqa: BLE001
    print(f"(tokenizer not loaded offline, no token counts: {type(e).__name__}: {str(e)[:120]})")
else:
    n = lambda s: len(tok.encode(s, add_special_tokens=False))
    rend = []
    for with_check in (False, True):
        r = GptOssRenderer(tok, use_system_prompt=True, reasoning_effort="high", current_date="2026-11-16")
        specs = agent.harmony_specs(tools.SPECS + ([checks.CHECK_SPEC] if with_check else []) + [trader.TOOL_LIVE])
        msgs = r.create_conversation_prefix_with_tools(specs, "sys") + [{"role": "user", "content": "s"}]
        rend.append(len(list(r.build_generation_prompt(msgs).to_ints())))
    print(f"     tokens: description {n(checks.CHECK_DESC)}; the tool as rendered for gpt-oss {rend[1] - rend[0]}; account line "
          f"{n(LINE)}; R1 {n(GOLD['R1'])}, R2 {n(GOLD['R2'])}, R3 {n(GOLD['R3'])}; Claude sentence {n(checks.CLAUDE_LINE)}")
# ---------------------------------------------------------------- IBKR error 103, duplicate order id (Mon 16 Nov 2026)
# The row as whatif_many gave it at 09:30 today (runs/live/2026-11-16): no figures, error_code 103.
DUP = {"symbol": "NVDA", "action": "SELL", "quantity": 60, "order_type": "MKT", "limit": None, "aux": None, "tif": "DAY",
       "init_before": None, "init_change": None, "init_after": None, "maint_before": None, "maint_change": None,
       "maint_after": None, "ewl_before": None, "ewl_change": None, "ewl_after": None, "outside_rth": None, "commission": None,
       "min_commission": None, "max_commission": None, "commission_currency": None, "warning": None, "error_code": 103,
       "error": "Duplicate order id: 2789"}
SPYB = {"symbol": "SPY", "action": "BUY", "quantity": 5, "type": "MKT", "tif": "DAY"}
SPYS = {"symbol": "SPY", "action": "SELL", "quantity": 5, "type": "MKT", "tif": "DAY"}
text, counted, raw, g = run([SPYB, SPYS], [dict(DUP), dict(DUP), dict(DUP), dict(DUP)])
check("103 on every order, twice: the check does not count, says it is our fault and not IBKR's answer, and was retried once",
      not counted and text == checks.DUP_TEXT and len(g.sent) == 2 and len(g.sent[1]) == 2 and not g.placed
      and raw["whatif_first"][0]["error_code"] == 103, (text, counted, len(g.sent)))
text, counted, raw, g = run([SPYB, SPYS], [dict(DUP), dict(DUP), dict(R["R3"]), dict(R["R3"])])
first, again = run([SPYB, SPYS], [dict(R["R3"]), dict(R["R3"])])[0], text
check("103 on every order, then clean on the retry: the check counts and reads exactly as one that was clean the first time",
      counted and again == first and len(g.sent) == 2 and "103" not in text, text[-200:])
text, counted, raw, g = run([SPYB, SPYS], [dict(R["R3"]), dict(DUP), dict(DUP)])
rows = text.split("\n")
check("103 on one order of two, still 103 on its retry: the check counts; that order's line says it was not checked, our "
      "fault, not IBKR's answer; the other order has IBKR's figures; the all-orders line leaves it out",
      counted and len(g.sent) == 2 and g.sent[1] == [g.sent[0][1]]
      and any(l.startswith("2. ") and l.endswith(checks.DUP_ROW + ".") for l in rows)
      and any(l.startswith("1. ") and "IBKR: initial margin after" in l for l in rows)
      and rows[-1].startswith("All orders") and "1 order without IBKR figures" in rows[-1], text[-500:])
text, counted, raw, g = run([SPYB], [dict(R["R3"])])
check("no 103: one what-if call, as before", counted and len(g.sent) == 1 and "whatif_first" not in raw)
o103 = dict(DUP, error_code=201, error="margin")            # a 201 with no figures is IBKR's answer: not retried
text, counted, raw, g = run([SPYB], [o103])
check("a 201 is not a 103: not retried, counted, shown as IBKR's refusal", counted and len(g.sent) == 1 and checks.REFUSAL in text)

print("ALL OK" if all(ok) else f"FAILED ({ok.count(False)})")
sys.exit(0 if all(ok) else 1)
