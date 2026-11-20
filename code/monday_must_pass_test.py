"""The two tests that must pass before Mon 16 Nov 2026 (the Researcher's frozen list, models-w3 1bbf8c7: "the two most likely to
break on Monday, and the test for each"). Offline: no IB Gateway, no Tinker, no Claude CLI; nothing under runs/ or data/ is
written (the look's records, log and LOOKING marker are fakes or go to a temp folder; one recorded look file is only read).

1. check_orders and the account line on a what-if gateway that raises GatewayError 504, then 1100, on account() and on
   whatif_many: the real live.look() runs for one text-A and two text-B gpt-oss sleeves (the real agent.gptoss_look with a
   scripted model); every sleeve reaches its submit; the account line and each affected check print their failure texts; the
   link reconnects and a later check works; place() is never called on the what-if connection (its place raises); a
   reconnect that itself fails costs that check only.
2. The text-A control on a recorded state (runs/live/2026-11-13/base-90k-1030.json, a no-tool base sleeve): its system text,
   steps line, account text (rendered from the stored state record and events), user message, submit tool, tool specs and
   the whole renderer input of gptoss_look are byte-identical to what fd95e2a's own trader.py, agent.py, tools.py and
   mcp_server.py give for the same inputs; the account text also matches the recorded text line for line; and the same
   state for text B differs only by what text B adds (the account line, the steps line's probabilities, the submit form).

    python monday_must_pass_test.py      # ALL OK and exit 0, else exit 1
"""
import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import types
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import agent  # noqa: E402
import checks  # noqa: E402
import live  # noqa: E402
import paper  # noqa: E402
import tools  # noqa: E402
import trader  # noqa: E402

try:
    from tws import GatewayError                            # the framework's own class (importing it connects nothing)
except Exception:                                           # noqa: BLE001
    class GatewayError(Exception):                          # as tws.GatewayError: "<code>: <message>"
        def __init__(self, message, code=None):
            super().__init__(f"{code}: {message}" if code is not None else message)
            self.code = code

ok = []
TMP = Path(tempfile.mkdtemp())
REF = "fd95e2a"


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail and not good else ""))


FIX = json.loads(Path("C:/Claude/TWS-Gateway-handoff/whatif_fixture_20261115.json").read_text(encoding="utf-8"))
R3, ACCT = FIX["whatifs"]["R3"]["result"], FIX["account"]
E504 = lambda: GatewayError("Not connected to IB Gateway (it restarts daily at 05:00 ET); call connect() to reconnect", 504)
E1100 = lambda: GatewayError("Connectivity between IBKR and Trader Workstation has been lost.", 1100)


# ================================================================ 1. the what-if connection after the 05:00 ET restart
class WhatifGW:
    """trader-whatif (17), as live.open_whatif opens it through `tws.Gateway`: scripted failures per method (popped one per
    call), connect() counted (and optionally failing), place() raising and recorded."""
    made = []
    plan = {}

    def __init__(self, client, timeout=10.0):
        self.client, self.timeout, self._lock = client, timeout, threading.RLock()
        self.fail = {k: list(v) for k, v in WhatifGW.plan.items()}
        self.connects, self.placed, self.closed, self.whatifs, self.connect_fail = 0, [], False, 0, []
        WhatifGW.made.append(self)

    def _maybe(self, what):
        q = self.fail.get(what)
        if q:
            e = q.pop(0)
            if e is not None:
                raise e()

    def connect(self):
        self.connects += 1
        if self.connect_fail:
            raise self.connect_fail.pop(0)()

    def close(self):
        self.closed = True

    def account(self):
        self._maybe("account")
        return dict(ACCT)

    def positions(self):
        self._maybe("positions")
        return [{"symbol": "SPY", "quantity": 82}]

    def all_open_orders(self):
        return []

    def whatif_many(self, orders, timeout=10.0):
        self._maybe("whatif")
        self.whatifs += 1
        return [dict(R3) for _ in orders]

    def place(self, *a, **k):
        self.placed.append(a)
        raise AssertionError("place() called on the what-if connection")


class Client10:
    """client 10 (the look's own connection): orders are placed here; everything else answers empty."""

    def __init__(self):
        self.placed = []

    def place(self, *a, **k):
        self.placed.append(a)
        return {"status": "Submitted"}

    def __getattr__(self, name):
        return lambda *a, **k: []


C10 = []


@contextlib.contextmanager
def fake_gateway():
    g = Client10()
    C10.append(g)
    yield g


class FakeData:
    dates = ["20261113"]

    def __init__(self, **k):
        pass

    def features(self, d):
        return {"SPY": {"close": 749.6}}, None

    def table_rows(self, d, extra=None):
        return ["SPY"], None

    def iso(self, d):
        return "2026-11-13"


ST = {"value": 90000.0, "cash": 90000.0, "gross": 0.0, "net": 0.0, "beta_net": 0.0, "positions": [], "orders": [], "events": []}
SLEEVES = {"base-90k": {"model": "m-A", "start_cash": 90000.0, "text": "A", "max_gross": None},
           "cand1": {"model": "m-B1", "start_cash": 90000.0}, "cand2": {"model": "m-B2", "start_cash": 90000.0}}


class FakeWorld:
    """PaperWorld's surface for live.look: the text from sleeves.json, a fixed state, and act() placing on client 10."""

    def __init__(self, gw, name, data):
        self.gw, self.name = gw, name
        meta = paper.sleeves().get(name) or {}
        self.TEXT, self.MAX_GROSS, self.SHOCK_CAP = trader.text_of(meta), trader.max_gross_of(meta), None

    def state(self, d, feats, live=None):
        return json.loads(json.dumps(ST))

    def symbols_in_play(self):
        return []

    def act(self, call):
        for o in call.get("place", []):
            self.gw.place(o["symbol"], o["action"], o["quantity"])
        return []


def fake_state_text(world, data, d, header=None, live=None, book=False, cash_clause=False):
    acct, st = trader.account_text(world, data, d, {"SPY": {"close": 749.6}}, header, live, book=book, cash_clause=cash_clause)
    return acct + "\n\nMarket table (test):\nSPY 749.60", st, {"SPY": {"close": 749.6}}


class FakeRenderer:
    seen = []

    def __init__(self, tok, **k):
        pass

    def create_conversation_prefix_with_tools(self, specs, system):
        FakeRenderer.seen.append(json.dumps(specs))
        return [{"role": "system", "content": system}]

    def build_generation_prompt(self, msgs):
        return types.SimpleNamespace(to_ints=lambda: [1, 2, 3])


fake_tok = types.SimpleNamespace(decode=lambda t: "", encode=lambda s, add_special_tokens=False: [7])
who = threading.local()
RUNNING = {}                                                # each model's script in progress, by model name
O3 = {"symbol": "SPY", "action": "SELL", "quantity": 100, "type": "TRAIL", "trail_percent": 0.02, "tif": "GTC"}
BUY = {"symbol": "SPY", "action": "BUY", "quantity": 5, "type": "MKT", "tif": "DAY"}
SCRIPTS = {"m-A": [("TOOL", "quote", {"symbols": ["SPY"]}), ("TOOL", "submit", {"place": [BUY], "cancel": []})],
           "m-B1": [("TOOL", "check_orders", {"place": [O3]})] * 3 + [("TOOL", "submit", {"place": [BUY], "cancel": []})],
           "m-B2": [("TOOL", "check_orders", {"place": [O3]})] * 2 + [("TOOL", "submit", {"place": [BUY], "cancel": []})]}


def fake_sample_once(sc, renew, model_input, params, trace, step):
    who.name = sc.name
    return types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=[5], stop_reason="stop")]), sc


def fake_parse(toks, stop, names):
    if toks and toks[0] == 7:                               # a forced submit (not expected here)
        return ("TOOL", "submit", {"place": [], "cancel": []})
    it = RUNNING.setdefault(who.name, iter(SCRIPTS[who.name]))
    kind, name, args = next(it)
    if name not in names:
        return ("MALFORMED", None, f"call to undeclared tool {name!r}")
    return kind, name, json.loads(json.dumps(args))


class FakeService:
    def __init__(self, **k):
        pass

    def create_sampling_client(self, base_model=None, model_path=None):
        return types.SimpleNamespace(name=model_path or base_model)


NOW = datetime(2026, 11, 16, 10, 31, 7, tzinfo=live.ET)


class FakeDT(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


def look_once(plan, connect_fail=()):
    """One real live.look() on the fakes; returns (finished, logs, the what-if gateway, client-10 gateways)."""
    finished, logs = {}, []
    WhatifGW.made, WhatifGW.plan, FakeRenderer.seen = [], plan, []
    RUNNING.clear()
    C10.clear()
    fake_tws = types.ModuleType("tws")
    fake_tws.Gateway, fake_tws.GatewayError = WhatifGW, GatewayError
    rmod = types.ModuleType("tinker_cookbook.renderers.gpt_oss")
    rmod.GptOssRenderer = FakeRenderer
    real_init = WhatifGW.__init__

    def init(self, *a, **k):
        real_init(self, *a, **k)
        self.connect_fail = list(connect_fail)
    patches = [
        (paper, "sleeves", lambda: SLEEVES), (paper, "sync_fills", lambda gw: None),
        (paper, "book", lambda *a, **k: {"positions": {}, "cash": 90000.0}),
        (live, "gateway", fake_gateway), (live, "reconcile", lambda gw: {}), (live, "ml2_guard", lambda: None),
        (live, "account_margin", lambda gw: None), (live, "LOOKING", TMP / "LOOKING"), (live, "GPTOSS_LANES", 1),
        (live, "LOOP", TMP / "LOOP.json"),
        (live, "finish", lambda now, name, meta, text, st, call, trace, notes, *a, **k: finished.update(
            {name: {"text": text, "call": call, "trace": trace, "st": st}})),
        (live, "log", logs.append), (live, "load_memo", lambda n: {}), (live, "sessions_left", lambda t: 5),
        (live, "datetime", FakeDT), (live.ml_tools, "unknown", lambda s: []), (live.rates, "line", lambda now: ""),
        (live.tinker, "ServiceClient", FakeService), (live.time, "sleep", lambda s: None),
        (tools, "available", lambda gw: [{"name": "quote", "description": "q", "parameters": {"type": "object", "properties": {}}}]),
        (tools, "run", lambda gw, name, args, **k: "quote result"),
        (agent, "sample_once", fake_sample_once), (agent, "parse_any", fake_parse),
        (trader, "Data", FakeData), (trader, "PaperWorld", FakeWorld), (trader, "state_text", fake_state_text),
        (trader, "tokenizer", lambda: fake_tok), (WhatifGW, "__init__", init),
    ]
    saved = [(m, a, getattr(m, a)) for m, a, _ in patches]
    saved_mods = {k: sys.modules.get(k) for k in ("tws", "tinker_cookbook.renderers.gpt_oss")}
    for m, a, v in patches:
        setattr(m, a, v)
    sys.modules["tws"], sys.modules["tinker_cookbook.renderers.gpt_oss"] = fake_tws, rmod
    try:
        live.look(dry_run=False)
    finally:
        for m, a, v in saved:
            setattr(m, a, v)
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return finished, logs, (WhatifGW.made or [None])[0], list(C10)


def checks_of(f, name):
    return [t["result"] for t in f[name]["trace"] if t.get("tool") == "check_orders"]


FAIL_1100 = ("error: the what-if connection failed (GatewayError: Connectivity between IBKR and Trader Workstation has been lost.); "
             "this check did not count.")
NO_LINE = "Whole paper account: IBKR's figures were not read for this look."
# account(): 504 at the account line's read and 1100 on its retry; 504 at cand1's first check. whatif_many: 1100 on that
# check's retry; 504, then 1100, on cand1's second check. Everything after works.
PLAN = {"account": [E504, E1100, E504], "whatif": [E1100, E504, E1100]}
f, logs, wg, c10 = look_once(PLAN)
check("504/1100: the look runs for every sleeve, each to its submit (one text A, two text B)",
      set(f) == set(SLEEVES) and all(f[n]["call"] and f[n]["trace"][-1].get("tool") == "submit" for n in SLEEVES),
      {n: (f[n]["call"], f[n]["trace"][-1:]) for n in f})
check("504/1100 on account() before the looks: the account line's failure text for text B, logged; no line for text A",
      all(NO_LINE in f[n]["text"] for n in ("cand1", "cand2")) and "Whole paper account" not in f["base-90k"]["text"]
      and any(m.startswith("account line: IBKR's figures not read for this look (GatewayError: Connectivity") for m in logs),
      [m for m in logs if "account line" in m])
c1, c2 = checks_of(f, "cand1"), checks_of(f, "cand2")
check("each affected check prints its failure text (504 then 1100 on one check, twice), and a later check works",
      c1[:2] == [FAIL_1100, FAIL_1100] and c1[2].endswith("(order check 1 of 3)") and "IBKR: initial margin after $192,488" in c1[2],
      c1)
check("the other text-B sleeve's checks all work after the reconnects", len(c2) == 2 and c2[0].endswith("(order check 1 of 3)")
      and c2[1].endswith("(order check 2 of 3)") and c2[0].split("\n")[1].startswith("Account before (IBKR): net liquidation $1,371,230"),
      c2)
check("the link reconnected (gw.connect()) once at the account line and once on each failed check: 3",
      wg is not None and wg.connects == 3 and wg.client == "trader-whatif", wg and wg.connects)
check("place() is never called on the what-if connection; the orders went to client 10 (one each)",
      not wg.placed and sorted(p[0] for g in c10 for p in g.placed) == ["SPY"] * 3, (wg.placed, [g.placed for g in c10]))
check("text A: no check_orders offered, no check run; text B: offered",
      ['"check_orders"' in s for s in FakeRenderer.seen] == [False, True, True] and not checks_of(f, "base-90k"),
      ['"check_orders"' in s for s in FakeRenderer.seen])
check("the what-if connection is closed after the look, and the LOOKING marker is gone", wg.closed
      and not (TMP / "LOOKING").exists() and not (TMP / f"LOOKING.{os.getpid()}").exists())

# the reconnect itself fails (the Gateway is still restarting): that check, and that look's account line, only
f, logs, wg, c10 = look_once({"account": [E504], "whatif": [E1100]}, connect_fail=[E504, lambda: GatewayError(
    "Couldn't connect to IB Gateway at 127.0.0.1:4002. Is it running and logged in to paper? It restarts daily at 05:00 ET.", 502)])
c1 = checks_of(f, "cand1")
check("a reconnect that fails costs the account line and one check; the look goes on and later checks work",
      all(NO_LINE in f[n]["text"] for n in ("cand1", "cand2")) and c1[0].startswith("error: the what-if connection failed (GatewayError: ")
      and c1[0].endswith("this check did not count.") and c1[1].endswith("(order check 1 of 3)")
      and all(f[n]["call"] for n in SLEEVES) and not wg.placed, c1[:2])

# no failures: the line is IBKR's, every check counts
f, logs, wg, c10 = look_once({})
check("with no failures: the account line from IBKR's figures and every check counted, no reconnect",
      all("Whole paper account (shared by the 3 sleeves, whose values total $270,000; IBKR's figures at" in f[n]["text"]
          for n in ("cand1", "cand2")) and checks_of(f, "cand1")[2].endswith("(order check 3 of 3)") and wg.connects == 0
      and not wg.placed, checks_of(f, "cand1")[-1][-60:])
check("the account's 82 SPY that no sleeve's ledger holds: the clause after the gross figure, at the look's mark",
      all("gross $775,595; $61,467 of the gross is in positions no sleeve holds. Your sleeve's value" in f[n]["text"]
          for n in ("cand1", "cand2")) and "no sleeve holds" not in f["base-90k"]["text"],
      [l for l in f["cand1"]["text"].split("\n") if l.startswith("Whole paper")])
check("the check's own lines are unchanged by it", "no sleeve holds" not in "".join(checks_of(f, "cand1") + checks_of(f, "cand2")))


# ================================================================ 2. the text-A control on a recorded state
def load(path, name, inject=None, env=None):
    """fd95e2a's `path` as a module (from git, into a temp file), with `inject` in sys.modules while it imports."""
    src = subprocess.run(["git", "show", f"{REF}:{path}"], capture_output=True, text=True, encoding="utf-8", cwd=ROOT).stdout
    assert src, path
    p = TMP / f"{name}.py"
    p.write_text(src, encoding="utf-8")
    saved = {k: sys.modules.get(k) for k in (inject or {})}
    old_env = {k: os.environ.get(k) for k in (env or {})}
    sys.modules.update(inject or {})
    os.environ.update(env or {})
    try:
        spec = importlib.util.spec_from_file_location(name, p)
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


old_tools = load("tools.py", "tools_ref")
old_tools.LISTS = tools.LISTS
old_trader = load("trader.py", "trader_ref", {"tools": old_tools})
old_agent = load("agent.py", "agent_ref", {"tools": old_tools, "trader": old_trader})
dumps = lambda x: json.dumps(x, ensure_ascii=False)

REC_FILE = ROOT / "runs" / "live" / "2026-11-13" / "base-90k-1030.json"     # read only
rec = json.loads(REC_FILE.read_text(encoding="utf-8"))
stored = rec["state_text"]
check("the recorded look is a no-tool base sleeve's", rec["model"] == "base" and not rec.get("ml_tool") and rec["dry_run"] is False)

# its parts, as live.py built them on 13 Nov: steps + header + goal (sleeve, Rates) + memo + account + market table
steps_rec, rest = stored[:stored.index("Date: ")], stored[stored.index("Date: "):]
lines = rest.split("\n")
header = lines[0]
i_memo = next(i for i, l in enumerate(lines) if l.startswith("Your notes from your last look ("))
goal = "\n".join(lines[1:i_memo])
memo_line = lines[i_memo]
when = memo_line[len("Your notes from your last look ("):memo_line.index("): ")]
memo = {"when": when, "notes": memo_line[memo_line.index("): ") + 3:]}
acct_rec = "\n".join([header] + lines[i_memo + 1:]).split("\n\nMarket table (")[0]
table_part = rest[rest.index("\n\nMarket table ("):]
shock = next(l for l in acct_rec.split("\n") if l.startswith("Single-shock limit"))
table_lines = table_part.split("\n")[4:]                    # after the label and the column names
live_px, feats = {}, {}
for l in table_lines:                                       # sym (may hold a space, "BRK B") close r1 r5 r20 r60 vol off_hi beta last today
    parts = l.split()
    try:
        sym, nums = " ".join(parts[:-10]), [float(x) for x in parts[-10:]]
    except ValueError:
        continue
    feats[sym], live_px[sym] = {"close": nums[0], "vol": nums[5]}, nums[8]
st = dict(rec["state"], events=list(rec["events"]))
# Each held name's vol as the recorded risk line used it: the table prints vol to 0.1, so it is solved from that line's own
# dollar figure (vol = loss x 100 / the position's value), which renders the same dollars and the same 0.1 vol again.
size = {p["symbol"]: abs(p["quantity"]) * p["close"] for p in st["positions"]}
for l in acct_rec.split("\n"):
    if l.startswith("  ") and "% of the sleeve; a " in l and " move = " in l:
        sym = l.strip().split(" ")[0]
        loss = float(l.split(" move = ")[1].split(" = ")[0].replace("$", "").replace(",", ""))
        feats[sym] = dict(feats.get(sym, {}), vol=loss * 100 / size[sym])
mg = trader.max_gross_of({})                                # 13 Nov: base-90k had no "max_gross", so the 2x term
new_acct = trader.render_account(json.loads(json.dumps(st)), feats, header, live_px, mg, shock=shock)
old_acct = old_trader.render_account(json.loads(json.dumps(st)), feats, header, live_px, mg, shock=shock)
check("A: the account text from the stored state and events is fd95e2a's, byte for byte", new_acct == old_acct)
check("A: and it is the recorded text's account part, line for line", new_acct == acct_rec,
      [(a, b) for a, b in zip(new_acct.split("\n"), acct_rec.split("\n")) if a != b][:2])
mon = trader.render_account(json.loads(json.dumps(st)), feats, header, live_px, None)    # Monday's control terms: no 2x, no shock
check("A: with Monday's control terms (max_gross None, no shock cap) too", mon == old_trader.render_account(
    json.loads(json.dumps(st)), feats, header, live_px, None))

steps_new, steps_old = trader.live_steps(False, probs=True), old_trader.live_steps(False)
check("A: the steps line is fd95e2a's and the recorded one", steps_new == steps_old == steps_rec)
day = date(2026, 11, 13)
sys_new = trader.live_system(90000.0, day, ml=False, kelly=bool(rec.get("kelly_prompt")), max_gross=mg)
sys_old = old_trader.live_system(90000.0, day, ml=False, kelly=bool(rec.get("kelly_prompt")), max_gross=mg)
check("A: the system text (the recorded sleeve's terms, and Monday's: no Kelly paragraph, max_gross None) is fd95e2a's",
      sys_new == sys_old and trader.live_system(90000.0, date(2026, 11, 16), max_gross=None)
      == old_trader.live_system(90000.0, date(2026, 11, 16), max_gross=None))
user_new = trader.live_user(new_acct + table_part, goal, memo, steps=steps_new)
user_old = old_trader.live_user(old_acct + table_part, goal, memo, steps=steps_old)
check("A: the user message is fd95e2a's, and the recorded state_text", user_new == user_old == stored)
check("A: the submit tool is fd95e2a's TOOL_LIVE", dumps(trader.TOOL_LIVE_PROBS) == dumps(old_trader.TOOL_LIVE))


class LiveGW:                                               # what tools.available sees on the live client-10 Gateway
    def __getattr__(self, name):
        if name in ("news_archive", "tape_sources"):
            raise AttributeError(name)
        return lambda *a, **k: []


specs_new, specs_old = tools.available(LiveGW()), old_tools.available(LiveGW())
check("A: the research tool specs are fd95e2a's (the seven live tools)", dumps(specs_new) == dumps(specs_old)
      and [s["name"] for s in specs_new] == ["quote", "bars", "news", "tape", "article", "stats", "scanner"])
check("A: the specs as rendered for gpt-oss (harmony_specs, with submit) are fd95e2a's",
      dumps(agent.harmony_specs(specs_new + [trader.TOOL_LIVE_PROBS])) == dumps(old_agent.harmony_specs(specs_old + [old_trader.TOOL_LIVE])))


class CaptureRenderer:
    seen = []

    def __init__(self, tok, **k):
        CaptureRenderer.seen.append(("init", sorted(k.items())))

    def create_conversation_prefix_with_tools(self, specs, system):
        CaptureRenderer.seen.append(("specs", dumps(specs), system))
        return [{"role": "system", "content": system}]

    def build_generation_prompt(self, msgs):
        CaptureRenderer.seen.append(("msgs", dumps(msgs)))
        return types.SimpleNamespace(to_ints=lambda: [1, 2, 3])


def renderer_input(mod_agent, mod_trader, *a, **k):
    rmod = types.ModuleType("tinker_cookbook.renderers.gpt_oss")
    rmod.GptOssRenderer = CaptureRenderer
    patches = [(mod_agent, "parse_any", lambda toks, stop, names: ("TOOL", "submit", {"place": [], "cancel": []})),
               (mod_trader, "tokenizer", lambda: fake_tok),
               (mod_agent, "sample_once", lambda sc, renew, mi, params, trace, step:
                (types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=[5], stop_reason="stop")]), sc))]
    saved = [(m, n, getattr(m, n)) for m, n, _ in patches]
    saved_r = sys.modules.get("tinker_cookbook.renderers.gpt_oss")
    for m, n, v in patches:
        setattr(m, n, v)
    sys.modules["tinker_cookbook.renderers.gpt_oss"] = rmod
    CaptureRenderer.seen = []
    try:
        mod_agent.gptoss_look(None, LiveGW(), *a, **k)
    finally:
        for m, n, v in saved:
            setattr(m, n, v)
        if saved_r is None:
            sys.modules.pop("tinker_cookbook.renderers.gpt_oss", None)
        else:
            sys.modules["tinker_cookbook.renderers.gpt_oss"] = saved_r
    return list(CaptureRenderer.seen)


# live.py's call for text A: submit_tool=TOOL_LIVE_PROBS, no checker; fd95e2a's had neither argument
r_new = renderer_input(agent, trader, sys_new, user_new, "2026-11-13", specs_new, "high", submit_tool=trader.TOOL_LIVE_PROBS,
                       sleeve_value=st["value"], look=("base-90k", "2026-11-13T10:30:00-05:00"), checker=None)
r_old = renderer_input(old_agent, old_trader, sys_old, user_old, "2026-11-13", specs_old, "high")
check("A: everything gptoss_look hands the renderer (options, tools, system, user message) is fd95e2a's", r_new == r_old and len(r_new) == 3,
      [x[0] for x in r_new])

# the Claude path's submit tool for text A (mcp_server.py), as text_ab_test checks it; here from the same fd95e2a commit
fake_mcp = {"mcp": types.ModuleType("mcp"), "mcp.server": types.ModuleType("mcp.server"),
            "mcp.server.mcpserver": types.ModuleType("mcp.server.mcpserver")}


class FakeServer:
    def __init__(self, name=None):
        self.tools = {}

    def tool(self, description=None):
        def deco(f):
            self.tools[f.__name__] = (description, f)
            return f
        return deco

    def run(self, *a):
        pass


fake_mcp["mcp.server.mcpserver"].MCPServer = FakeServer
env = {"TRADER_SUBMIT_FILE": str(TMP / "submit.json"), "TRADER_LOG_FILE": str(TMP / "log.jsonl"), "TRADER_HAS_NEWS": "1",
       "TRADER_HAS_SCANNER": "1", "TRADER_HAS_STATS": "1", "TRADER_HAS_FORECAST": "0", "TRADER_ML_TOOL": "", "TRADER_TEXT": "A",
       "TRADER_SLEEVE_VALUE": "", "TRADER_MAX_GROSS": "none", "TRADER_HAS_CHECK": "0", "TRADER_CHECK_FILE": ""}
m_ref = load("mcp_server.py", "mcp_ref", dict(fake_mcp, tools=old_tools), env=env)
src = (ROOT / "mcp_server.py").read_text(encoding="utf-8")
(TMP / "mcp_now.py").write_text(src, encoding="utf-8")
saved = {k: sys.modules.get(k) for k in fake_mcp}
old_env = {k: os.environ.get(k) for k in env}
sys.modules.update(fake_mcp)
os.environ.update(env)
try:
    spec = importlib.util.spec_from_file_location("mcp_now", TMP / "mcp_now.py")
    m_now = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m_now)
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
check("A (Claude path): every tool and description the MCP server offers is fd95e2a's",
      {k: v[0] for k, v in m_now.server.tools.items()} == {k: v[0] for k, v in m_ref.server.tools.items()})

# the same recorded state for text B: only what text B adds
INFO = {"account": dict(ACCT), "sleeves": 9, "total": 690000.0}
b_acct = trader.render_account(json.loads(json.dumps(st)), feats, header, live_px, mg, shock=shock, cash_clause=True, account=INFO)
a_l, b_l = new_acct.split("\n"), b_acct.split("\n")
added = [l for l in b_l if l not in a_l]
check("B, the same state: the account text is text A's plus the account line after the sleeve-value line, nothing else",
      len(b_l) == len(a_l) + 1 and b_l[2].startswith("Whole paper account (shared by the 9 sleeves") and b_l[:2] + b_l[3:] == a_l
      and added == [b_l[2]], added)
neg = dict(st, cash=-1000.0)
b_neg = trader.render_account(json.loads(json.dumps(neg)), feats, header, live_px, mg, shock=shock, cash_clause=True, account=INFO)
a_neg = trader.render_account(json.loads(json.dumps(neg)), feats, header, live_px, mg, shock=shock)
check("B, with negative cash: the clause follows the cash figure, and the account line; nothing else changes",
      b_neg.split("\n")[1] == a_neg.split("\n")[1].replace("). Gross", ")." + trader.CASH_NEGATIVE + " Gross")
      and b_neg.split("\n")[:2] + b_neg.split("\n")[3:] == [a_neg.split("\n")[0], b_neg.split("\n")[1]] + a_neg.split("\n")[2:])
user_b = trader.live_user(b_acct + table_part, goal, memo, steps=trader.live_steps(False))
check("B: the user message differs from text A's only in the steps line (no p_fill/p_win) and the account line",
      user_b.replace(trader.live_steps(False), steps_new, 1).replace("\n" + b_l[2], "", 1) == user_new and user_b != user_new)
check("B: the system text is text A's; the submit form is text B's", trader.live_system(90000.0, day, kelly=True, max_gross=mg) == sys_new
      and dumps(trader.TOOL_LIVE) != dumps(trader.TOOL_LIVE_PROBS))

# the first prompt's tokens, A and B, on this recorded state (gpt-oss tokenizer, offline, if it loads)
try:
    from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
    tok = trader.tokenizer()
except Exception as e:                                      # noqa: BLE001
    print(f"(tokenizer not loaded offline, no token counts: {type(e).__name__}: {str(e)[:120]})")
else:
    n = {}
    for text, specs, user in (("A", specs_new + [trader.TOOL_LIVE_PROBS], user_new),
                              ("B", specs_new + [checks.CHECK_SPEC, trader.TOOL_LIVE], user_b)):
        r = GptOssRenderer(tok, use_system_prompt=True, reasoning_effort="high", current_date="2026-11-13")
        msgs = r.create_conversation_prefix_with_tools(agent.harmony_specs(specs), sys_new) + [{"role": "user", "content": user}]
        n[text] = len(list(r.build_generation_prompt(msgs).to_ints()))
    print(f"     tokens, first prompt on base-90k-1030's state: text A {n['A']}, text B {n['B']} ({n['B'] - n['A']:+d}; "
          "B with check_orders and the account line)")
print("ALL OK" if all(ok) else f"FAILED ({ok.count(False)})")
sys.exit(0 if all(ok) else 1)
