"""The A/B rule of Mon 16 Nov 2026 (Clayton, 14 Nov): text A, the control (base-90k, "text": "A"), reads exactly what it read
at fd95e2a; text B gets the new texts (no p_fill/p_win, the trailing-stop unit, the cash clause, the order echo), each
exactly as specified and nowhere else. Plus the changes for all sleeves: an order that still carries p_fill and p_win is
accepted, "symbols" given as a string, and the forecast tool's sleeve value and look key wired from both look paths. Text A
is tested WITH the agreed exception (trader.text_of): its results differ from fd95e2a's only for a string "symbols", a bare
or lower-case article id, and an enveloped call.
Offline: fd95e2a's trader.py, agent.py, tools.py and mcp_server.py are loaded from git into a temp folder as the
reference; fakes stand in for the gateway, paper, the sampler, the renderer, the MCP server class and the Claude CLI.
If the gpt-oss tokenizer loads offline, a real prompt of each text is rendered and its tokens counted.

    python text_ab_test.py
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import agent
import ml_tools
import paper
import tools
import trader

ok = []
REF = "fd95e2a"
TMP = Path(tempfile.mkdtemp())


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail and not good else ""))


def load(path, name, inject=None, src=None, env=None):
    """A module from fd95e2a's source of `path` (or from `src`), with `inject` in sys.modules while it imports, so the
    old agent binds the old trader and tools; `env` is set while it imports (mcp_server reads its environment then)."""
    if src is None:
        src = subprocess.run(["git", "show", f"{REF}:{path}"], capture_output=True, text=True, encoding="utf-8", cwd=ROOT).stdout
        assert src, path
    f = TMP / f"{name}.py"
    f.write_text(src, encoding="utf-8")
    saved = {k: sys.modules.get(k) for k in (inject or {})}
    old_env = {k: os.environ.get(k) for k in (env or {})}
    sys.modules.update(inject or {})
    os.environ.update(env or {})
    try:
        spec = importlib.util.spec_from_file_location(name, f)
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
old_tools.LISTS = tools.LISTS          # its names come from lists/ (read only here: the fake gateway is the simulator's, no lookups)
old_trader = load("trader.py", "trader_ref", {"tools": old_tools})
old_agent = load("agent.py", "agent_ref", {"tools": old_tools, "trader": old_trader})
dumps = lambda x: json.dumps(x, ensure_ascii=False)          # key order counts: the renderer writes them in order


# ---------------------------------------------------------------- fakes
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


fake_mcp = {"mcp": types.ModuleType("mcp"), "mcp.server": types.ModuleType("mcp.server"),
            "mcp.server.mcpserver": types.ModuleType("mcp.server.mcpserver")}
fake_mcp["mcp.server.mcpserver"].MCPServer = FakeServer


def mcp(text, n, ref=False, **env):
    """mcp_server (fd95e2a's with ref=True) as a Claude look of this text would start it."""
    e = {"TRADER_SUBMIT_FILE": str(TMP / f"submit{n}.json"), "TRADER_LOG_FILE": str(TMP / f"log{n}.jsonl"),
         "TRADER_HAS_NEWS": "1", "TRADER_HAS_SCANNER": "1", "TRADER_HAS_STATS": "1", "TRADER_HAS_FORECAST": "0",
         "TRADER_ML_TOOL": "", "TRADER_TEXT": text, "TRADER_SLEEVE_VALUE": "", "TRADER_MAX_GROSS": "none"} | env
    inject = dict(fake_mcp, tools=old_tools if ref else tools)
    if ref:
        return load("mcp_server.py", f"mcp_ref{n}", inject, env=e)
    return load("mcp_server.py", f"mcp_{text}{n}", inject, src=(ROOT / "mcp_server.py").read_text(encoding="utf-8"), env=e)


class GW:                                                   # research tools on the simulator path (no livenews, no lookups)
    news_archive = True

    def __init__(self, px=None):
        self.px, self.asked = px or {}, []

    def quotes(self, syms):
        self.asked.append(list(syms))
        return [{"symbol": s, "error": "no data"} if s not in self.px and self.px else
                {"symbol": s, "last": self.px.get(s, 100.0), "bid": self.px.get(s, 100.0), "ask": self.px.get(s, 100.0),
                 "close": 99.0, "volume": 1000} for s in syms]

    def bars(self, sym, duration, bar_size, what, rth):
        return [{"date": "2026-11-13", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}]

    def news(self, sym, limit):
        return [{"time": "2026-11-13T14:00:00Z", "provider": "BRFG", "article_id": "BRFG$1", "headline": "h"}]

    def tape(self, seconds=5):
        return self.news("", 5)

    def article(self, provider, article_id):
        return {"text": f"body of {provider} {article_id}"}         # the ids asked for show in the result

    def scan(self, code, rows, **f):
        return [{"rank": 0, "symbol": "AAPL"}]

    def stats(self, syms):
        return [{"symbol": s, "hist_vol": 0.3, "implied_vol": 0.35, "avg_volume": 1e6, "borrow": "available",
                 "shortable_shares": 1e5} for s in syms]


class FakePaper:
    def __init__(self, pos, cash, refuse=()):
        self.pos, self.cash, self.refuse, self.placed, self.kept = pos, cash, set(refuse), [], []

    def book(self, sleeve, marks=None):
        return {"positions": {s: {"quantity": q, "avg_cost": 100.0} for s, q in self.pos.items()}, "cash": self.cash}

    def mark(self, q):
        return q["last"]

    def close_only_working(self, gw, sleeve, open_rows=None):   # paper.py's (17 Nov): these sleeves have no close_only orders
        return []

    def place(self, gw, sleeve, symbol, action, quantity, *a, probs=None, **k):
        self.placed.append((action, quantity, symbol, probs))
        if symbol in self.refuse:
            return {"result": {"status": "Inactive", "messages": ["refused by the broker"]}}
        return {"result": {"status": "Submitted", "messages": []}}

    def add_notes(self, sleeve, notes):
        self.kept += notes

    def cancel(self, gw, sleeve, ref):
        return {}

    def et_today(self):
        return "20261113"


PX = {"AMAT": 200.0, "NVDA": 180.0, "TSLA": 400.0, "TLT": 90.0}


def world(mod, meta, pos, nav=90000.0, refuse=()):
    """A PaperWorld of `mod` (trader or the fd95e2a reference) for a sleeve with this meta, worth `nav` at PX."""
    real = paper.sleeves
    paper.sleeves = lambda: {"s": meta}
    try:
        w = mod.PaperWorld(GW(PX), "s", None)
    finally:
        paper.sleeves = real
    w.paper = FakePaper(dict(pos), nav - sum(q * PX[s] for s, q in pos.items()), refuse)
    w.log = lambda m: logged.append(m)
    return w


logged = []
A_META, B_META = {"text": "A", "max_gross": None}, {"max_gross": None}
day = date(2026, 11, 16)

# ---------------------------------------------------------------- text A: byte-identical to fd95e2a
sl = json.loads((ROOT / "runs" / "week6_sleeves.json").read_text(encoding="utf-8"))
check("week 6: base-90k is the one text-A sleeve, and it has no tool",
      [k for k, m in sl.items() if trader.text_of(m) == "A"] == ["base-90k"] and not sl["base-90k"].get("ml"))
check("text_of: only \"A\" is A", [trader.text_of(m) for m in ({"text": "A"}, {}, {"text": "B"}, {"text": "a"})] == ["A", "B", "B", "B"])
check("A: the system prompt ($90,000, no tool, max_gross None) is fd95e2a's",
      trader.live_system(90000, day, max_gross=None) == old_trader.live_system(90000, day, max_gross=None))
check("A: the steps line is fd95e2a's", trader.live_steps(False, probs=True) == old_trader.live_steps(False))
check("A: the submit tool is fd95e2a's TOOL_LIVE", dumps(trader.TOOL_LIVE_PROBS) == dumps(old_trader.TOOL_LIVE)
      and dumps(trader.tool_live(probs=True)) == dumps(old_trader.TOOL_LIVE))

POS = [{"symbol": "AMAT", "quantity": 50, "avg_cost": 190.0, "close": 200.0, "unrealized": 500.0, "days": 2},
       {"symbol": "TSLA", "quantity": -20, "avg_cost": 410.0, "close": 400.0, "unrealized": 200.0, "days": 1}]
ORD = [{"ref": "o1", "symbol": "AMAT", "action": "SELL", "quantity": 50, "type": "TRAIL", "limit": None, "stop": None,
        "trail_percent": 2.5, "tif": "GTC", "cancel_pending": False},
       {"ref": "o2", "symbol": "TSLA", "action": "BUY", "quantity": 20, "type": "STP", "limit": None, "stop": 420.0,
        "trail_percent": None, "tif": "DAY", "cancel_pending": True}]
EV = ["filled BUY 50 AMAT @ 200.00 (commission $1.00, ref o0)", "BUY 5 XYZ MKT: Inactive refused"]


def state(cash, value, positions=(), orders=(), events=()):
    return {"value": value, "cash": cash, "gross": 0.5, "net": 0.1, "beta_net": 0.1, "positions": list(positions),
            "orders": list(orders), "events": list(events)}


STATES = {"cash positive": state(12000.0, 90000.0, POS, ORD, EV), "cash negative": state(-30000.0, 90000.0, POS[:1], ORD[:1]),
          "cash above the value": state(110000.0, 90000.0, POS[1:], [], EV[1:]), "no positions or orders": state(90000.0, 90000.0),
          "value not positive": state(-500.0, -200.0, POS[:1])}
FEATS = {"AMAT": {"vol": 2.5}, "TSLA": {"vol": 4.0}}
same = all(trader.render_account(st, FEATS, "H", live, mg) == old_trader.render_account(st, FEATS, "H", live, mg)
           == trader.render_account(st, FEATS, "H", live, mg, cash_clause=False)
           for st in STATES.values() for live in (None, {"AMAT": 200.0}) for mg in (None, 2.0))
check(f"A: the account text is fd95e2a's for {len(STATES)} states x live/close x max_gross None/2", same)


class W:
    MAX_GROSS = None

    def __init__(self, st):
        self.st = st

    def state(self, d, feats, live=None):
        return dict(self.st, events=list(self.st["events"]))

    def symbols_in_play(self):
        return ["AMAT", "TSLA"]


class D:
    def iso(self, d):
        return "2026-11-16"

    def table(self, d, extra=(), live=None):
        return "sym close\nAMAT 200.00", dict(FEATS)


_fp, trader.FILL_POSITION = trader.FILL_POSITION, False
check("A: account_text and state_text (the default, the fill note aside) are fd95e2a's",
      all(trader.account_text(W(st), D(), 1, FEATS)[0] == old_trader.account_text(W(st), D(), 1, FEATS)[0]
          and trader.state_text(W(st), D(), 1, header="H")[0] == old_trader.state_text(W(st), D(), 1, header="H")[0]
          for st in STATES.values()))
trader.FILL_POSITION = True
# the one change to text A here (18 Nov 2026, from the 10:30 look; the Researcher's decision, Clayton having left it to
# the Researcher): the last fill line of each symbol ends with the sleeve's position now
check("A: with the fill note on, the only change is that note on the fill line",
      all(trader.account_text(W(st), D(), 1, FEATS)[0].replace(" (you are now long 50 AMAT)", "")
          == old_trader.account_text(W(st), D(), 1, FEATS)[0] for st in STATES.values())
      and "(commission $1.00, ref o0) (you are now long 50 AMAT)" in trader.account_text(W(STATES["cash positive"]), D(), 1,
                                                                                        FEATS)[0])
trader.FILL_POSITION = _fp
memo = {"when": "Fri 13 Nov 15:30 ET", "notes": "hold"}
user = lambda mod, steps, cc=None: mod.live_user(mod.state_text(W(STATES["cash negative"]), D(), 1, header="Date: H",
                                                                **({} if cc is None else {"cash_clause": cc}))[0],
                                                 "Sleeve now $90,000.", memo, steps=steps)
check("A: the user message as live.py builds it for text A is fd95e2a's",
      user(trader, trader.live_steps(False, probs=True), cc=False) == user(old_trader, old_trader.live_steps(False)))


# the specs and system prompt handed to the renderer, from gptoss_look of each version, with fakes for the rest
class FakeRenderer:
    seen = []

    def __init__(self, tok, **k):
        pass

    def create_conversation_prefix_with_tools(self, specs, system):
        FakeRenderer.seen.append((dumps(specs), system))
        return [{"role": "system", "content": system}]

    def build_generation_prompt(self, msgs):
        FakeRenderer.seen.append(("user", msgs[-1]["content"]))
        return types.SimpleNamespace(to_ints=lambda: [1, 2, 3])


fake_tok = types.SimpleNamespace(decode=lambda t: "", encode=lambda s, add_special_tokens=False: [7])


def run_look(mod_agent, mod_trader, script, *a, **k):
    """mod_agent.gptoss_look with parse_any answering from `script` in turn (no tokenizer, no sampler)."""
    it = iter(script)
    rmod = types.ModuleType("tinker_cookbook.renderers.gpt_oss")
    rmod.GptOssRenderer = FakeRenderer
    patches = [(mod_agent, "parse_any", lambda toks, stop, names: next(it)), (mod_trader, "tokenizer", lambda: fake_tok),
               (mod_agent, "sample_once", lambda sc, renew, mi, params, trace, step:
                (types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=[5], stop_reason="stop")]), sc))]
    saved = [(m, n, getattr(m, n)) for m, n, _ in patches]
    saved_r = sys.modules.get("tinker_cookbook.renderers.gpt_oss")
    for m, n, v in patches:
        setattr(m, n, v)
    sys.modules["tinker_cookbook.renderers.gpt_oss"] = rmod
    try:
        return mod_agent.gptoss_look(None, *a, **k)
    finally:
        for m, n, v in saved:
            setattr(m, n, v)
        if saved_r is None:
            sys.modules.pop("tinker_cookbook.renderers.gpt_oss", None)
        else:
            sys.modules["tinker_cookbook.renderers.gpt_oss"] = saved_r


SUB = ("TOOL", "submit", {"place": [], "cancel": []})
sysA = trader.live_system(90000, day, max_gross=None)
FakeRenderer.seen = []
run_look(agent, trader, [SUB], GW(), sysA, "STATE", "2026-11-16", tools.available(GW()), "high", submit_tool=trader.TOOL_LIVE_PROBS)
run_look(old_agent, old_trader, [SUB], GW(), old_trader.live_system(90000, day, max_gross=None), "STATE", "2026-11-16",
         old_tools.available(GW()), "high")
new_a, old_a = FakeRenderer.seen[0], FakeRenderer.seen[2]
check("A: the full tool list and system prompt given to the renderer are fd95e2a's", new_a == old_a,
      f"{len(new_a[0])} vs {len(old_a[0])} characters")
check("A: the list has the seven research tools and submit", [s["name"] for s in json.loads(new_a[0])]
      == ["quote", "bars", "news", "tape", "article", "stats", "scanner", "submit"])
gw = GW()
res = [(n, a) for n, a in (("quote", {"symbols": ["AAPL", "TLT"]}), ("bars", {"symbol": "TLT", "bar_size": "1 day", "duration": "5 D"}),
                           ("news", {"symbol": "TLT"}), ("tape", {}), ("article", {"provider": "BRFG", "article_id": "BRFG$1"}),
                           ("stats", {"symbols": ["TLT"]}), ("scanner", {"scan": "TOP_PERC_GAIN"}))]
check("A: the research tools' results (list arguments) are fd95e2a's",
      all(tools.run(gw, n, a) == old_tools.run(gw, n, a) for n, a in res))
# The agreed exception (trader.text_of's docstring; frozen 15 Nov, item e): text A runs with the three call-repair fixes, so
# its tool results differ from fd95e2a's for exactly these malformed calls, each as the well-formed call reads at fd95e2a.
fixed = [(("quote", {"symbols": "AMAT"}), ("quote", {"symbols": ["AMAT"]})),
         (("quote", {"symbols": "AMAT, NVDA"}), ("quote", {"symbols": ["AMAT", "NVDA"]})),
         (("stats", {"symbols": "TLT"}), ("stats", {"symbols": ["TLT"]})),
         (("article", {"provider": "BRFG", "article_id": "1fc7"}), ("article", {"provider": "BRFG", "article_id": "BRFG$1fc7"})),
         (("article", {"provider": "brfg", "article_id": "BRFG$1fc7"}), ("article", {"provider": "BRFG", "article_id": "BRFG$1fc7"})),
         (("article", {"provider": "EDGAR", "article_id": "EDGAR$0001-26"}), ("article", {"provider": "EDGAR", "article_id": "0001-26"}))]
check("A with the fixes: a string \"symbols\" and a bare or lower-case article id read as the well-formed call reads at fd95e2a",
      all(tools.run(GW(), *bad) == old_tools.run(GW(), *good) for bad, good in fixed))
check("A with the fixes: those calls are the only research results that differ from fd95e2a's (these differ; the rest above match)",
      all(tools.run(GW(), *bad) != old_tools.run(GW(), *bad) for bad, _ in fixed))
CALL_TXT = {}
env_tok = types.SimpleNamespace(decode=lambda t: CALL_TXT["s"], encode=lambda s, add_special_tokens=False: [7])
saved_tok = (trader.tokenizer, old_trader.tokenizer)
trader.tokenizer = old_trader.tokenizer = lambda: env_tok
try:
    parsed = []
    for body in ('{"name": "quote", "arguments": {"symbols": ["AMAT"]}}', '{"symbols": ["AMAT"]}',
                 '{"name": "functions.submit", "arguments": {"place": [], "cancel": []}}', '{"place": [], "cancel": []}'):
        name = "submit" if "place" in body else "quote"
        CALL_TXT["s"] = f"<|channel|>commentary to=functions.{name} <|constrain|>json<|message|>{body}<|call|>"
        parsed.append((agent.parse_any([trader.CALL], "stop", {"quote", "submit"}),
                       old_agent.parse_any([trader.CALL], "stop", {"quote", "submit"})))
finally:
    trader.tokenizer, old_trader.tokenizer = saved_tok
check("A with the fixes: an enveloped call runs as the call it wraps; a plain call parses as at fd95e2a",
      parsed[0][0] == parsed[1][1] == ("TOOL", "quote", {"symbols": ["AMAT"]}) and parsed[0][1][2] != parsed[0][0][2]
      and parsed[1][0] == parsed[1][1] and parsed[2][0] == parsed[3][1] == ("TOOL", "submit", {"place": [], "cancel": []})
      and parsed[2][1][0] == "MALFORMED" and parsed[3][0] == parsed[3][1], parsed)
FakeRenderer.seen = []
run_look(agent, trader, [SUB], GW(), sysA, "STATE", "2026-11-16", tools.available(GW()), "high")
check("gptoss_look's default submit form is text B's TOOL_LIVE", json.loads(FakeRenderer.seen[0][0])[-1] == trader.TOOL_LIVE)

# the Claude path's submit tool for text A, and its reply
mA, mR = mcp("A", 1), mcp("A", 2, ref=True)
check("A: the Claude submit description is fd95e2a's", mA.server.tools["submit"][0] == mR.server.tools["submit"][0])
check("A: the other Claude tools' descriptions are fd95e2a's",
      {k: v[0] for k, v in mA.server.tools.items()} == {k: v[0] for k, v in mR.server.tools.items()})
replies = []
for g in ("none", "2"):
    os.environ["TRADER_MAX_GROSS"] = g
    for m in (mA, mR):
        m.SUBMIT.unlink(missing_ok=True)
        replies.append(m.server.tools["submit"][1]([], [], "n"))
        replies.append(m.server.tools["submit"][1]([], [], "n"))     # the second submit's reply too
os.environ.pop("TRADER_MAX_GROSS", None)
check("A: the submit replies are fd95e2a's", replies[0:2] == replies[2:4] and replies[4:6] == replies[6:8], replies)

# act() for text A: the notes (the next look's events) are fd95e2a's, with no echo
CALLS = [{"place": [{"symbol": "NVDA", "action": "BUY", "quantity": 10, "type": "MKT", "tif": "DAY"},
                    {"symbol": "AMAT", "action": "SELL", "quantity": 80, "type": "TRAIL", "trail_percent": 2.0, "tif": "GTC"},
                    {"symbol": "TSLA", "action": "BUY", "quantity": 5, "type": "LMT", "limit": 390.0, "tif": "DAY"},
                    {"symbol": "TLT", "action": "BUY", "tif": "DAY", "type": "MKT"}], "cancel": []},
         {"place": [{"symbol": "TLT", "action": "BUY", "quantity": 100, "type": "MKT", "tif": "DAY", "p_fill": 0.9, "p_win": 0.6}],
          "cancel": []}]
same, quiet = True, True
for call in CALLS:
    for pos in ({"AMAT": 50, "TSLA": -20}, {}):
        wa, wr = world(trader, A_META, pos, refuse={"TSLA"}), world(old_trader, A_META, pos, refuse={"TSLA"})
        na, nr = wa.act(json.loads(json.dumps(call))), wr.act(json.loads(json.dumps(call)))
        same &= na == nr and wa.paper.kept == wr.paper.kept and wa.paper.placed == wr.paper.placed
        quiet &= wa.TEXT == "A" and not any("at the last price" in n for n in wa.paper.kept)
check("A: act() sends the same orders and adds the same notes as fd95e2a's", same)
check("A: act() adds no echo", quiet)

# ---------------------------------------------------------------- text B: each new text, exactly and only where specified
TRAIL = "trail_percent is in percent: the stop trails by price x trail_percent / 100."
check("B: the trail sentence is the specified one", tools.TRAIL_DESC == TRAIL)
want = json.loads(dumps(old_trader.TOOL_LIVE))
# From 17 Nov (Clayton: close-only orders): the close_only field after tif, and tools.CLOSE_DESC after the trail sentence
want["description"] = want["description"].replace(" " + tools.PROBS_DESC, " " + TRAIL + " " + tools.CLOSE_DESC)
fields = want["parameters"]["properties"]["place"]["items"]["properties"]
del fields["p_fill"], fields["p_win"]
fields["trail_percent"]["description"] = TRAIL
fields["close_only"] = {"type": "boolean", "description": tools.CLOSE_DESC}
check("B: the submit form is fd95e2a's less p_fill and p_win, with the trail sentence and the close_only sentence at the end "
      "of its description, as trail_percent's and close_only's", dumps(trader.TOOL_LIVE) == dumps(want)
      and trader.TOOL_LIVE["description"].endswith("no change. " + TRAIL + " " + tools.CLOSE_DESC))
tb = dumps(trader.TOOL_LIVE)
check("B: no p_fill or p_win in the submit form; the sentence twice", "p_fill" not in tb and "p_win" not in tb and tb.count(TRAIL) == 2)
check("B: the order fields keep their order", list(trader.TOOL_LIVE["parameters"]["properties"]["place"]["items"]["properties"])
      == ["symbol", "action", "quantity", "type", "limit", "stop", "trail_percent", "tif", "close_only"])
mB = mcp("B", 3)
dB = mB.server.tools["submit"][0]
check("B: the Claude submit description", dB == (
    "Submit this look's decision: orders to place, open orders to cancel, and notes for your next look. Call it exactly once; "
    "it ends the look. Each order: {symbol, action BUY|SELL, quantity (whole shares), type MKT|MOO|LMT|STP|STP LMT|TRAIL|MOC|LOC, "
    "limit, stop, trail_percent (as needed), tif DAY|GTC, close_only true|false}. " + TRAIL + " " + tools.CLOSE_DESC
    + " Empty lists mean no change. notes: " + tools.NOTES_DESC
    + " Up to 8,000 characters are kept."), dB)
check("B: the Claude tools other than submit are text A's", {k: v[0] for k, v in mB.server.tools.items() if k != "submit"}
      == {k: v[0] for k, v in mA.server.tools.items() if k != "submit"})
for m in (mB,):
    os.environ["TRADER_MAX_GROSS"] = "none"
    m.SUBMIT.unlink(missing_ok=True)
    check("B: the Claude submit reply is text A's", m.server.tools["submit"][1]([], [], "n") == replies[0])
    os.environ.pop("TRADER_MAX_GROSS", None)
sysB = trader.live_system(90000, day, max_gross=None)
check("B: the trail sentence is nowhere else (system, steps, text A's forms)",
      all(TRAIL not in t for t in (sysB, trader.live_steps(False), trader.live_steps(True), dumps(trader.TOOL_LIVE_PROBS),
                                   mA.server.tools["submit"][0])))
check("B: the steps line asks for no probabilities", "p_fill" not in trader.live_steps(False) and "p_win" not in trader.live_steps(True))

NEG = " Negative cash is already subtracted in the sleeve value."            # the frozen wording of 15 Nov
SHO = " Cash includes the proceeds of your short sales. Your short positions are subtracted in the sleeve value."
check("B: the cash clauses are the frozen ones", trader.CASH_NEGATIVE == NEG and trader.CASH_SHORTS == SHO)
line2 = lambda st, cc: trader.render_account(st, FEATS, "H", None, None, cash_clause=cc).split("\n")[1]
check("B: cash negative", line2(STATES["cash negative"], True) == "Sleeve value at last close: $90,000 (cash $-30,000)." + NEG
      + " Gross exposure 0.50x, net +0.10x, beta-weighted +0.10x.", line2(STATES["cash negative"], True))
check("B: cash above the value", line2(STATES["cash above the value"], True) == "Sleeve value at last close: $90,000 (cash $110,000)."
      + SHO + " Gross exposure 0.50x, net +0.10x, beta-weighted +0.10x.", line2(STATES["cash above the value"], True))
check("B: cash between 0 and the value, and equal to it: no clause",
      all(trader.render_account(STATES[k], FEATS, "H", None, None, cash_clause=True)
          == trader.render_account(STATES[k], FEATS, "H", None, None) for k in ("cash positive", "no positions or orders")))
diffs = []
for k, st in STATES.items():
    a, b = trader.render_account(st, FEATS, "H", None, None).split("\n"), trader.render_account(st, FEATS, "H", None, None, cash_clause=True).split("\n")
    diffs.append(len(a) == len(b) and all(x == y for i, (x, y) in enumerate(zip(a, b)) if i != 1)
                 and (b[1] == a[1] or b[1] in (a[1].replace("). Gross", ")." + NEG + " Gross"), a[1].replace("). Gross", ")." + SHO + " Gross"))))
check("B: the clause changes only the sleeve-value line, directly after the cash figure", all(diffs))
check("B: state_text passes the clause through", NEG in user(trader, trader.live_steps(False), cc=True)
      and NEG not in user(trader, trader.live_steps(False)))

# the order echo
wb = world(trader, B_META, {"AMAT": 50, "TSLA": -20}, refuse={"TSLA"})
notes = wb.act({"place": [{"symbol": "NVDA", "action": "BUY", "quantity": 10, "type": "MKT", "tif": "DAY"},
                          {"symbol": "AMAT", "action": "SELL", "quantity": 50, "type": "TRAIL", "trail_percent": 2.5, "tif": "GTC"},
                          {"symbol": "AMAT", "action": "SELL", "quantity": 80, "type": "MKT", "tif": "DAY"},
                          {"symbol": "TSLA", "action": "BUY", "quantity": 30, "type": "LMT", "limit": 390.0, "tif": "DAY"},
                          {"symbol": "AMAT", "action": "SELL", "quantity": 60, "type": "TRAIL", "trail_percent": 1, "tif": "GTC"},
                          {"symbol": "TLT", "action": "BUY", "tif": "DAY", "type": "MKT"},
                          {"symbol": "ZZZZ", "action": "BUY", "quantity": 5, "type": "LMT", "limit": 10.0, "tif": "DAY"},
                          {"symbol": "TSLA", "action": "BUY", "quantity": 10, "type": "MKT", "tif": "DAY"}], "cancel": []})
want = ["order rejected: BUY TLT: quantity is required",                  # field errors are noted before any order is sent
        "BUY 30 TSLA LMT: Inactive refused by the broker",
        "BUY 10 TSLA MKT: Inactive refused by the broker",
        "BUY 10 NVDA MKT: about $1,800 at the last price, 0.02x your sleeve's value.",
        "SELL 50 AMAT TRAIL: about $10,000 at the last price, 0.11x your sleeve's value, trails $5.00 behind the price.",
        "SELL 80 AMAT MKT: about $16,000 at the last price, 0.18x your sleeve's value, your position is 50 shares.",
        "SELL 60 AMAT TRAIL: about $12,000 at the last price, 0.13x your sleeve's value, trails $2.00 behind the price, "
        "your position is 50 shares."]
check("B: the echoes (plain, TRAIL, closing side larger than the position, TRAIL and larger, no price), after the refusals "
      "and in the order sent", wb.paper.kept == want and notes == want and wb.TEXT == "B", "\n  " + "\n  ".join(wb.paper.kept))
check("B: an order refused by the broker gets its refusal note and no echo (review, 15 Nov); one rejected on its fields, or "
      "with no price, gets none", [n for n in want if "TSLA" in n] == want[1:3] and not any("ZZZZ" in n or "TLT" in n and "about" in n
                                                                                       for n in want))
wb = world(trader, B_META, {"TSLA": -20})
wb.act({"place": [{"symbol": "TSLA", "action": "BUY", "quantity": 30, "type": "LMT", "limit": 390.0, "tif": "DAY"}], "cancel": []})
check("B: a buy larger than the short it covers, accepted: echoed with the position",
      wb.paper.kept == ["BUY 30 TSLA LMT: about $12,000 at the last price, 0.13x your sleeve's value, your position is 20 shares."],
      wb.paper.kept)
quiet_refusals = []
for reply in ({"error": "GatewayError: Not connected to IB Gateway"}, {"result": {"status": None, "messages": ["321 bad"]}},
              {"result": {"status": "Cancelled", "messages": ["x"]}}, {"result": {"status": "ApiCancelled", "messages": []}},
              {"result": {"status": "Inactive", "messages": []}}):
    wb = world(trader, B_META, {})
    wb.paper.place = lambda *a, r=reply, **k: r
    wb.act({"place": [{"symbol": "NVDA", "action": "BUY", "quantity": 5, "type": "MKT", "tif": "DAY"}], "cancel": []})
    quiet_refusals.append(len(wb.paper.kept) == 1 and "at the last price" not in wb.paper.kept[0])
check("B: no echo for an order that never reached IBKR (paper.place's error), with no status, Cancelled, ApiCancelled or "
      "Inactive: only its refusal note", all(quiet_refusals), quiet_refusals)
wb = world(trader, {}, {"TLT": 600}, nav=30000.0)                        # 2x: the second add is rejected, so not echoed
wb.MAX_GROSS = 2.0
wb.act({"place": [{"symbol": "TLT", "action": "BUY", "quantity": 50, "type": "MKT", "tif": "DAY"},
                  {"symbol": "TLT", "action": "BUY", "quantity": 100, "type": "MKT", "tif": "DAY"}], "cancel": []})
check("B: an order stopped by a software limit gets no echo", len(wb.paper.kept) == 2 and "insufficient margin" in wb.paper.kept[0]
      and wb.paper.kept[1] == "BUY 50 TLT MKT: about $4,500 at the last price, 0.15x your sleeve's value.", wb.paper.kept)
wb = world(trader, B_META, {"AMAT": 50}, nav=-100.0)
wb.act({"place": [{"symbol": "AMAT", "action": "SELL", "quantity": 10, "type": "MKT", "tif": "DAY"}], "cancel": []})
check("B: no echo when the sleeve's value is not positive", wb.paper.placed and wb.paper.kept == [], wb.paper.kept)
check("order_echo: no price, no echo", trader.order_echo({"symbol": "X", "action": "BUY", "quantity": 1, "type": "MKT"}, None, 9e4) is None)
wb = world(trader, B_META, {})
wb.paper.place = lambda *a, **k: {"result": {"status": "Submitted"}}
real_echo = trader.order_echo
trader.order_echo = lambda *a: 1 / 0
try:
    n = wb.act({"place": [{"symbol": "NVDA", "action": "BUY", "quantity": 1, "type": "MKT", "tif": "DAY"}], "cancel": []})
finally:
    trader.order_echo = real_echo
check("B: a failing echo costs the echo only, and is logged", n == [] and logged and "order echo failed" in logged[-1], logged[-1:])

# ---------------------------------------------------------------- both texts
for meta in (A_META, B_META):
    w = world(trader, meta, {})
    w.act(CALLS[1])
    check(f"{trader.text_of(meta)}: an order with p_fill and p_win is accepted and sent, the two logged only",
          w.paper.placed == [("BUY", 100, "TLT", {"p_fill": 0.9, "p_win": 0.6})]
          and not any("rejected" in x for x in w.paper.kept), w.paper.kept)
check("symbol_list splits a string on commas and whitespace", [tools.symbol_list(x) for x in ("AMAT", "AMAT, NVDA", " a ,b  c ", ["X"], 5)]
      == [["AMAT"], ["AMAT", "NVDA"], ["a", "b", "c"], ["X"], 5])
gw = GW()
tools.run(gw, "quote", {"symbols": "AMAT, NVDA"})
tools.run(gw, "stats", {"symbols": "AMAT"})
check("quote and stats take a string of symbols", gw.asked[0] == ["AMAT", "NVDA"], gw.asked)
check("symbols_in reads a string the same way", tools.symbols_in("stats", {"symbols": "amat nvda"}) == ["AMAT", "NVDA"]
      and tools.symbols_in("forecast", {"symbols": "AMAT"}) == ["AMAT"])

# ---------------------------------------------------------------- wiring: the forecast tool gets the sleeve's value and the look key
got = []
fake_fc = types.SimpleNamespace(VERSION="FAKE", SPEC={"name": "forecast", "description": "fc", "parameters": {"type": "object", "properties": {
    "symbols": {"type": "array", "description": "s"}, "notional_usd": {"type": "number", "description": "n"},
    "sleeve_usd": {"type": "number", "description": "u"}}}},
    run=lambda gw, args, sleeve_value=None, look=None: got.append((args.get("symbols"), sleeve_value, look)) or "fc result")
real_module = ml_tools.module
ml_tools.module = lambda v=None: fake_fc
try:
    FC = ("TOOL", "forecast", {"symbols": "AMAT, NVDA"})
    call, trace, _, _ = run_look(agent, trader, [FC, ("TOOL", "forecast", {"symbols": ["XLE"]}), SUB], GW(), sysB, "STATE",
                                 "2026-11-16", tools.available(GW()) + [fake_fc.SPEC], "high", ml_tool="ML3",
                                 sleeve_value=87654.5, look=("cand1-ml", "2026-11-16T10:00:00-05:00"))
    key = ("cand1-ml", "2026-11-16T10:00:00-05:00")
    check("gpt-oss path: tools.run reaches the forecast with the sleeve's value and the look key, the same on each call",
          got == [(["AMAT", "NVDA"], 87654.5, key), (["XLE"], 87654.5, key)], got)
    got.clear()
    mF = mcp("B", 4, TRADER_HAS_FORECAST="1", TRADER_ML_TOOL="ML3", TRADER_SLEEVE_VALUE="87654.5")
    mF._gw = GW()
    mF.server.tools["forecast"][1](["AMAT"])
    mF.research("forecast", {"symbols": "AMAT NVDA"})
    check("MCP path: the forecast gets the value from TRADER_SLEEVE_VALUE and the look key \"mcp\"",
          got == [(["AMAT"], 87654.5, "mcp"), (["AMAT", "NVDA"], 87654.5, "mcp")], got)
    got.clear()
    mN = mcp("B", 5, TRADER_HAS_FORECAST="1", TRADER_ML_TOOL="ML3", TRADER_SLEEVE_VALUE="")
    mN._gw = GW()
    mN.research("forecast", {"symbols": ["AMAT"]})
    check("MCP path: no TRADER_SLEEVE_VALUE gives None (the tool's own fallback)", got == [(["AMAT"], None, "mcp")], got)
finally:
    ml_tools.module = real_module

# claude_look puts the value and the text in the env and in mcp.json's env keys
seen = {}
real_run, real_exe = agent.subprocess.run, agent.claude_exe
agent.subprocess.run = lambda cmd, **k: seen.update(env=k["env"]) or types.SimpleNamespace(returncode=0, stdout="", stderr="")
agent.claude_exe = lambda: str(TMP / "cli" / "1.0" / "x" / "claude.exe")
try:
    out = {}
    for text, v in (("A", 90123.25), ("B", None)):
        wd = TMP / f"claude-{text}"
        agent.claude_look("claude-x", "sys", "state", tools.available(GW()), wd, "max", sleeve_value=v, text=text)
        out[text] = (json.loads((wd / "mcp.json").read_text(encoding="utf-8"))["mcpServers"]["trader"]["env"], dict(seen["env"]))
finally:
    agent.subprocess.run, agent.claude_exe = real_run, real_exe
check("claude_look: TRADER_SLEEVE_VALUE and TRADER_TEXT in the CLI's env and in mcp.json's",
      out["A"][0]["TRADER_SLEEVE_VALUE"] == out["A"][1]["TRADER_SLEEVE_VALUE"] == "90123.25" and out["A"][0]["TRADER_TEXT"] == "A"
      and out["B"][0]["TRADER_SLEEVE_VALUE"] == "" and out["B"][0]["TRADER_TEXT"] == "B" and out["B"][1]["TRADER_TEXT"] == "B",
      {k: (v[0].get("TRADER_SLEEVE_VALUE"), v[0].get("TRADER_TEXT")) for k, v in out.items()})
live_src = (ROOT / "live.py").read_text(encoding="utf-8")
check("live.py passes the value, the look key, the per-text submit form, the clause and the Claude text",
      all(s in live_src for s in ('sleeve_value=st["value"], look=(name, now.isoformat())', 'trader.TOOL_LIVE_PROBS if trader.text_of(meta) == "A"',
                                  'cash_clause=trader.text_of(meta) == "B"', 'sleeve_value=st["value"],', "text=trader.text_of(meta),",   # then check_ctx (Build 2)
                                  "world.dry_run, world.log = dry_run, log")))

# ---------------------------------------------------------------- the rendered prompts, if the tokenizer loads offline
try:
    from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
    tok = trader.tokenizer()
except Exception as e:                                      # noqa: BLE001
    print(f"(tokenizer not loaded offline, no token counts: {type(e).__name__}: {str(e)[:120]})")
else:
    counts = {}
    for text in ("A", "B"):
        r = GptOssRenderer(tok, use_system_prompt=True, reasoning_effort="high", current_date="2026-11-16")
        specs = agent.harmony_specs(tools.available(GW()) + [trader.TOOL_LIVE_PROBS if text == "A" else trader.TOOL_LIVE])
        msgs = r.create_conversation_prefix_with_tools(specs, trader.live_system(90000, day, max_gross=None))
        msgs.append({"role": "user", "content": user(trader, trader.live_steps(False, probs=text == "A"), cc=text == "B")})
        ids = list(r.build_generation_prompt(msgs).to_ints())
        counts[text] = (len(ids), tok.decode(ids))
    check("rendered: text A's prompt holds p_fill and no trail sentence; text B's the reverse, and the cash clause",
          "p_fill" in counts["A"][1] and TRAIL not in counts["A"][1] and NEG not in counts["A"][1]
          and "p_fill" not in counts["B"][1] and TRAIL in counts["B"][1] and NEG in counts["B"][1])
    print(f"     tokens, the same state (cash negative, one position, one order): text A {counts['A'][0]}, text B {counts['B'][0]} "
          f"({counts['B'][0] - counts['A'][0]:+d})")
print("ALL OK" if all(ok) else "FAILED")
sys.exit(0 if all(ok) else 1)
