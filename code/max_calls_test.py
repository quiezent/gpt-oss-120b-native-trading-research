"""A sleeve's own research-call limit (15 Nov 2026), and its own forced-submit settings (5.). Clayton, relayed by the Trading Model Researcher: "Yes lets go for 15
and pass the info to Trader Training to implement.", on its recommendation of 15 on every gpt-oss sleeve except the
text-A control, which keeps 12. Offline: no IB Gateway, no Tinker, no Claude CLI; nothing under runs/ or data/ is written.

1. agent.gptoss_look(max_calls=): None and 12 are the look as at 19405df on every kind of look; 15 runs 15 calls, counts
   "of 15" and names 15 in the limit line; every mix of research calls, checks and malformed turns reaches a decision for
   any limit from 1 to tools.MAX_CALLS_TOP; with the room running out around the limit line every look decides, and no
   submit it opens has less than the floor; anything but a whole number from 1 to MAX_CALLS_TOP raises before a sample.
2. trader.max_calls_of and max_calls_bad.
3. runs/week6_sleeves.json: 15 on every gpt-oss sleeve but the text-A control (12); none on the Claude sleeves.
4. The real live.look() on fakes: each gpt-oss sleeve's look gets its own limit, and its record and research.jsonl row say
   which limit was in force; a malformed one keeps 12 and is logged.
5. live.forced_mode_of, final_line_of and switch_bad: a sleeve's "forced_mode" and "final_line" (prepared 15 Nov for the
   Researcher's second forced-submit draw, which may set them differently by sleeve); unset, each sleeve gets the module's
   own agent.FORCED_MODE and FINAL_LINE as before; the lineup sets neither until Clayton says; in the live.look of 4.
   each look gets its sleeve's settings and records them, and a malformed one is logged and keeps the default.

    python max_calls_test.py      # ALL OK and exit 0, else exit 1
"""
import contextlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import types
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import agent  # noqa: E402
import checks  # noqa: E402
import tools  # noqa: E402
import trader  # noqa: E402

ok = []
real_run, real_sample = tools.run, agent.sample_once


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail and not good else ""))


def committed(rev):
    src = subprocess.run(["git", "show", f"{rev}:agent.py"], capture_output=True, text=True, encoding="utf-8", cwd=ROOT).stdout
    f = Path(tempfile.mkdtemp()) / f"agent_{rev}.py"
    f.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"agent_{rev}", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ================================================================ 1. agent.gptoss_look
OLD = committed("19405df")
tok = trader.tokenizer()
enc = lambda s: list(tok.encode(s, add_special_tokens=False))
OPEN, A = agent.SUBMIT_OPEN(), agent.ANALYSIS_OPEN()
CTX, C3 = agent.CONTEXT, checks.MAX_CHECKS
FILLER = enc(" the")[0]
BASE_P = []


class PromptR:
    def __init__(self, *a, **k):
        pass

    def create_conversation_prefix_with_tools(self, specs, system):
        return []

    def build_generation_prompt(self, msgs):
        return types.SimpleNamespace(to_ints=lambda: list(BASE_P))


rmod_saved = sys.modules.get("tinker_cookbook.renderers.gpt_oss")
pr = types.ModuleType("tinker_cookbook.renderers.gpt_oss")
pr.GptOssRenderer = PromptR
sys.modules["tinker_cookbook.renderers.gpt_oss"] = pr

SPEC = [{"name": "quote", "description": "q", "parameters": {"type": "object", "properties": {"symbols": {"type": "array", "items": {"type": "string"}}}}}]
ORDER = {"symbol": "SPY", "action": "SELL", "quantity": 100, "type": "TRAIL", "trail_percent": 2, "tif": "GTC"}
tail = lambda name, args: enc(f"<|end|><|start|>assistant<|channel|>commentary to=functions.{name} <|constrain|>json<|message|>"
                              + json.dumps(args) + "<|call|>")
PAD = enc(" step" * 3000)[:3000]
Q = A + PAD[:150] + tail("quote", {"symbols": ["SPY"]})            # a free turn that calls a research tool
K = A + PAD[:150] + tail("check_orders", {"place": [ORDER]})       # a free turn that calls the check tool
QL = A + PAD[:1800] + tail("quote", {"symbols": ["SPY"]})          # a long one (about 1,830 tokens): it can leave less
                                                                   # than SUBMIT_ROOM, so the dropped-turn rule acts
P = enc("<|channel|>analysis<|message|>Reasoning.<|end|><|start|>assistant<|channel|>final<|message|>I would hold.<|return|>")
DECISION = enc(json.dumps({"place": [ORDER], "cancel": [], "notes": "decided"}) + "<|call|>")
REASON = enc("I sell 100 SPY on a 2% trailing stop.") + [agent.END_TOKEN()]
NOTES = agent.NOTES_OPEN()
AFTER_NOTES = enc('Sell 100 SPY.", "place": ' + json.dumps([ORDER]) + ', "cancel": []}<|call|>')


def fake(free, rec, bad_forced=0):
    """bad_forced: the first that many opened submits are answered with a malformed turn (P)."""
    its, spoilt = iter(free), [0]

    def sample_once(sc, renew, model_input, params, trace, step):
        ints = list(model_input.to_ints())
        notes = ints[-len(OPEN) - len(NOTES):] == OPEN + NOTES
        kind = "forced" if notes or ints[-len(OPEN):] == OPEN else "reason" if ints[-len(A):] == A else "free"
        if kind == "forced" and spoilt[0] < bad_forced:
            spoilt[0] += 1
            kind = "forced, malformed"
        t = (P if kind == "forced, malformed" else AFTER_NOTES if notes
             else {"forced": DECISION, "reason": REASON}.get(kind) or next(its, P))
        t = t[:params.max_tokens]
        rec.append({"prompt": ints, "max": params.max_tokens, "stop": list(params.stop), "seed": params.seed, "step": step, "kind": kind})
        stop = "length" if len(t) >= params.max_tokens and t[-1] not in params.stop else "stop"
        return types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=t, stop_reason=stop)]), sc
    return sample_once


def run(mod, n, free, with_check=True, seed=5, bad_forced=0, counted=True, result_lines=100, **kw):
    BASE_P[:] = [FILLER] * n
    rec = []
    sizes = list(result_lines) if isinstance(result_lines, (list, tuple)) else [result_lines]   # lines per call, in turn;
    tools.run = lambda gw, name, args, **k: "\n".join(["line of text"] * (sizes.pop(0) if len(sizes) > 1 else sizes[0]))   # the last repeats
    mod.sample_once = fake(list(free), rec, bad_forced)
    checker = (lambda args: ("checked\n" + "\n".join(["a line"] * 20), counted, None)) if with_check else None
    call, trace, prefill, sampled = mod.gptoss_look(None, None, "s", "s", "2026-11-16", SPEC, "high", seed=seed, checker=checker, **kw)
    return call, trace, prefill, sampled, rec


results = lambda trace: [t["result"] for t in trace if "result" in t]
counters = lambda trace: [r.rsplit("\n", 1)[-1] for r in results(trace) if r.rsplit("\n", 1)[-1].startswith("(research call ")]
N = tools.MAX_CALLS

# ---------------------------------------------------------------- None and 12 are the look as it was
same, kinds = True, 0
for free, wc in (([Q] * (N + 2), True), ([Q] * (N + 2), False), ([K] * (C3 + 3), True), ([Q] * (N + 1) + [K] * 2, True),
                 ([Q, K, P, P], True), ([P, Q, P], False), ([Q] * 5 + [K] * (C3 + 3) + [Q] * (N + 2), True)):
    for extra in ({"final_line": False, "forced_mode": "plain"}, {"final_line": True, "forced_mode": "plain"},   # given, so
                  {"final_line": False, "forced_mode": "reasoning"}, {"final_line": False, "forced_mode": "notes"}):   # the
        # comparison holds whatever the module's own FORCED_MODE and FINAL_LINE are (both may change on Monday)
        old = run(OLD, 8000, free, wc, **extra)
        for kw in ({}, {"max_calls": None}, {"max_calls": 12}):
            same = same and run(agent, 8000, free, wc, **extra, **kw) == old
        kinds += 1
check(f"max_calls None (the default) and 12 equal 19405df on {kinds} kinds of look: requests, trace, counters, decision", same)

# ---------------------------------------------------------------- 15
PIN = {"final_line": False, "forced_mode": "plain"}                # the checks below hold whatever the module defaults are
c, trace, _, _, rec = run(agent, 8000, [Q] * 17, True, max_calls=15, **PIN)
rs = results(trace)
check("15: fifteen research calls run, counted 'of 15'",
      counters(trace) == [f"(research call {k} of 15)" for k in range(1, 16)], counters(trace)[-3:])
check("15: the sixteenth call is told 'research limit reached (15 calls): call submit now'; the seventeenth opens the submit",
      rs[-2:] == ["research limit reached (15 calls): call submit now"] * 2 and rec[-1]["kind"] == "forced" and c is not None
      and trace[-1].get("tool") == "submit", rs[-2:])
c, trace, _, _, rec = run(agent, 8000, [Q] * 17, True, max_calls=15, final_line=True, forced_mode="plain")
check("15 with the final line on: the line that opens the submit names 15",
      results(trace)[-1] == "research limit reached (15 calls): no more tool calls are run in this look, check_orders included. "
      "Your next message is the submit call; it ends the look.", results(trace)[-1])
c, trace, _, _, rec = run(agent, 8000, [Q] * 14 + [P], True, max_calls=15, **PIN)
check("15: a model that stops at 14 calls is never told a limit",
      counters(trace)[-1] == "(research call 14 of 15)" and not any("research limit" in r for r in results(trace)))
c, trace, _, _, rec = run(agent, 8000, [Q] * 3 + [K] * 4 + [Q] * 14, True, max_calls=15, **PIN)
check("15: checks stay outside the research count (3 counted, the 4th told the check limit, then 12 more research calls)",
      counters(trace)[-1] == "(research call 15 of 15)" and sum(1 for t in trace if t.get("tool") == "check_orders") == 4
      and c is not None, counters(trace)[-1:])

# ---------------------------------------------------------------- every mix decides, for any limit
def mixes(n):
    """(script, exact): exact when the look must run all n research calls (at most one malformed turn, and no check past
    the counted ones); with two malformed turns, or checks past the check limit, the submit may be opened sooner."""
    plain = [[Q] * (n + 30), [K] * C3 + [Q] * (n + 30), [Q] * n + [K] * C3 + [Q] * 30, [K, Q] * C3 + [Q] * (n + 30)]
    spent = [[K] * 8 + [Q] * (n + 30), [Q] * (n - 1) + [K] * 8 + [Q] * 30, [K, Q] * (n + 20)]
    for s, exact in [(x, True) for x in plain] + [(x, False) for x in spent]:
        yield s, exact
        for i in sorted({0, max(0, n - 1), n, n + 1, n + 3}):        # one malformed turn, then two, at the places that matter
            one = s[:i] + [P] + s[i:]
            yield one, exact
            for j in sorted({i + 1, n + 2, n + 4}):
                yield one[:j] + [P] + one[j:], False


bad_looks, looks = [], 0
for n in (1, 2, 12, 15, 20, tools.MAX_CALLS_TOP):
    for s, exact in mixes(n):
        for wc in (True, False):
            sc = s if wc else [Q if t is K else t for t in s]      # no checker: each check turn is a research call instead
            c, trace, _, _, rec = run(agent, 4000, sc, wc, max_calls=n)
            looks += 1
            full = any("context nearly full" in str(t.get("note", "")) + str(t.get("result", "")) for t in trace)
            ran = sum(1 for t in trace if t.get("tool") == "quote" and str(t.get("result", "")).rsplit("\n", 1)[-1].startswith("(research call"))
            if c is None or trace[-1].get("tool") != "submit" or ran > n or full or (exact and ran != n):
                bad_looks.append((n, len(s), wc, exact, ran, full, trace[-1]))
check(f"{looks} looks, every mix of research calls, checks and up to two malformed turns, limits 1 to {tools.MAX_CALLS_TOP}: "
      "each reaches its submit and never runs more than the limit; with at most one malformed turn and no check past the "
      "check limit it runs exactly the limit", not bad_looks and looks > 1000, bad_looks[:3])

# ---------------------------------------------------------------- the step range's worst case uses its last step
# The longest look: the limit's research calls, one told the limit, (with a checker: MAX_TRIES checks run but never
# counted, one told the check limit; the told flag is shared, so the next opens the submit) the call that opens the submit,
# then two malformed opened submits and the decision: exactly the range, limit + 5 (+ MAX_TRIES + 1 with a checker).
worst = []
for n in (1, 2, 12, 15, 20, tools.MAX_CALLS_TOP):
    for wc in (False, True):
        free = [Q] * (n + 1) + ([K] * (checks.MAX_TRIES + 2) if wc else [Q])
        c, trace, _, _, rec = run(agent, 4000, free, wc, bad_forced=2, counted=False, max_calls=n, **PIN)
        steps = max(t.get("step", -1) for t in trace) + 1
        want = n + 5 + (checks.MAX_TRIES + 1 if wc else 0)
        if c is None or steps != want or [r["kind"] for r in rec[-3:]] != ["forced, malformed", "forced, malformed", "forced"]:
            worst.append((n, wc, steps, want, [r["kind"] for r in rec[-3:]], trace[-1]))
check("the longest look (two malformed opened submits after every call a limit allows) decides on the range's last step, "
      f"for limits 1 to {tools.MAX_CALLS_TOP}, with and without a checker: a range one step shorter would end it undecided",
      not worst, worst[:2])

# ---------------------------------------------------------------- the room, with 15
lines = [f"research limit reached (15 calls): call submit now",
         f"research limit reached (15 calls): " + agent.final_words(True), f"research limit reached (15 calls): " + agent.final_words(False)]
lines += [checks.cap_text(k) for k in range(C3 + 1)] + [checks.cap_text(k).rstrip(".") + "; call submit now" for k in range(C3 + 1)]
lines += [checks.cap_text(k).rstrip(".") + "; " + agent.final_words(False) for k in range(C3 + 1)]
longest = max(len(agent.tool_result_tokens(nm, l)) for l in lines for nm in ("quote", "check_orders"))
FLOOR = agent.SUBMIT_ROOM - longest - len(OPEN)
least, undecided, at_line, looks, dropped, cut = CTX, 0, 0, 0, 0, 0
# Short turns and short results (the limit line reached with room to spare), then long turns (QL, about 1,830 tokens) and
# long results near the limit line, so that a turn can leave less than SUBMIT_ROOM (dropped) and a result is cut to fit.
SWEEPS = [([Q] * 17, 100, 14500, 26500, 11), ([Q] * 15 + [K] * (C3 + 3), 100, 17000, 28000, 11),
          ([Q] * 12 + [QL] * 5, [100] * 12 + [400], 6000, 20000, 7), ([Q] * 12 + [QL] * 5, [100] * 12 + [700], 1000, 14000, 7),
          ([Q] * 13 + [QL] * 2 + [K] * (C3 + 3), [100] * 13 + [400], 6000, 20000, 7)]
for fl in (False, True):
    for free, lines_n, lo, hi, step in SWEEPS:
        for n in range(lo, hi, step * (2 if fl else 1)):
            c, trace, _, _, rec = run(agent, n, free, True, max_calls=15, final_line=fl, forced_mode="plain", result_lines=lines_n)
            looks += 1
            undecided += c is None
            dropped += any(t.get("dropped") for t in trace)
            cut += any(agent.CUT_LINE in str(t.get("result", "")) for t in trace)
            if any("research limit reached" in r or "check limit reached" in r for r in results(trace)):
                at_line += 1
                least = min([least] + [r["max"] for r in rec if r["kind"] == "forced"])
check(f"15, {looks} looks with the room running out around the limit line (final line off and on; {dropped} with a dropped "
      f"turn, {cut} with a cut result): every look decides, {at_line} reached a limit line, and no submit opened there has "
      f"less than the floor {FLOOR}; the least comes within 60 tokens of it, so the floor is reached, not just respected",
      undecided == 0 and at_line > 200 and dropped > 50 and cut > 50 and FLOOR <= least <= FLOOR + 60,
      (undecided, at_line, dropped, cut, least, FLOOR))
print(f"     (least room for an opened submit {least}; floor {FLOOR})")

# ---------------------------------------------------------------- malformed limits raise before a sample
raised = []
for v in (0, -1, tools.MAX_CALLS_TOP + 1, 15.0, "15", True, False, [15]):
    rec = []
    agent.sample_once = fake([Q], rec)
    try:
        agent.gptoss_look(None, None, "s", "s", "2026-11-16", SPEC, "high", max_calls=v)
        raised.append((v, "no raise"))
    except ValueError:
        if rec:
            raised.append((v, "sampled first"))
check("0, -1, 31, 15.0, '15', True, False and [15] raise ValueError before anything is sampled", not raised, raised)

if rmod_saved is None:
    sys.modules.pop("tinker_cookbook.renderers.gpt_oss", None)
else:
    sys.modules["tinker_cookbook.renderers.gpt_oss"] = rmod_saved
tools.run, agent.sample_once = real_run, real_sample

# ================================================================ 2. trader.max_calls_of, max_calls_bad
TABLE = [({}, 12, False), ({"max_calls": 15}, 15, False), ({"max_calls": 12}, 12, False), ({"max_calls": 1}, 1, False),
         ({"max_calls": tools.MAX_CALLS_TOP}, tools.MAX_CALLS_TOP, False), ({"max_calls": None}, 12, True),
         ({"max_calls": 0}, 12, True), ({"max_calls": 31}, 12, True), ({"max_calls": 15.0}, 12, True),
         ({"max_calls": "15"}, 12, True), ({"max_calls": True}, 12, True),
         ({"model": "base", "max_calls": 15}, 15, False), ({"model": "tinker://x", "text": "A", "max_calls": 15}, 15, False),
         ({"model": "claude-fable-5-2"}, 12, False), ({"model": "claude-fable-5-2", "max_calls": 12}, 12, False),
         ({"model": "claude-fable-5-2", "max_calls": 15}, 15, False), ({"model": "claude-fable-5-2", "max_calls": 0}, 12, True)]
wrong = [(m, trader.max_calls_of(m), trader.max_calls_bad(m)) for m, want, bad in TABLE
         if (trader.max_calls_of(m), trader.max_calls_bad(m)) != (want, bad)]
check(f"max_calls_of / max_calls_bad on {len(TABLE)} entries: 12 by default; a whole number 1 to {tools.MAX_CALLS_TOP} as "
      "given, on a gpt-oss or a Claude sleeve; anything else 12 and flagged", not wrong, wrong)

# ================================================================ 3. the Week 6 lineup
lineup = json.loads((ROOT / "runs" / "week6_sleeves.json").read_text(encoding="utf-8"))
got = {k: (trader.max_calls_of(v), v.get("max_calls")) for k, v in lineup.items()}
check("the lineup (Clayton, 15 Nov: \"Every sleeve to 15\"): 15 on all nine sleeves, the text-A control and the two Fable "
      "sleeves included", got == {k: (15, 15) for k in lineup} and len(got) == 9 and trader.text_of(lineup["base-90k"]) == "A"
      and sum(1 for v in lineup.values() if str(v["model"]).startswith("claude-")) == 2
      and not any(trader.max_calls_bad(v) for v in lineup.values()), got)

# ================================================================ 3b. the Claude path: the CLI's prompt and the MCP server
import os  # noqa: E402

cli = []
real_subrun, real_exe = agent.subprocess.run, agent.claude_exe
agent.subprocess.run = lambda cmd, **k: cli.append((cmd, k["env"])) or types.SimpleNamespace(returncode=0, stdout="", stderr="")
agent.claude_exe = lambda: str(Path(tempfile.mkdtemp()) / "x" / "y" / "claude.exe")
try:
    got_cli = {}
    for mc in (None, 12, 15):
        cli.clear()
        wd = Path(tempfile.mkdtemp())
        call, trace = agent.claude_look("claude-fable-5-2", "SYSTEM", "STATE", SPEC, wd, "xhigh", max_calls=mc)
        cmd, env = cli[0]
        cfg = json.loads((wd / "mcp.json").read_text(encoding="utf-8"))["mcpServers"]["trader"]["env"]
        got_cli[mc] = (cmd[cmd.index("--system-prompt") + 1].split("\n\n", 1)[1][:69], env["TRADER_MAX_CALLS"], cfg["TRADER_MAX_CALLS"])
    raised = []
    for v in (0, 31, "15", 15.0, True):
        cli.clear()
        try:
            agent.claude_look("claude-fable-5-2", "S", "S", SPEC, Path(tempfile.mkdtemp()), "xhigh", max_calls=v)
            raised.append((v, "no raise"))
        except ValueError:
            if cli:
                raised.append((v, "CLI started first"))
finally:
    agent.subprocess.run, agent.claude_exe = real_subrun, real_exe
check("claude_look: the system prompt says 'at most N calls' and the MCP server is given N (TRADER_MAX_CALLS): 12 by "
      "default, 15 when the sleeve has 15; a malformed limit raises before the CLI starts",
      got_cli == {None: ("Use the tools named quote for research (at most 12 calls), then call ", "12", "12"),
                  12: ("Use the tools named quote for research (at most 12 calls), then call ", "12", "12"),
                  15: ("Use the tools named quote for research (at most 15 calls), then call ", "15", "15")} and not raised,
      (got_cli, raised))

PROBE = """
import json, sys, types
sys.path.insert(0, sys.argv[1])
import mcp_server, tools
tools.run = lambda gw, name, args, **k: "result"
mcp_server.gw = lambda: object()
print(json.dumps({"limit": mcp_server.MAX_CALLS, "out": [mcp_server.research("quote", {"symbols": ["SPY"]}) for _ in range(int(sys.argv[2]))]}))
"""
mcp = {}
for val, n in ((None, 14), ("15", 17), ("x", 14), ("31", 14)):
    d = Path(tempfile.mkdtemp())
    env = dict(os.environ, TRADER_SUBMIT_FILE=str(d / "submit.json"), TRADER_LOG_FILE=str(d / "research.jsonl"))
    env.pop("TRADER_MAX_CALLS", None)
    if val is not None:
        env["TRADER_MAX_CALLS"] = val
    p = subprocess.run([sys.executable, "-c", PROBE, str(ROOT), str(n)], capture_output=True, text=True, encoding="utf-8", env=env, cwd=d)
    mcp[val] = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 and p.stdout.strip() else {"error": p.stderr[-400:]}
ok_mcp = all(
    r.get("limit") == lim and [o.rsplit("\n", 1)[-1] for o in r["out"][:lim]] == [f"(research call {k} of {lim})" for k in range(1, lim + 1)]
    and r["out"][lim:] == [f"research limit reached ({lim} calls): call submit now"] * 2
    for r, lim in ((mcp[None], 12), (mcp["15"], 15), (mcp["x"], 12), (mcp["31"], 12)))
check("the MCP server counts to TRADER_MAX_CALLS: 12 when it is absent, 15 when it is 15 (counter 'of 15', limit line '(15 "
      "calls)'), and 12 when it is malformed or above the top", ok_mcp, {k: (v.get("limit"), (v.get("out") or [v.get("error")])[-1]) for k, v in mcp.items()})

# ================================================================ 4. the real live.look() on fakes
import live  # noqa: E402
import paper  # noqa: E402

TMP = Path(tempfile.mkdtemp())
ST = {"value": 90000.0, "cash": 90000.0, "gross": 0.0, "net": 0.0, "beta_net": 0.0, "positions": [], "orders": [], "events": []}
SLEEVES = {"base-90k": {"model": "m-A", "start_cash": 90000.0, "text": "A", "max_gross": None, "max_calls": 12},
           "cand1": {"model": "m-B1", "start_cash": 90000.0, "max_gross": None, "max_calls": 15},
           "plain": {"model": "m-B2", "start_cash": 90000.0, "max_gross": None},
           "odd": {"model": "m-B3", "start_cash": 90000.0, "max_gross": None, "max_calls": "15"},
           "reasoner": {"model": "m-B4", "start_cash": 90000.0, "max_gross": None, "max_calls": 15, "forced_mode": "reasoning",
                        "final_line": True},
           "noter": {"model": "m-B5", "start_cash": 90000.0, "max_gross": None, "forced_mode": "notes", "final_line": False},
           "garbled": {"model": "m-B6", "start_cash": 90000.0, "max_gross": None, "forced_mode": "fancy", "final_line": "yes"},
           "fab": {"model": "claude-fable-5-2", "start_cash": 30000.0, "max_gross": None, "effort": "xhigh", "max_calls": 15}}
QUOTES = {"m-A": 14, "m-B1": 17, "m-B2": 14, "m-B3": 14}      # each: its limit's calls, one told, one that opens the submit
who = threading.local()
RUNNING = {}
BUY = {"symbol": "SPY", "action": "BUY", "quantity": 5, "type": "MKT", "tif": "DAY"}


class Client10:
    def __init__(self):
        self.placed = []

    def place(self, *a, **k):
        self.placed.append(a)
        return {"status": "Submitted"}

    def __getattr__(self, name):
        return lambda *a, **k: []


@contextlib.contextmanager
def fake_gateway():
    yield Client10()


class FakeData:
    dates = ["20261113"]

    def __init__(self, **k):
        pass

    def features(self, d):
        return {"SPY": {"close": 749.6}, "QQQ": {"close": 600.0}}, None

    def table_rows(self, d, extra=None):
        return ["SPY"], None

    def iso(self, d):
        return "2026-11-13"


class FakeWorld:
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
    def __init__(self, tok, **k):
        pass

    def create_conversation_prefix_with_tools(self, specs, system):
        return [{"role": "system", "content": system}]

    def build_generation_prompt(self, msgs):
        return types.SimpleNamespace(to_ints=lambda: [1, 2, 3])


fake_tok = types.SimpleNamespace(decode=lambda t: "", encode=lambda s, add_special_tokens=False: [7],
                                 vocab_size=201088, all_special_ids=[])     # read by agent._reasoning (the reasoning phase)


def fake_sample_once(sc, renew, model_input, params, trace, step):
    who.name = sc.name
    return types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=[5], stop_reason="stop")]), sc


def fake_parse(toks, stop, names):
    if toks and toks[0] == 7:                               # the submit the loop opened
        return ("TOOL", "submit", {"place": [BUY], "cancel": []})
    it = RUNNING.setdefault(who.name, iter([("TOOL", "quote", {"symbols": ["SPY"]})] * 40))
    return next(it)


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


logs, rows, seen = [], [], {}
real_gptoss = agent.gptoss_look


def spy_gptoss(*a, **k):
    seen[who_name(a)] = tuple(k.get(x, "absent") for x in ("max_calls", "forced_mode", "final_line"))
    return real_gptoss(*a, **k)


def spy_claude(*a, **k):
    seen_claude[a[0]] = k.get("max_calls", "absent")
    return None, [{"note": "a fake Claude look"}]


seen_claude = {}


def who_name(a):
    return a[0].name                                        # the sampling client FakeService made, named after the model


patches = [
    (paper, "sleeves", lambda: SLEEVES), (paper, "sync_fills", lambda gw: None),
    (paper, "book", lambda *a, **k: {"positions": {}, "cash": 90000.0}), (paper, "append", lambda f, row: rows.append((Path(f).name, row))),
    (paper, "add_notes", lambda *a, **k: None),
    (live, "gateway", fake_gateway), (live, "reconcile", lambda gw: {}), (live, "ml2_guard", lambda: None),
    (live, "account_margin", lambda gw: None), (live, "LOOKING", TMP / "LOOKING"), (live, "GPTOSS_LANES", 1),
    (live, "LOOP", TMP / "LOOP.json"), (live, "LIVE", TMP / "live"), (live, "save_memo", lambda *a, **k: None),
    (live, "open_whatif", lambda: (None, False)), (live, "read_account", lambda link, gw: None),
    (live, "sleeves_total", lambda *a, **k: 360000.0),
    (live, "log", logs.append), (live, "load_memo", lambda n: {}), (live, "sessions_left", lambda t: 5),
    (live, "datetime", FakeDT), (live.ml_tools, "unknown", lambda s: []), (live.rates, "line", lambda now: ""),
    (live.tinker, "ServiceClient", FakeService), (live.time, "sleep", lambda s: None),
    (tools, "available", lambda gw: [{"name": "quote", "description": "q", "parameters": {"type": "object", "properties": {}}}]),
    (tools, "run", lambda gw, name, args, **k: "quote result"),
    (agent, "sample_once", fake_sample_once), (agent, "parse_any", fake_parse), (agent, "gptoss_look", spy_gptoss), (agent, "claude_look", spy_claude),
    (trader, "Data", FakeData), (trader, "PaperWorld", FakeWorld), (trader, "state_text", fake_state_text),
    (trader, "tokenizer", lambda: fake_tok),
]
saved = [(m, a, getattr(m, a)) for m, a, _ in patches]
saved_mod = sys.modules.get("tinker_cookbook.renderers.gpt_oss")
rmod = types.ModuleType("tinker_cookbook.renderers.gpt_oss")
rmod.GptOssRenderer = FakeRenderer
for m, a, v in patches:
    setattr(m, a, v)
sys.modules["tinker_cookbook.renderers.gpt_oss"] = rmod
try:
    live.look(dry_run=False)
finally:
    for m, a, v in saved:
        setattr(m, a, v)
    if saved_mod is None:
        sys.modules.pop("tinker_cookbook.renderers.gpt_oss", None)
    else:
        sys.modules["tinker_cookbook.renderers.gpt_oss"] = saved_mod

recs = {p.stem.rsplit("-", 1)[0]: json.loads(p.read_text(encoding="utf-8")) for p in (TMP / "live").glob("*/*.json")}
research = {r["sleeve"]: r for f, r in rows if f == "research.jsonl"}
check("live.look gives each gpt-oss look its sleeve's limit: text A 12, cand1 15, none given 12, a malformed '15' 12",
      {m: v[0] for m, v in seen.items()} == {"m-A": 12, "m-B1": 15, "m-B2": 12, "m-B3": 12, "m-B4": 15, "m-B5": 12, "m-B6": 12},
      (seen, logs[-5:]))
want_lim = {"base-90k": 12, "cand1": 15, "plain": 12, "odd": 12, "reasoner": 15, "noter": 12, "garbled": 12, "fab": 15}
check("live.look gives the Claude sleeve's look its limit (15)", seen_claude == {"claude-fable-5-2": 15}, seen_claude)
check("each look ran its limit's research calls, was told that limit, and decided",
      set(recs) == set(SLEEVES) and all(
          r["research"]["research_calls"] == want_lim[n] + 2 and r["call"] and r["trace"][-1].get("tool") == "submit"
          and f"(research call {want_lim[n]} of {want_lim[n]})" in " ".join(str(t.get("result")) for t in r["trace"])
          and f"research limit reached ({want_lim[n]} calls): call submit now" in [t.get("result") for t in r["trace"]]
          for n, r in recs.items() if n != "fab"), {n: (r["research"]["research_calls"], r["trace"][-1]) for n, r in recs.items()})
check("each look's record and research.jsonl row say which limit was in force",
      {n: r.get("max_calls") for n, r in recs.items()} == want_lim
      and {n: r.get("max_calls") for n, r in research.items()} == want_lim, (research.keys(), {n: r.get("max_calls") for n, r in recs.items()}))
check("a malformed max_calls is logged at the look, naming the limit kept",
      any(m.startswith("odd: max_calls '15' is not a whole number from 1 to 30") and m.endswith("so the limit is 12 research calls")
          for m in logs) and not any(m.startswith(("cand1: max_calls", "plain: max_calls", "base-90k: max_calls")) for m in logs),
      [m for m in logs if "max_calls" in m])

# ================================================================ 5. the forced-submit settings, per sleeve
D_MODE, D_LINE = agent.FORCED_MODE, agent.FINAL_LINE
check("live.look gives each gpt-oss look its sleeve's forced_mode and final_line, and the module's defaults when it has none "
      "or a malformed one", {m: v[1:] for m, v in seen.items()} == {
          "m-A": (D_MODE, D_LINE), "m-B1": (D_MODE, D_LINE), "m-B2": (D_MODE, D_LINE), "m-B3": (D_MODE, D_LINE),
          "m-B4": ("reasoning", True), "m-B5": ("notes", False), "m-B6": (D_MODE, D_LINE)}, seen)
want_sw = {n: (live.forced_mode_of(m), live.final_line_of(m)) for n, m in SLEEVES.items()}
check("each look's record and research.jsonl row say which forced_mode and final_line were in force",
      {n: (r.get("forced_mode"), r.get("final_line")) for n, r in recs.items()} == want_sw
      and {n: (r.get("forced_mode"), r.get("final_line")) for n, r in research.items()} == want_sw
      and want_sw["reasoner"] == ("reasoning", True) and want_sw["noter"] == ("notes", False), want_sw)
check("the reasoner's look ran the reasoning phase and the final line; the noter's opened its submit at the notes",
      "research limit reached (15 calls): no more tool calls are run in this look. Your next message is the submit call; "
      "it ends the look." in [t.get("result") for t in recs["reasoner"]["trace"]]
      and any("reasoned" in t or "reasoning phase discarded" in str(t.get("note", "")) for t in recs["reasoner"]["trace"])
      and recs["noter"]["trace"][-1].get("opened") == "notes", (recs["reasoner"]["trace"][-2:], recs["noter"]["trace"][-1:]))
check("a malformed forced_mode or final_line is logged at the look, naming the default kept",
      f"garbled: forced_mode 'fancy' is not one of ['plain', 'reasoning', 'notes'], so the default {D_MODE!r} applies" in logs
      and f"garbled: final_line 'yes' is not true or false, so the default {D_LINE!r} applies" in logs
      and not any(m.startswith(("reasoner: f", "noter: f", "cand1: f", "plain: f")) for m in logs),
      [m for m in logs if "forced_mode" in m or "final_line" in m])
SW = [({"model": "base"}, D_MODE, D_LINE, 0), ({"model": "tinker://x", "forced_mode": "plain", "final_line": False}, "plain", False, 0),
      ({"model": "base", "forced_mode": "reasoning"}, "reasoning", D_LINE, 0), ({"model": "base", "final_line": True}, D_MODE, True, 0),
      ({"model": "base", "forced_mode": None, "final_line": None}, D_MODE, D_LINE, 2), ({"model": "base", "final_line": 1}, D_MODE, D_LINE, 1),
      ({"model": "base", "forced_mode": "Reasoning"}, D_MODE, D_LINE, 1), ({"model": "base", "forced_mode": ["notes"]}, D_MODE, D_LINE, 1),
      ({"model": "claude-fable-5-2"}, None, None, 0), ({"model": "claude-fable-5-2", "forced_mode": "notes", "final_line": True}, None, None, 2)]
wrong = [(m, live.forced_mode_of(m), live.final_line_of(m), live.switch_bad(m)) for m, fm, fl, nb in SW
         if (live.forced_mode_of(m), live.final_line_of(m), len(live.switch_bad(m))) != (fm, fl, nb)]
check(f"forced_mode_of / final_line_of / switch_bad on {len(SW)} entries: the module's defaults unless the sleeve gives a valid "
      "value; a malformed value flagged; a Claude sleeve None, flagged if it sets either", not wrong, wrong)
want_lineup = {k: (None, None) if str(v["model"]).startswith("claude-") else (D_MODE, D_LINE) if trader.text_of(v) == "A"
               else ("reasoning", True) for k, v in lineup.items()}
check("the lineup (Clayton, 15 Nov: \"1. Reasoning turn plus new final line 2. Take base setup 3. Follow Researcher text-A "
      "control recommendations\"): reasoning + final line on base's two text-B sleeves and the candidates' four; the text-A "
      "control and the Claude sleeves set neither", {k: (live.forced_mode_of(v), live.final_line_of(v)) for k, v in lineup.items()}
      == want_lineup and sum(1 for v in want_lineup.values() if v == ("reasoning", True)) == 6
      and not any(k in lineup[n] for n in ("base-90k", "fable", "fable-ml") for k in ("forced_mode", "final_line"))
      and not any(live.switch_bad(v) for v in lineup.values()), want_lineup)
check("the module's defaults are as at 19405df: plain, final line off", (D_MODE, D_LINE) == (OLD.FORCED_MODE, OLD.FINAL_LINE) == ("plain", False))

print("ALL OK" if all(ok) else "FAILED")
sys.exit(0 if all(ok) else 1)
