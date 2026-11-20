"""agent.gptoss_look's FINAL_LINE (15 Nov 2026): the occurrence of a limit line that opens the submit states that no more
tool calls are run in this look and that the next message is the submit call. Off by default; off equals cda1905.
Scripted turns, the real tokenizer; offline (no gateway, no Tinker).

    python final_line_test.py
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import agent
import checks
import tools
import trader

ok = []
real_run, real_sample = tools.run, agent.sample_once


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail else ""))


def committed(rev):
    src = subprocess.run(["git", "show", f"{rev}:agent.py"], capture_output=True, text=True, encoding="utf-8",
                         cwd=Path(__file__).resolve().parent).stdout
    f = Path(tempfile.mkdtemp()) / f"agent_{rev}.py"
    f.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"agent_{rev}", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OLD = committed("cda1905")
tok = trader.tokenizer()
enc = lambda s: list(tok.encode(s, add_special_tokens=False))
OPEN, A = agent.SUBMIT_OPEN(), agent.ANALYSIS_OPEN()
CTX, N, C3 = agent.CONTEXT, tools.MAX_CALLS, checks.MAX_CHECKS
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
PAD = enc(" step" * 16000)[:16000]
Q = A + PAD[:150] + tail("quote", {"symbols": ["SPY"]})            # a free turn that calls a research tool
K = A + PAD[:150] + tail("check_orders", {"place": [ORDER]})       # a free turn that calls the check tool
PROSE = enc("<|channel|>analysis<|message|>Reasoning.<|end|><|start|>assistant<|channel|>final<|message|>I would hold.<|return|>")
DECISION = enc(json.dumps({"place": [ORDER], "cancel": [], "notes": "decided"}) + "<|call|>")
REASON = enc("I sell 100 SPY on a 2% trailing stop.") + [agent.END_TOKEN()]
NOTES = agent.NOTES_OPEN()
AFTER_NOTES = enc('Sell 100 SPY.", "place": ' + json.dumps([ORDER]) + ', "cancel": []}<|call|>')


def fake(free, rec):
    its = iter(free)

    def sample_once(sc, renew, model_input, params, trace, step):
        ints = list(model_input.to_ints())
        notes = ints[-len(OPEN) - len(NOTES):] == OPEN + NOTES
        kind = "forced" if notes or ints[-len(OPEN):] == OPEN else "reason" if ints[-len(A):] == A else "free"
        t = AFTER_NOTES if notes else {"forced": DECISION, "reason": REASON}.get(kind) or next(its, PROSE)
        t = t[:params.max_tokens]
        rec.append({"prompt": ints, "max": params.max_tokens, "stop": list(params.stop), "seed": params.seed, "step": step, "kind": kind})
        stop = "length" if len(t) >= params.max_tokens and t[-1] not in params.stop else "stop"
        return types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=t, stop_reason=stop)]), sc
    return sample_once


def run(mod, n, free, with_check=True, seed=5, **kw):
    BASE_P[:] = [FILLER] * n
    rec = []
    tools.run = lambda gw, name, args, **k: "\n".join(["line of text"] * 100)
    mod.sample_once = fake(list(free), rec)
    checker = (lambda args: ("checked\n" + "\n".join(["a line"] * 20), True, None)) if with_check else None
    call, trace, prefill, sampled = mod.gptoss_look(None, None, "s", "s", "2026-11-16", SPEC, "high", seed=seed, checker=checker, **kw)
    return call, trace, prefill, sampled, rec


results = lambda trace: [t["result"] for t in trace if "result" in t]
RL = [Q] * (N + 2)                                                   # 12 research calls, one told the limit, one more: the submit is opened
FIRST = f"research limit reached ({N} calls): call submit now"
WITH = f"research limit reached ({N} calls): no more tool calls are run in this look, check_orders included. Your next message is the submit call; it ends the look."
WITHOUT = f"research limit reached ({N} calls): no more tool calls are run in this look. Your next message is the submit call; it ends the look."
CAP = "check limit reached (3 of 3); no further check is run this look"
CAP_FINAL = CAP + "; no more tool calls are run in this look. Your next message is the submit call; it ends the look."

# ---------------------------------------------------------------- the switch, and off = cda1905
check("the switch is off in the module", agent.FINAL_LINE is False)
same = True
for free, wc in ((RL, True), (RL, False), ([K] * (C3 + 3), True), ([Q] * (N + 1) + [K] * 2, True), ([Q, K, PROSE, PROSE], True)):
    old = run(OLD, 8000, free, wc)
    for kw in ({}, {"final_line": False}):
        same = same and run(agent, 8000, free, wc, **kw) == old
check("off (the default and final_line=False) equals cda1905 on five kinds of look: requests, trace, counters, decision", same)

# ---------------------------------------------------------------- the research limit
c, trace, _, _, rec = run(agent, 8000, RL, True, final_line=True)
off = run(agent, 8000, RL, True)
check("research limit, a look with check_orders: the first occurrence is unchanged; the one that opens the submit states the facts",
      results(trace)[-2:] == [FIRST + "", WITH] and results(off[1])[-2:] == [FIRST, FIRST] and c == off[0], results(trace)[-2:])
check("the model reads it: the opened submit's prompt ends with that line, then the submit's prefix",
      rec[-1]["kind"] == "forced" and rec[-1]["prompt"][:-len(OPEN)][-len(agent.tool_result_tokens("quote", WITH)):]
      == agent.tool_result_tokens("quote", WITH))
check("every step before the last line is the same with the switch on", trace[:-2] == off[1][:-2] and rec[:-1] == off[4][:-1])
c, trace, _, _, rec = run(agent, 8000, RL, False, final_line=True)
check("a look without check_orders (text A's control, or no what-if link) is not told of a tool it does not have",
      results(trace)[-1] == WITHOUT and "check_orders" not in results(trace)[-1], results(trace)[-1])

# ---------------------------------------------------------------- the check limit
free = [K] * (C3 + 3)        # three counted checks, one answered with the limit, one told to submit, one more: the submit is opened
c, trace, _, _, rec = run(agent, 8000, free, True, final_line=True)
off = run(agent, 8000, free, True)
check("check limit: the limit text, then '; call submit now', then the line that opens the submit states the facts (no "
      "'check_orders included': the limit text has just said no further check is run)",
      results(trace)[-3:] == [CAP + ".", CAP + "; call submit now", CAP_FINAL]
      and results(off[1])[-3:] == [CAP + ".", CAP + "; call submit now", CAP + "; call submit now"] and c == off[0], results(trace)[-3:])
free = [Q] * (N + 1) + [K] * 2      # told at the research limit, then a second check past the check limit... with no check counted
c, trace, _, _, rec = run(agent, 8000, [Q] * (N + 1) + [K] * (C3 + 2), True, final_line=True)
check("told at the research limit, then checks past the check limit: the second one past it opens the submit with the facts",
      results(trace)[-1].endswith("no more tool calls are run in this look. Your next message is the submit call; it ends the look.")
      and results(trace)[-1].startswith("check limit reached") and rec[-1]["kind"] == "forced", results(trace)[-1])

# ---------------------------------------------------------------- what it leaves alone, and the modes
for label, n, free in (("the context reserve", 31000, [Q]), ("two malformed turns", 8000, [PROSE, PROSE]),
                       ("a free submit", 8000, [Q, A + PAD[:50] + tail("submit", {"place": [ORDER], "cancel": [], "notes": "free"})])):
    check(f"{label}: the same with the switch on or off", run(agent, n, free, True, final_line=True) == run(agent, n, free, True))
c, trace, _, _, rec = run(agent, 8000, RL, True, final_line=True, forced_mode="reasoning")
check("with forced_mode \"reasoning\": the line is shown and the reasoning phase still runs before the call",
      WITH in results(trace) and [r["kind"] for r in rec[-2:]] == ["reason", "forced"] and "reasoned" in trace[-1])
c, trace, _, _, rec = run(agent, 8000, RL, True, final_line=True, forced_mode="notes")
check("with forced_mode \"notes\": the line is shown before the opened submit, which begins at its notes",
      WITH in results(trace) and rec[-1]["prompt"][-len(NOTES):] == NOTES and c is not None and trace[-1].get("opened") == "notes")

# ---------------------------------------------------------------- the room
longest = max(len(agent.tool_result_tokens("check_orders", checks.cap_text(k).rstrip(".") + "; " + agent.final_words(False))) for k in range(C3 + 1))
FLOOR = agent.SUBMIT_ROOM - longest - len(OPEN)
least, undecided, shown, looks = CTX, 0, 0, 0
for free, lo, hi in (([Q] * (N + 2), 18000, 27000), ([K] * (C3 + 3), 24000, 31000)):   # short turns, so the limit line is reached
    for n in range(lo, hi, 7):                                                           # with the room running out around it
        c, trace, _, _, rec = run(agent, n, free, True, final_line=True)
        looks += 1
        undecided += c is None
        said = any("it ends the look" in r for r in results(trace))
        shown += said
        if said:
            least = min([least] + [r["max"] for r in rec if r["kind"] == "forced"])
check(f"the longest line is now {longest} tokens (41 before), so the opened submit's floor with the switch on is {FLOOR} "
      f"(1,448 with it off); {looks} looks with the room running out around the limit line: every look decides, and no "
      f"submit opened by that line has less", FLOOR == 1428 and undecided == 0 and shown > 200 and least >= FLOOR,
      (FLOOR, undecided, shown, least))
print(f"     (sweeps: {shown} of {looks} looks were shown the line; least room for a submit it opened {least})")

# ---------------------------------------------------------------- the context line (Clayton, 17 Nov: "Yes, deploy it")
# A research call refused for context: room 3,268 for the first free turn (above RESERVE), which leaves 3,092 after it, so
# its result (estimated 1,500) would leave under RESERVE: the call is not run, the line says so, and the submit is opened.
CTX_LINE = []
for fl, wc in ((True, True), (True, False), (False, True)):
    c, trace, _, _, rec = run(agent, CTX - 3268, [Q] * 3, wc, final_line=fl)
    CTX_LINE.append(([r for r in results(trace) if r.startswith("context nearly full")], c, rec[-1]["kind"]))
on_c, on_n, off = CTX_LINE
check("the context line with the final line on ends with its words (check_orders named only when the look has it); with it "
      "off the line is as before; each look's next turn is the opened submit and it decides",
      len(on_c[0]) == len(on_n[0]) == len(off[0]) == 1
      and on_c[0][0].endswith("; no research call is run; " + agent.final_words(True))
      and on_n[0][0].endswith("; no research call is run; " + agent.final_words(False))
      and off[0][0].endswith("tokens left); no research call is run")
      and all(x[1] is not None and x[2] == "forced" for x in CTX_LINE), CTX_LINE)

if rmod_saved is None:
    sys.modules.pop("tinker_cookbook.renderers.gpt_oss", None)
else:
    sys.modules["tinker_cookbook.renderers.gpt_oss"] = rmod_saved
tools.run, agent.sample_once = real_run, real_sample
print("ALL OK" if all(ok) else "FAILED")
sys.exit(0 if all(ok) else 1)
