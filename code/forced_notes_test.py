"""agent.gptoss_look's one setting for a submit the loop opens (FORCED_MODE, 15 Nov 2026): "plain" as through 2edf75f,
"reasoning" (forced_reasoning_test.py) and "notes", where the opened submit's prefix ends with {"notes": " so the model
writes its plan before "place". Scripted turns, the real tokenizer; offline (no gateway, no Tinker).

    python forced_notes_test.py
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import agent
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


OLD = committed("2edf75f")
tok = trader.tokenizer()
enc = lambda s: list(tok.encode(s, add_special_tokens=False))
OPEN, NOTES, A = agent.SUBMIT_OPEN(), agent.NOTES_OPEN(), agent.ANALYSIS_OPEN()
CTX, N = agent.CONTEXT, tools.MAX_CALLS
FILLER = enc(" the")[0]
BASE_P = []


class PromptR:                                              # the prompt as a list of n tokens
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
CALL_T = A + PAD[:150] + tail("quote", {"symbols": ["SPY"]})       # a free turn that calls a research tool
SUB_T = A + PAD[:150] + tail("submit", {"place": [ORDER], "cancel": [], "notes": "free"})
PROSE = enc("<|channel|>analysis<|message|>Reasoning.<|end|><|start|>assistant<|channel|>final<|message|>I would hold.<|return|>")
PLAIN_DECISION = enc(json.dumps({"place": [ORDER], "cancel": [], "notes": "decided"}) + "<|call|>")
# what a model writes after the prefix ...<|message|>{"notes": "
AFTER = lambda rest: enc(rest + "<|call|>")
GOOD = AFTER('Sell 100 SPY on a 2% trailing stop: the trend has turned.", "place": ' + json.dumps([ORDER]) + ', "cancel": []}')
HOLD = AFTER('Nothing to do: the book is as I want it.", "place": [], "cancel": []}')
RAW_NEWLINE = AFTER('Sell 100 SPY.\nThe trend has turned.", "place": ' + json.dumps([ORDER]) + ', "cancel": []}')
NOTES_ONLY = AFTER('Sell 100 SPY on a trailing stop."}')
LONG = enc('I think ') + PAD + PAD                                   # notes that run past any max_tokens (32,000 tokens)


def fake(free, forced, rec):
    """Free turns from `free`; an opened submit (plain, or begun at its notes) from `forced`. Cut at max_tokens with stop
    "length" when the sample ran to it without a stop token. rec gets every request."""
    its = {"free": iter(free), "forced": iter(forced)}

    def sample_once(sc, renew, model_input, params, trace, step):
        ints = list(model_input.to_ints())
        notes = ints[-len(OPEN) - len(NOTES):] == OPEN + NOTES
        kind = "forced" if notes or ints[-len(OPEN):] == OPEN else "free"
        t = next(its[kind], PROSE if kind == "free" else (GOOD if notes else PLAIN_DECISION))
        t = t[:params.max_tokens]
        rec.append({"prompt": ints, "max": params.max_tokens, "stop": list(params.stop), "seed": params.seed, "step": step,
                    "kind": kind, "notes": notes, "tokens": list(t)})
        stop = "length" if len(t) >= params.max_tokens and t[-1] not in params.stop else "stop"
        return types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=t, stop_reason=stop)]), sc
    return sample_once


def run(mod, n, free=(), forced=(), seed=None, **kw):
    BASE_P[:] = [FILLER] * n
    rec = []
    tools.run = lambda gw, name, args, **k: "\n".join(["line of text"] * 100)
    mod.sample_once = fake(list(free), list(forced), rec)
    call, trace, prefill, sampled = mod.gptoss_look(None, None, "s", "s", "2026-11-16", SPEC, "high", seed=seed, **kw)
    return call, trace, prefill, sampled, rec


strip = lambda rec: [{k: v for k, v in r.items() if k != "notes"} for r in rec]
LIMIT = [CALL_T] * (N + 2)                                           # 12 research calls, one told the limit, one more: the submit is opened

# ---------------------------------------------------------------- the setting
check('the module\'s mode is "plain", of ("plain", "reasoning", "notes")',
      agent.FORCED_MODE == "plain" and agent.FORCED_MODES == ("plain", "reasoning", "notes"))
check('{"notes": " follows the submit\'s prefix as text', tok.decode(OPEN + NOTES).endswith('<|message|>{"notes": "') and 0 < len(NOTES) <= 6,
      (len(NOTES), tok.decode(NOTES)))
before = len(strip([]))
try:
    BASE_P[:] = [FILLER] * 8000
    rec = []
    agent.sample_once = fake([], [], rec)
    agent.gptoss_look(None, None, "s", "s", "2026-11-16", SPEC, "high", forced_mode="note")
    check("an unknown mode raises before anything is sampled", False)
except ValueError as e:
    check("an unknown mode raises before anything is sampled", not rec and "note" in str(e), str(e))

# ---------------------------------------------------------------- plain, however it is asked for, is 2edf75f
old = run(OLD, 8000, LIMIT, [PLAIN_DECISION], seed=5)
same = True
for kw in ({}, {"forced_mode": "plain"}, {"forced_reasoning": False}, {"forced_mode": "plain", "forced_reasoning": True}):
    new = run(agent, 8000, LIMIT, [PLAIN_DECISION], seed=5, **kw)
    same = same and new[:4] == old[:4] and strip(new[4]) == strip(old[4])
check('"plain" (the default, forced_mode="plain", forced_reasoning=False, and forced_mode winning over forced_reasoning) equals '
      "2edf75f: requests, trace, counters, decision", same)
r = run(agent, 8000, LIMIT, [PLAIN_DECISION], forced_reasoning=True)
check('forced_reasoning=True alone is "reasoning" (the first build\'s keyword, kept): a request on the analysis prefix is made',
      any(q["prompt"][-len(A):] == A for q in r[4]) and not any(q["prompt"][-len(A):] == A for q in old[4]))

# ---------------------------------------------------------------- notes first, after the research limit
c, trace, prefill, sampled, rec = run(agent, 8000, LIMIT, [GOOD], seed=5, forced_mode="notes")
last, sub = rec[-1], trace[-1]
P = old[4][-1]["prompt"][:-len(OPEN)]                                # the prompt at the forced step, as plain
check("the opened submit's request is the prompt + the submit's prefix + {\"notes\": \", one sample, sampled as plain (stops, "
      "seed, max_tokens all the room left)", last["notes"] and last["prompt"] == P + OPEN + NOTES
      and last["stop"] == [trader.RET, trader.CALL] and last["seed"] == old[4][-1]["seed"]
      and last["max"] == min(agent.MAX_OUT, CTX - len(P) - len(OPEN) - len(NOTES)) and len(rec) == len(old[4]),
      (last["max"], len(rec), len(old[4])))
check("the decision is the JSON the model finished: its notes first, then its orders; the entry says how it was opened",
      c == {"notes": "Sell 100 SPY on a 2% trailing stop: the trend has turned.", "place": [ORDER], "cancel": []}
      and sub["said"] == tok.decode(OPEN + NOTES + GOOD) and sub.get("opened") == "notes" and sub["tokens"] == len(GOOD)
      and set(sub) == {"step", "tool", "said", "tokens", "opened"}, sub.get("opened"))
check("the steps before the opened submit are unchanged by the mode", strip(rec[:-1]) == strip(old[4][:-1]) and trace[:-1] == old[1][:-1])
check("prefill counts the few tokens more; sampled is the one sample", prefill - old[2] == len(NOTES)
      and sampled - old[3] == len(GOOD) - len(PLAIN_DECISION), (prefill - old[2], sampled - old[3]))

# ---------------------------------------------------------------- nothing is chosen for the model
c, trace, _, _, rec = run(agent, 8000, LIMIT, [HOLD], forced_mode="notes")
check("an empty place after its notes is the decision: taken as it comes, no second sample", c is not None and c["place"] == []
      and c["notes"].startswith("Nothing to do") and len(rec) == N + 3 and not any("error" in t for t in trace), len(rec))

# ---------------------------------------------------------------- what notes first can cost
c, trace, _, _, rec = run(agent, 8000, LIMIT, [NOTES_ONLY, GOOD], forced_mode="notes")
check("a call that holds only notes (no place list) is a malformed turn, not an empty decision; the next try begins at its "
      "notes again", c is not None and c["place"] == [ORDER] and [t.get("error", "") for t in trace if "error" in t]
      == ["MALFORMED: submit arguments malformed"] and rec[-1]["notes"] and rec[-2]["notes"], [t.get("error") for t in trace if "error" in t])
c, trace, _, _, rec = run(agent, 8000, LIMIT, [RAW_NEWLINE, GOOD], forced_mode="notes")
check("a raw line break inside the notes string is bad JSON: one malformed turn, then another try", c is not None
      and sum("error" in t for t in trace) == 1 and "bad JSON" in next(t["error"] for t in trace if "error" in t)
      and trace[-1].get("opened") == "notes", [t.get("error") for t in trace if "error" in t])
c, trace, _, _, rec = run(agent, 8000, LIMIT, [RAW_NEWLINE, NOTES_ONLY, PLAIN_DECISION], forced_mode="notes")
check("after two malformed turns the last try is the plain opened submit, exactly as without the mode", c is not None
      and c["notes"] == "decided" and [r["notes"] for r in rec[-3:]] == [True, True, False] and rec[-1]["prompt"][-len(OPEN):] == OPEN
      and "opened" not in trace[-1] and sum("error" in t for t in trace) == 2, [r["notes"] for r in rec[-3:]])
c, trace, _, _, rec = run(agent, 8000, LIMIT, [RAW_NEWLINE, NOTES_ONLY, AFTER("{bad")], forced_mode="notes")
check("three malformed turns end the look with no decision, as always", c is None and sum("error" in t for t in trace) == 3)
c, trace, _, _, rec = run(agent, 8000, LIMIT, [LONG, PLAIN_DECISION], forced_mode="notes")
check("notes that run to max_tokens: one malformed turn, and every later try in the look is opened plain",
      c is not None and c["notes"] == "decided" and [r["notes"] for r in rec[-2:]] == [True, False]
      and len(rec[-2]["tokens"]) == rec[-2]["max"] and sum("error" in t for t in trace) == 1, [r["notes"] for r in rec[-2:]])
both = [run(agent, 8000, [PROSE, PROSE], [PLAIN_DECISION], seed=3, **kw) for kw in ({}, {"forced_mode": "notes"})]
check("two malformed free turns: the submit is opened plain in notes mode too, and the look equals the plain one",
      both[0][:4] == both[1][:4] and strip(both[0][4]) == strip(both[1][4]) and not both[1][4][-1]["notes"])

# ---------------------------------------------------------------- free turns, and the room
free_looks = [run(agent, 8000, [CALL_T, CALL_T, SUB_T], seed=9, **kw) for kw in ({}, {"forced_mode": "notes"}, {"forced_mode": "reasoning"})]
check("a look that submits freely is the same in every mode", all(x[:4] == free_looks[0][:4] and strip(x[4]) == strip(free_looks[0][4])
                                                                  for x in free_looks[1:]))
least, undecided, opened = CTX, 0, 0
big = A + PAD[:3000 - len(A) - len(tail("quote", {"symbols": ["SPY"]}))] + tail("quote", {"symbols": ["SPY"]})   # 3,000-token research turns
for n in range(20000, 30501, 64):
    c, trace, _, _, rec = run(agent, n, [big] * 30, [], forced_mode="notes")      # the fake answers each opening in its own form
    undecided += c is None
    for r in rec:
        if r["kind"] == "forced":
            opened += r["notes"]
            least = min(least, r["max"])
FLOOR = 1448 - len(NOTES)
check(f"a sweep of 165 starting prompts with 3,000-token research turns, notes mode: every look decides, and the opened submit "
      f"has at least {FLOOR} tokens (plain's 1,448 less the {len(NOTES)} of the notes prefix)",
      undecided == 0 and least >= FLOOR and opened > 100, (undecided, least, opened))
print(f"     (sweep: {opened} submits begun at their notes; least room {least})")
c, trace, _, _, rec = run(agent, 31000, [CALL_T], [GOOD], forced_mode="notes")
check("a submit the context reserve opens begins at its notes too (one sample, the same room less the prefix)",
      c is not None and len(rec) == 1 and rec[0]["notes"] and rec[0]["max"] == CTX - 31000 - len(OPEN) - len(NOTES), rec[0]["max"])

if rmod_saved is None:
    sys.modules.pop("tinker_cookbook.renderers.gpt_oss", None)
else:
    sys.modules["tinker_cookbook.renderers.gpt_oss"] = rmod_saved
tools.run, agent.sample_once = real_run, real_sample
print("ALL OK" if all(ok) else "FAILED")
sys.exit(0 if all(ok) else 1)
