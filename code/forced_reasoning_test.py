"""agent.gptoss_look's reasoning before an opened submit (FORCED_MODE "reasoning", 15 Nov 2026), on scripted turns with the real
tokenizer: switch off, every request, trace and decision equal the committed 2edf75f step for step; switch on, an opened
submit is a reasoning sample (stopped on <|end|>) and then the call written after it, with the skip rules, the floor and
the context rules of agent_cap_test kept. Offline (the gpt-oss tokenizer from the local cache; no gateway, no Tinker).

    python forced_reasoning_test.py
"""
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import time
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


OLD = committed("2edf75f")
tok = trader.tokenizer()
enc = lambda s: list(tok.encode(s, add_special_tokens=False))
OPEN, A, GLUE, END = agent.SUBMIT_OPEN(), agent.ANALYSIS_OPEN(), agent.REASON_GLUE(), agent.END_TOKEN()
CTX = agent.CONTEXT
ASIDE = len(A) + len(GLUE) + len(OPEN) + agent.SUBMIT_ROOM          # the call's share when the turn reasons first
N = tools.MAX_CALLS

check("<|end|> is one token, the tokenizer's (200007), and the glue is <|end|><|start|>assistant",
      END == tok.convert_tokens_to_ids("<|end|>") == 200007 and GLUE[0] == END and tok.decode(GLUE) == "<|end|><|start|>assistant"
      and tok.decode(A) == "<|channel|>analysis<|message|>", (END, GLUE, A))
check("the switch is off in the module (FORCED_MODE is \"plain\"; forced_reasoning=True is the first build's keyword for "
      "\"reasoning\")", agent.FORCED_MODE == "plain")

# ---------------------------------------------------------------- a fake renderer and a recording sampler
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
lines = lambda n: "\n".join(["line of text"] * (n // 4))
ORDER = {"symbol": "SPY", "action": "SELL", "quantity": 100, "type": "TRAIL", "trail_percent": 0.02, "tif": "GTC"}
tail = lambda name, args: enc(f"<|end|><|start|>assistant<|channel|>commentary to=functions.{name} <|constrain|>json<|message|>"
                              + json.dumps(args) + "<|call|>")
QUOTE_TAIL, CHECK_TAIL = tail("quote", {"symbols": ["SPY"]}), tail("check_orders", {"place": [ORDER]})
SUB_TAIL = tail("submit", {"place": [ORDER], "cancel": [], "notes": "free"})
PAD_T = enc(" step" * 16000)[:16000]
PROSE = enc("<|channel|>analysis<|message|>Reasoning.<|end|><|start|>assistant<|channel|>final<|message|>I would hold.<|return|>")
DECISION = enc(json.dumps({"place": [ORDER], "cancel": [], "notes": "decided"}) + "<|call|>")
EMPTY = enc(json.dumps({"place": [], "cancel": [], "notes": "I meant to sell 100 SPY on a trailing stop"}) + "<|call|>")
BAD = enc("{bad<|call|>")
TEMPLATE = enc('{"place": [{"symbol": "...", "action": "BUY", "quantity": "..."}], "cancel": [], "notes": "..."}<|call|>')
LONG_BODY = enc('{"place": [], "notes": "') + PAD_T                  # an opened submit that runs to max_tokens
REASON = enc("I sell 100 SPY on a 2% trailing stop: the trend has turned.") + [END]
DROP_NOTE = "turn left too little room for the decision: dropped, submit opened"


def sized(t, n):
    """A free turn of exactly n tokens: reasoning, then the call `t`."""
    return A + PAD_T[:max(0, n - len(A) - len(t))] + t


class Boom(Exception):
    pass


def fake(free, forced, reason, rec):
    """Free turns from `free`; the opened submit (plain, or the call after reasoning) from `forced`; the reasoning phase from
    `reason` (an Exception item raises). Items are token lists or callables of the prompt's length; cut at max_tokens with
    stop "length" when the sample ran to it without one of its stops. rec gets every request."""
    its = {k: iter(v) for k, v in (("free", free), ("forced", forced), ("reason", reason))}
    dflt = {"free": PROSE, "forced": DECISION, "reason": REASON}

    def sample_once(sc, renew, model_input, params, trace, step):
        ints = list(model_input.to_ints())
        kind = "forced" if ints[-len(OPEN):] == OPEN else "reason" if ints[-len(A):] == A else "free"
        t = next(its[kind], dflt[kind])
        t = t(len(ints)) if callable(t) else t
        rec.append({"prompt": ints, "max": params.max_tokens, "stop": list(params.stop), "seed": params.seed, "step": step,
                    "kind": kind, "trace_len": len(trace), "tokens": None if isinstance(t, Exception) else list(t[:params.max_tokens])})
        if isinstance(t, Exception):
            raise t
        t = t[:params.max_tokens]
        stop = "length" if len(t) >= params.max_tokens and t[-1] not in params.stop else "stop"
        return types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=t, stop_reason=stop)]), sc
    return sample_once


def run(mod, n, free=(), forced=(), reason=(), checker=None, seed=None, sizes=(400,), **kw):
    BASE_P[:] = [FILLER] * n
    rec, it_sizes = [], iter(list(sizes) * 50)
    tools.run = lambda gw, name, args, **k: lines(next(it_sizes))
    mod.sample_once = fake(list(free), list(forced), list(reason), rec)
    call, trace, prefill, sampled = mod.gptoss_look(None, None, "s", "s", "2026-11-16", SPEC, "high", seed=seed, checker=checker, **kw)
    return call, trace, prefill, sampled, rec


on = lambda *a, **k: run(agent, *a, forced_reasoning=True, **k)
ends_open = lambda r: r["prompt"][-len(OPEN):] == OPEN
small_check = lambda a: ("checked\n" + lines(600), True, None)

# ---------------------------------------------------------------- switch on: a forced submit after the research limit
CALL_T = sized(QUOTE_TAIL, 200)
off_r = run(agent, 8000, [CALL_T] * (N + 2), [DECISION], seed=7)
c, trace, prefill, sampled, rec = on(8000, [CALL_T] * (N + 2), [DECISION], [REASON], seed=7)
p1, p2 = rec[-2], rec[-1]
P = off_r[4][-1]["prompt"][:-len(OPEN)]                              # the prompt at the forced step, as without the switch
sub = trace[-1]
check("research limit then another call: the reasoning request is the prompt + the analysis prefix, stop <|end|> only, "
      "max_tokens leaving SUBMIT_ROOM after the glue and the submit's prefix, seed seed*100+50+step",
      p1["kind"] == "reason" and p1["prompt"] == P + A and p1["stop"] == [END, trader.RET, trader.CALL]
      and p1["max"] == min(agent.MAX_OUT, CTX - len(P) - ASIDE) and p1["seed"] == 7 * 100 + 50 + p1["step"],
      (p1["max"], p1["stop"], p1["seed"], p1["step"]))
check("the call's request is the prompt + analysis prefix + the reasoning + <|end|><|start|>assistant + the submit's prefix, "
      "sampled as the plain opened submit (stops, seed seed*100+step, max_tokens all the room)",
      p2["prompt"] == P + A + REASON[:-1] + GLUE + OPEN and p2["stop"] == [trader.RET, trader.CALL]
      and p2["seed"] == 7 * 100 + p2["step"] == off_r[4][-1]["seed"] and p2["step"] == p1["step"]
      and p2["max"] == min(agent.MAX_OUT, CTX - len(p2["prompt"])), (p2["seed"], p2["max"]))
check("the decision is the JSON the model wrote; the trace's said holds the reasoning and the call; reasoned is there; "
      "tokens is both samples", c == {"place": [ORDER], "cancel": [], "notes": "decided"}
      and sub["said"] == tok.decode(A + REASON[:-1] + GLUE + OPEN + DECISION) and sub["reasoned"] == {"tokens": len(REASON), "cut": False}
      and sub["tokens"] == len(REASON) + len(DECISION) and set(sub) == {"step", "tool", "said", "tokens", "reasoned"}, sub.get("reasoned"))
check("prefill and sampled count both samples", prefill - off_r[2] == len(P) + len(A) + len(A) + len(REASON) - 1 + len(GLUE)
      and sampled - off_r[3] == len(REASON), (prefill - off_r[2], sampled - off_r[3]))
check("the steps before the forced one are unchanged by the switch", [r for r in rec[:-2]] == off_r[4][:-1] and trace[:-1] == off_r[1][:-1])

# ---------------------------------------------------------------- skip rule (a): too little room
c, trace, _, _, rec = on(31000, [CALL_T])
check("(a) the reserve case (step 0, room 1,768): the plain opened submit, one request, no reasoning", c and len(rec) == 1
      and ends_open(rec[0]) and rec[0]["prompt"] == [FILLER] * 31000 + OPEN and "reasoned" not in trace[-1]
      and run(agent, 31000, [CALL_T])[:4] == (c, trace, *on(31000, [CALL_T])[2:4]), len(rec))
at = CTX - ASIDE - agent.MIN_REASON                                  # the longest prompt with MIN_REASON of reasoning room
c1, t1_, _, _, r1 = on(at, [CALL_T])
c2, t2_, _, _, r2 = on(at + 1, [CALL_T])
check(f"(a) at the MIN_REASON threshold (room {CTX - at}, reserve-opened): the reasoning gets exactly {agent.MIN_REASON}; "
      f"one token less room (room {CTX - at - 1}): plain", c1 and r1[0]["kind"] == "reason" and r1[0]["max"] == agent.MIN_REASON
      and "reasoned" in t1_[-1] and c2 and len(r2) == 1 and ends_open(r2[0]) and "reasoned" not in t2_[-1],
      (CTX - at, r1[0]["max"], [r["kind"] for r in r2]))
check(f"the threshold is room {ASIDE + agent.MIN_REASON} = SUBMIT_ROOM + MIN_REASON + {len(A)}+{len(GLUE)}+{len(OPEN)} prefix tokens",
      CTX - at == agent.SUBMIT_ROOM + agent.MIN_REASON + len(A) + len(GLUE) + len(OPEN) == 2117, CTX - at)

# ---------------------------------------------------------------- skip rule (b): the last try
c_on = on(8000, [PROSE, PROSE], [DECISION], seed=3)
c_off = run(agent, 8000, [PROSE, PROSE], [DECISION], seed=3)
check("(b) two malformed turns: the opened submit is the plain one, and the look is the same as with the switch off",
      c_on == c_off and len(c_on[4]) == 3 and c_on[4][2]["kind"] == "forced" and not any(r["kind"] == "reason" for r in c_on[4]))
c, trace, _, _, rec = on(8000, [CALL_T] * (N + 2), [BAD, BAD, DECISION], [REASON, REASON, REASON])
off = run(agent, 8000, [CALL_T] * (N + 2), [BAD, BAD, DECISION])
kinds = [r["kind"] for r in rec[N + 2:]]
check("(b) a forced submit whose call is malformed twice: reason, call, reason, call, then the plain last try; two malformed "
      "counted, as with the switch off", c and c["notes"] == "decided" and kinds == ["reason", "forced", "reason", "forced", "forced"]
      and rec[-1]["prompt"] == rec[-2]["prompt"][:len(rec[-3]["prompt"]) - len(A)] + OPEN
      and sum("error" in t for t in trace) == sum("error" in t for t in off[1]) == 2
      and all("reasoned" in t for t in trace if "error" in t) and "reasoned" not in trace[-1], kinds)
c, trace, _, _, rec = on(8000, [CALL_T] * (N + 2), [BAD, BAD, BAD], [REASON, REASON])
check("(b) three malformed calls still end the look with no decision", c is None and sum("error" in t for t in trace) == 3)

# ---------------------------------------------------------------- skip rule (c): an unusable reasoning phase
off = run(agent, 8000, [CALL_T] * (N + 2), [DECISION])
for label, item, why in (("a special token", enc("I sell") + [200013] + enc(" SPY") + [END], "a special token"),
                         ("the text <|", enc("I sell <|SPY|> now") + [END], 'the text "<|"'),
                         ("a call token inside", enc("I sell") + [trader.CALL] + [END], "a special token"),
                         ("empty (ended at once)", [END], "empty"),
                         ("the sampler raised", Boom("connection reset"), "the sample failed")):
    c, trace, prefill, sampled, rec = on(8000, [CALL_T] * (N + 2), [DECISION], [item])
    note = next((t for t in trace if "reasoning phase discarded" in str(t.get("note"))), {})
    raised = isinstance(item, Exception)
    check(f"(c) reasoning with {label}: discarded with a note, the plain opened submit used for this try, the look decides",
          c == off[0] and rec[-1]["prompt"] == off[4][-1]["prompt"] and rec[-2]["kind"] == "reason" and why in note.get("note", "")
          and trace[-1] == dict(off[1][-1], tokens=off[1][-1]["tokens"] + (0 if raised else len(item)))
          and [t for t in trace if t is not note][:-1] == off[1][:-1]
          and prefill - off[2] == (0 if raised else len(off[4][-1]["prompt"]) - len(OPEN) + len(A))
          and sampled - off[3] == (0 if raised else len(item)), note.get("note"))

# ---------------------------------------------------------------- phase 1 cut at max_tokens
n = CTX - 2400                                                       # reserve-opened at step 0: reasoning room 883
c, trace, prefill, sampled, rec = on(n, [CALL_T], [DECISION], [PAD_T])
budget = 2400 - ASIDE
check("a reasoning phase cut at max_tokens: kept as written, the loop writes <|end|>, the call has exactly SUBMIT_ROOM, the look "
      "decides; reasoned says cut", c and c["notes"] == "decided" and rec[0]["max"] == budget == 883
      and rec[1]["prompt"] == [FILLER] * n + A + PAD_T[:budget] + GLUE + OPEN and CTX - len(rec[1]["prompt"]) == agent.SUBMIT_ROOM
      and trace[-1]["reasoned"] == {"tokens": budget, "cut": True} and not any("error" in t for t in trace),
      (rec[0]["max"], CTX - len(rec[1]["prompt"])))
c, trace, _, _, rec = on(n, [CALL_T], [DECISION], [PAD_T[:budget - 1] + [END]])
check("a reasoning phase ending in <|end|> at its last token is not cut; the call then has SUBMIT_ROOM + 1",
      c and trace[-1]["reasoned"] == {"tokens": budget, "cut": False} and CTX - len(rec[1]["prompt"]) == agent.SUBMIT_ROOM + 1)

# ---------------------------------------------------------------- a malformed call, then a good one
c, trace, _, _, rec = on(8000, [CALL_T] * (N + 2), [TEMPLATE, DECISION], [REASON, REASON])
kinds = [r["kind"] for r in rec[N + 2:]]
check("a malformed call after reasoning counts one malformed turn; the next try reasons again from the same prompt and decides",
      c and c["notes"] == "decided" and kinds == ["reason", "forced", "reason", "forced"] and rec[-2]["prompt"] == rec[-4]["prompt"]
      and sum("error" in t for t in trace) == 1 and trace[-2]["reasoned"] and trace[-1]["reasoned"], kinds)

# ---------------------------------------------------------------- an empty place after reasoning is the decision
c, trace, _, _, rec = on(8000, [CALL_T] * (N + 2), [EMPTY], [REASON])
off = run(agent, 8000, [CALL_T] * (N + 2), [EMPTY])
check("an empty place after reasoning is taken as the decision: no re-ask, nothing chosen (one request more than off: the "
      "reasoning)", c == json.loads(tok.decode(EMPTY[:-1])) and len(rec) == len(off[4]) + 1 and not any("error" in t for t in trace), c)

# ---------------------------------------------------------------- free turns are untouched
for label, free, chk in (("research then a free submit", [CALL_T, CALL_T, sized(SUB_TAIL, 900)], None),
                         ("checks and research", [sized(CHECK_TAIL, 300), CALL_T, sized(SUB_TAIL, 900)], small_check),
                         ("a malformed free turn, then a free submit", [PROSE, sized(SUB_TAIL, 900)], None),
                         ("the limit told, then a free submit", [CALL_T] * (N + 1) + [sized(SUB_TAIL, 900)], None)):
    a1, a2 = on(8000, free, checker=chk, seed=11), run(agent, 8000, free, checker=chk, seed=11)
    check(f"a look with no opened submit ({label}) is the same with the switch on and off", a1 == a2 and a1[0] is not None)

# ---------------------------------------------------------------- review findings (second author, 15 Nov)
# Finding 1: a call after reasoning that runs to max_tokens. Room 2,400 (reserve-opened): the reasoning runs to its cap
# (883), so the call has SUBMIT_ROOM (1,500), and a 1,600-token JSON that fits the plain call (room 2,389) is cut. The next
# try must be the plain opened submit, not another reasoning that cuts the same call again.
JSON1600 = enc('{"place": [], "cancel": [], "notes": "decided') + enc(" word") * 1580 + enc('"}') + [trader.CALL]
check("finding 1: the long JSON is between SUBMIT_ROOM and the plain call's room", 1500 < len(JSON1600) < 2389, len(JSON1600))
n = CTX - 2400
off = run(agent, n, [CALL_T], [JSON1600])
c, trace, prefill, sampled, rec = on(n, [CALL_T], [JSON1600, JSON1600], [PAD_T, PAD_T])
kinds = [r["kind"] for r in rec]
check("finding 1: the plain call fits a 1,600-token JSON (switch off decides on its first try)", off[0] and len(off[4]) == 1, len(off[4]))
check("finding 1: a call after reasoning cut at max_tokens is one malformed turn, and the next try is the plain opened submit "
      "(prompt + the submit's prefix, all the room), which decides", c and kinds == ["reason", "forced", "forced"]
      and rec[2]["prompt"] == [FILLER] * n + OPEN and rec[2]["max"] == min(agent.MAX_OUT, CTX - n - len(OPEN))
      and sum("error" in t for t in trace) == 1 and "reasoned" in next(t for t in trace if "error" in t)
      and "reasoned" not in trace[-1], kinds)
c, trace, _, _, rec = on(8000, [CALL_T] * (N + 2), [TEMPLATE, DECISION], [REASON, REASON])
check("finding 1: a malformed call after reasoning that was not cut still reasons again (the rule is only for a cut call)",
      c and [r["kind"] for r in rec[N + 2:]] == ["reason", "forced", "reason", "forced"])
# Finding 2: a phase-1 sample that times out twice: the client sample_once renewed to is kept for the plain call.
class Fut:
    def __init__(self, v):
        self.v = v

    def result(self, timeout=None):
        if isinstance(self.v, Exception):
            raise self.v
        return self.v


class Client:
    def __init__(self, name, log, timeouts):
        self.name, self.log, self.timeouts = name, log, timeouts

    def sample(self, model_input, n, params):
        ints = list(model_input.to_ints())
        self.log.append((self.name, "reason" if ints[-len(A):] == A else "call"))
        if self.timeouts:
            self.timeouts -= 1
            return Fut(TimeoutError())
        t = DECISION
        return Fut(types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=t, stop_reason="stop")]))


log, made = [], []
clients = iter([Client("new0", log, 1), Client("new1", log, 0)])
renew = lambda: made.append(next(clients)) or made[-1]
BASE_P[:] = [FILLER] * (CTX - 2400)
agent.sample_once = real_sample
c, trace, _, _ = agent.gptoss_look(Client("old", log, 5), None, "s", "s", "2026-11-16", SPEC, "high", renew=renew,
                                   forced_reasoning=True)
check("finding 2: phase 1 times out on the old client and on the renewed one; the plain call goes to the renewed client "
      "(no third client), and the look decides", c and log == [("old", "reason"), ("new0", "reason"), ("new0", "call")]
      and len(made) == 1 and "sample failed" in str(trace[-2].get("note")), (log, len(made)))
# Finding 3: phase 1 stops on <|return|> and <|call|> too (tinker.SamplingParams: an explicit stop list replaces the
# implicit EOS stop), so a stray call in the reasoning ends the sample at once; it is then discarded as before.
def stopping(reason_t, rec):
    def sample_once(sc, renew, model_input, params, trace, step):
        ints = list(model_input.to_ints())
        t = reason_t if ints[-len(A):] == A else DECISION
        t = t[:params.max_tokens]
        cut = next((i + 1 for i, x in enumerate(t) if x in params.stop), None)
        t = t[:cut] if cut else t
        rec.append({"stop": list(params.stop), "tokens": len(t), "reason": ints[-len(A):] == A})
        return types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=t, stop_reason="stop" if cut else "length")]), sc
    return sample_once


for label, stray in (("<|call|>", trader.CALL), ("<|return|>", trader.RET)):
    rec = []
    BASE_P[:] = [FILLER] * (CTX - 2400)                              # reserve-opened at step 0: reasoning room 883
    agent.sample_once = stopping(enc("I sell SPY<|start|>assistant") + OPEN + DECISION[:-1] + [stray] + PAD_T, rec)
    c, trace, _, sampled = agent.gptoss_look(None, None, "s", "s", "2026-11-16", SPEC, "high", forced_reasoning=True)
    r1 = [r for r in rec if r["reason"]]
    check(f"finding 3: phase 1 stops on <|end|>, <|return|> and <|call|>; a reasoning with a stray {label} ends there (not at "
          f"max_tokens) and is discarded; the plain call decides", c and r1 and r1[0]["stop"] == [END, trader.RET, trader.CALL]
          and r1[0]["tokens"] < 200 and "reasoning phase discarded (a special token)" in str([t.get("note") for t in trace]),
          (r1[0]["stop"] if r1 else None, r1[0]["tokens"] if r1 else None))
# Finding 4: after a discarded phase, the submit entry's "tokens" is the step's total; the note does not count them again.
LT = enc("I sell SPY <|now " + "and more " * 200) + [END]
off = run(agent, 8000, [CALL_T] * (N + 2), [DECISION])
c, trace, prefill, sampled, rec = on(8000, [CALL_T] * (N + 2), [DECISION], [LT])
note = next(t for t in trace if "reasoning phase discarded" in str(t.get("note")))
step_tokens = sum(t.get("tokens", 0) for t in trace if t.get("step") == trace[-1]["step"])
check("finding 4: a discarded phase's tokens are in the submit entry's tokens (the step's total) and not counted again in "
      "the note", trace[-1]["tokens"] == len(LT) + len(DECISION) and "tokens" not in note and note["discarded_tokens"] == len(LT)
      and step_tokens == len(LT) + len(DECISION) == sampled - off[3] + len(DECISION), (trace[-1]["tokens"], step_tokens))

# ---------------------------------------------------------------- the floor
one_liners = ["context nearly full (9999 tokens left); no research call is run", "context nearly full (9999 tokens left); no check is run",
              f"research limit reached ({tools.MAX_CALLS} calls): call submit now",
              checks.cap_text(0).rstrip(".") + "; call submit now", checks.cap_text(3).rstrip(".") + "; call submit now"]
FLOOR = agent.SUBMIT_ROOM - max(len(agent.tool_result_tokens(nm, t)) for t in one_liners for nm in ("quote", "check_orders")) - len(OPEN)
check(f"the floor (agent_cap_test's arithmetic) is {FLOOR}; a call after reasoning keeps SUBMIT_ROOM ({agent.SUBMIT_ROOM})",
      FLOOR == 1448, FLOOR)

# ---------------------------------------------------------------- the sweeps, switch on
FIX = json.loads(Path("C:/Claude/TWS-Gateway-handoff/whatif_fixture_20261115.json").read_text(encoding="utf-8"))


class FixGW:
    def account(self):
        return dict(FIX["account"])

    def positions(self):
        return [{"symbol": "SPY", "quantity": 82}]

    def all_open_orders(self):
        return []

    def whatif_many(self, orders, timeout=10.0):
        return [dict(FIX["whatifs"]["R3"]["result"]) for _ in orders]


SCTX = {"value": 90000.0, "positions": {}, "prices": {"SPY": [749.6, "delayed last"]}, "sleeves": 9, "looking": 9}
up = checks.Link(FixGW())
CHECKERS = {"working": lambda a: checks.check(up, a, SCTX), "absent": lambda a: checks.check(None, a, SCTX)}
FILL = enc(" step" * 3000)[:3000]
TURN = {"check_orders": A + FILL + CHECK_TAIL, "quote": A + FILL + QUOTE_TAIL}
opened_rooms, reasoned_rooms = [], []


def opened(rec, start):
    """(room of every opened call, room of those after a reasoning phase) in a look's requests."""
    o = [CTX - len(r["prompt"]) for r in rec if r["kind"] == "forced"]
    rs = [CTX - len(rec[i]["prompt"]) for i in range(1, len(rec)) if rec[i]["kind"] == "forced" and rec[i - 1]["kind"] == "reason"
          and rec[i]["prompt"][:len(rec[i - 1]["prompt"])] == rec[i - 1]["prompt"]]
    return o, rs


for label, kinds, chk in (("only checks, link working", ["check_orders"], CHECKERS["working"]),
                          ("only checks, link absent", ["check_orders"], CHECKERS["absent"]),
                          ("only research", ["quote"], None),
                          ("research and checks alternating", ["quote", "check_orders"], CHECKERS["working"])):
    ends = {"submit": 0, "other": 0, "reasoned": 0, "over": 0}
    for n in range(20000, 30501, 64):
        free = [TURN[kinds[i % len(kinds)]] for i in range(40)]
        c, trace, _, _, rec = on(n, free, [DECISION] * 4, [enc(" think" * 1200)[:1200] + [END]] * 4, checker=chk, sizes=(1500,))
        ends["submit" if c and c.get("notes") == "decided" else "other"] += 1
        ends["reasoned"] += any("reasoned" in t for t in trace)
        ends["over"] += any(len(r["prompt"]) + r["max"] > CTX or r["max"] < 1 for r in rec)
        o, rs = opened(rec, n)
        opened_rooms.extend(o)
        reasoned_rooms.extend(rs)
    check(f"switch on, 3,000-token turns, {label}, 20,000 to 30,500 every 64 (165 looks): every look decides, no request over "
          f"the context", ends["submit"] == 165 and not ends["over"], ends)
    print(f"     ({label}: {ends['reasoned']} looks reasoned before the opened call)")
check(f"switch on, the sweeps: every opened call had at least the floor {FLOOR} (least {min(opened_rooms)}), every call after "
      f"reasoning at least SUBMIT_ROOM (least {min(reasoned_rooms)}, {len(reasoned_rooms)} calls)",
      min(opened_rooms) >= FLOOR and reasoned_rooms and min(reasoned_rooms) >= agent.SUBMIT_ROOM)

# ---------------------------------------------------------------- random looks: off = 2edf75f; on = the context rules
rng = random.Random(1115)
lognorm = lambda med, p90: max(40, int(med * (p90 / med) ** (rng.gauss(0, 1) / 1.2816)))
TAILS = {"quote": QUOTE_TAIL, "check_orders": CHECK_TAIL, "submit": SUB_TAIL}


def random_look():
    r = rng.random()
    n = rng.randint(11000, 31500) if r < 0.7 else rng.randint(29500, 32700) if r < 0.85 else rng.randint(2000, 11000)
    free, hunt = [], rng.random() < 0.15                    # hunt: research past the limit, from a short prompt
    if hunt:
        n = rng.randint(2000, 9000)
    for _ in range(26):
        k = rng.choices(["quote", "check_orders", "submit", "runaway", "prose", "bad"],
                        weights=[10, 0.5, 0, 0.1, 0.1, 0.1] if hunt else [6, 3, 1, 0.4, 0.3, 0.2])[0]
        if k == "runaway":
            free.append(A + PAD_T[:min(lognorm(6000, 14000), 15900)])
        elif k == "prose":
            free.append(PROSE)
        elif k == "bad":
            free.append(A + enc("x") + GLUE + enc("<|channel|>commentary to=functions.quote <|constrain|>json<|message|>{bad") + [trader.CALL])
        else:
            L = lognorm(2974, 6336) if k == "submit" else lognorm(178, 2678)
            free.append(sized(TAILS[k], min(L, 15000) + len(A) + len(TAILS[k])))
    chk = small_check if rng.random() < 0.7 else None
    if chk is None:
        free = [s for s in free if s[-len(CHECK_TAIL):] != CHECK_TAIL] or [sized(SUB_TAIL, 500)]
    forced = [rng.choices([DECISION, EMPTY, BAD, TEMPLATE, LONG_BODY], weights=[6, 1, 0.6, 0.3, 0.3])[0] for _ in range(4)]
    reason = []
    for _ in range(4):
        k = rng.choices(["ok", "runaway", "empty", "special", "lt", "raise"], weights=[8, 1, 0.4, 0.4, 0.4, 0.4])[0]
        reason.append({"ok": lambda: enc(" because" * lognorm(400, 1500))[:15000] + [END], "runaway": lambda: PAD_T,
                       "empty": lambda: [END], "special": lambda: enc("a") + [200013, END], "lt": lambda: enc("a <|b") + [END],
                       "raise": lambda: Boom("sampler")}[k]())
    sizes = [rng.choice([200, 400] if hunt else [200, 800, 2000, 4000]) for _ in range(24)]
    seed = rng.choice([None, rng.randint(0, 10 ** 6)])
    return dict(n=n, free=free, forced=forced, reason=reason, checker=chk, seed=seed, sizes=sizes)


st = {"looks": 0, "off same": 0, "on decided": 0, "on context full": 0, "on three malformed": 0, "on bad": [], "free only": 0,
      "free same": 0, "reasoned": 0, "discarded": 0, "cut reasoning": 0, "plain last try": 0, "on rule breaks": []}
src = {"reserve": 0, "malformed": 0, "dropped": 0, "not run": 0, "research limit": 0, "check limit": 0, "cut turn": 0, "far": 0}
on_open, on_reasoned = [], []
t0 = time.time()
LOOKS = 2000
for i in range(LOOKS):
    L = random_look()
    kw = dict(free=L["free"], forced=L["forced"], reason=L["reason"], checker=L["checker"], seed=L["seed"], sizes=L["sizes"])
    old = run(OLD, L["n"], **kw)
    new_const = run(agent, L["n"], **kw)
    new_kw = run(agent, L["n"], forced_reasoning=False, **kw)
    st["looks"] += 1
    st["off same"] += old == new_const == new_kw
    tr, rec = old[1], old[4]
    any_open = any(r["kind"] == "forced" for r in rec)
    src["reserve"] += any(t.get("note") == "context nearly full: submit opened" for t in tr)
    src["dropped"] += any(t.get("note") == DROP_NOTE for t in tr)
    src["not run"] += any("is run" in str(t.get("result")) and "context nearly full" in str(t.get("result")) for t in tr)
    src["research limit"] += any("research limit" in str(t.get("result")) for t in tr) and any_open
    src["check limit"] += any("call submit now" in str(t.get("result")) and t.get("tool") == "check_orders" for t in tr) and any_open
    src["malformed"] += sum("error" in t and "said" in t for t in tr) >= 2
    src["cut turn"] += any(r["tokens"] is not None and len(r["tokens"]) >= r["max"] for r in rec)
    src["far"] += L["n"] < 11000
    # switch on
    c, tr, _, _, rec = run(agent, L["n"], forced_reasoning=True, **kw)
    errs = sum("error" in t and "said" in t for t in tr)
    if c is not None:
        st["on decided"] += 1
    elif any("context full" in str(t.get("error")) for t in tr) and CTX - L["n"] - len(OPEN) < agent.MIN_SUBMIT:
        st["on context full"] += 1
    elif errs == 3:
        st["on three malformed"] += 1
    else:
        st["on bad"].append((i, L["n"], [t.get("note") or t.get("error") or t.get("tool") for t in tr]))
    breaks = []
    for j, r in enumerate(rec):
        if len(r["prompt"]) + r["max"] > CTX or r["max"] < 1:
            breaks.append("over the context")
        if r["kind"] == "reason":
            before = sum("error" in t and "said" in t and t["step"] < r["step"] for t in tr)
            if before >= 2 or r["stop"] != [END, trader.RET, trader.CALL] or r["max"] < agent.MIN_REASON:
                breaks.append("a reasoning phase it should not have")
            if any(q["kind"] == "forced" and k and rec[k - 1]["kind"] == "reason" and q["prompt"][:len(rec[k - 1]["prompt"])]
                   == rec[k - 1]["prompt"] and len(q["tokens"]) >= q["max"] and q["tokens"][-1] not in (trader.RET, trader.CALL)
                   for k, q in enumerate(rec[:j])):
                breaks.append("a reasoning phase after a call after reasoning was cut (finding 1)")
            if L["seed"] is not None and r["seed"] != L["seed"] * 100 + 50 + r["step"]:
                breaks.append("phase 1 seed")
        elif L["seed"] is not None and r["seed"] != L["seed"] * 100 + r["step"]:
            breaks.append("seed")
        if r["kind"] == "forced" and j and rec[j - 1]["kind"] == "reason":
            p0 = rec[j - 1]["prompt"]
            if not (r["prompt"][:len(p0)] == p0 or r["prompt"] == p0[:-len(A)] + OPEN):
                breaks.append("the call after reasoning is not built on it")
    o, rs = opened(rec, L["n"])
    if CTX - L["n"] >= agent.RESERVE:
        on_open.extend(o)
        on_reasoned.extend(rs)
    if breaks:
        st["on rule breaks"].append((i, L["n"], sorted(set(breaks))))
    st["reasoned"] += any("reasoned" in t for t in tr)
    st["discarded"] += any("reasoning phase discarded" in str(t.get("note")) for t in tr)
    st["cut reasoning"] += any(t.get("reasoned", {}).get("cut") for t in tr)
    st["plain last try"] += any(r["kind"] == "forced" and sum("error" in t and "said" in t and t["step"] < r["step"] for t in tr) >= 2
                                for r in rec)
    if not any_open:
        st["free only"] += 1
        st["free same"] += (c, tr, rec) == (new_const[0], new_const[1], new_const[4])
secs = time.time() - t0
check(f"switch off, {LOOKS} random looks (prompts 2,000-32,700; free submits, research, checks, limits, malformed and cut "
      f"turns, every forced source): requests (prompt, max_tokens, stop, seed), trace, counts and decision equal 2edf75f "
      f"step for step, through the constant and through forced_reasoning=False", st["off same"] == LOOKS, (st["off same"], LOOKS))
check(f"the random looks reach every forced source and the far start: {src}", all(v > 0 for v in src.values()), src)
check(f"switch on, the same {LOOKS} looks: every look decides, or ends as today ('context full' only when the opened submit "
      f"at the start has under MIN_SUBMIT, or three malformed turns)", not st["on bad"] and st["on decided"] + st["on context full"] + st["on three malformed"] == LOOKS,
      st["on bad"][:2])
check("switch on: no request over the context or under 1 token; a reasoning phase only with MIN_REASON of room and before the "
      "last try; seeds as designed; the call built on its reasoning", not st["on rule breaks"], st["on rule breaks"][:3])
check(f"switch on, from a start with room >= RESERVE: every opened call had at least the floor {FLOOR} (least {min(on_open)}); "
      f"every call after reasoning at least SUBMIT_ROOM (least {min(on_reasoned)})",
      min(on_open) >= FLOOR and min(on_reasoned) >= agent.SUBMIT_ROOM)
check(f"switch on: the {st['free only']} looks with no opened submit are identical on and off", st["free only"] > 200
      and st["free same"] == st["free only"], (st["free same"], st["free only"]))
print(f"     (random looks: {LOOKS}; switch on: decided {st['on decided']}, context full {st['on context full']}, three malformed "
      f"{st['on three malformed']}; {st['reasoned']} reasoned, {st['discarded']} with a discarded phase, {st['cut reasoning']} with a "
      f"cut reasoning, {st['plain last try']} with a plain last try; {len(on_open)} opened calls; {secs:.0f} s)")

if rmod_saved is None:
    sys.modules.pop("tinker_cookbook.renderers.gpt_oss", None)
else:
    sys.modules["tinker_cookbook.renderers.gpt_oss"] = rmod_saved
tools.run, agent.sample_once = real_run, real_sample
print("ALL OK" if all(ok) else "FAILED")
sys.exit(0 if all(ok) else 1)
