"""agent.gptoss_look at the research limit, on scripted turns: the turn after the limit is free (it reasons and submits),
another call after that opens the submit, and the longest path fits the step limit. Offline (the gpt-oss tokenizer from
the local cache; no gateway, no Tinker).

    python agent_cap_test.py
"""
import sys
import types

import agent
import tools
import trader

SPEC = [{"name": "quote", "description": "q", "parameters": {"type": "object", "properties": {"symbols": {"type": "array", "items": {"type": "string"}}}}}]
THINK = "<|channel|>analysis<|message|>Reasoning.<|end|><|start|>assistant"
CALL = THINK + '<|channel|>commentary to=functions.quote <|constrain|>json<|message|>{"symbols": ["A"]}<|call|>'
SUBMIT = THINK + '<|channel|>commentary to=functions.submit <|constrain|>json<|message|>{"place": [{"symbol": "A"}], "cancel": [], "notes": "free"}<|call|>'
BODY = '{"place": [], "cancel": [], "notes": "forced"}<|call|>'
BAD = "{bad<|call|>"
real_run, real_sample = tools.run, agent.sample_once
tools.run = lambda gw, name, args, ml_tool=None: f"{name} result"
ok = []


def run(turns):
    it = iter(turns)

    def sample_once(sc, renew, model_input, params, trace, step):
        toks = list(trader.tokenizer().encode(next(it), add_special_tokens=False))
        return types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=toks, stop_reason="stop")]), sc
    agent.sample_once = sample_once
    call, trace, _, _ = agent.gptoss_look(None, types.SimpleNamespace(), "system", "state", "2026-11-02", SPEC, "high")
    return call, trace


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail else ""))


N = tools.MAX_CALLS
call, trace = run([CALL] * (N + 1) + [SUBMIT])
said = next((t["said"] for t in trace if t.get("tool") == "submit"), "")
check("the turn after the limit is free: its reasoning is kept", call and call["notes"] == "free" and "<|channel|>analysis" in said, said[:50])
check(f"{N} calls ran and the next was told the limit",
      sum(t.get("result") == "quote result" or str(t.get("result", "")).startswith("quote result") for t in trace) == N
      and sum("limit reached" in str(t.get("result")) for t in trace) == 1)

call, trace = run([CALL] * (N + 2) + [BODY])
said = next((t["said"] for t in trace if t.get("tool") == "submit"), "")
check("another call after the limit: the submit is opened", call and call["notes"] == "forced" and said.startswith("<|channel|>commentary to=functions.submit"), said[:45])

call, trace = run([CALL] * (N + 2) + [BAD, BAD, BAD])
check("three strikes still end the look with no decision", call is None, str(trace[-1].get("error"))[:40])

call, trace = run([CALL] * (N + 2) + [BAD, BAD, BODY])
check("the longest path fits the step limit", call and call["notes"] == "forced")

call, trace = run([CALL] * 3 + [SUBMIT])
check("a look under the limit is unchanged", call and call["notes"] == "free" and not any("limit" in str(t.get("result")) for t in trace))

call, trace = run([BAD, BAD, BODY])
check("malformed turns still open the submit", call and call["notes"] == "forced")

# A closing turn that holds the decision is the decision; prose with none is re-asked with a reasoning turn
SUBMIT_RET = SUBMIT.removesuffix("<|call|>") + "<|return|>"
ANSWER = THINK + '<|channel|>final<|message|>**Submit**\n```json\n{"place": [{"symbol": "A"}], "cancel": [], "notes": "answer"}\n```<|return|>'
DRAFT = THINK + '<|channel|>final<|message|>Draft: {"place": [{"symbol": "A"}], "cancel": [], "notes": "draft"}\nToo risky, I will hold.<|return|>'
WRONG_KEY = THINK + '<|channel|>commentary to=functions.submit json<|message|>{"new_orders": [{"symbol": "A"}], "cancel": [], "notes": "lost"}<|call|>'
PROSE = THINK + "<|channel|>final<|message|>I would hold.<|return|>"

call, trace = run([SUBMIT_RET])
check("a whole submit ending in <|return|> is the decision, with no second sample", call and call["notes"] == "free" and len(trace) == 1)

call, trace = run([ANSWER])
check("an answer holding the submit's JSON is the decision", call and call["notes"] == "answer" and len(trace) == 1)

call, trace = run([DRAFT, SUBMIT])
check("a draft with words after it is re-asked, not sent", call and call["notes"] == "free" and "FINAL" in str(trace[0].get("error")))

call, trace = run([WRONG_KEY, SUBMIT])
check("a submit with its orders under another key is re-asked, not an empty hold", call and call["notes"] == "free" and "malformed" in str(trace[0].get("error")))

call, trace = run([WRONG_KEY, WRONG_KEY, WRONG_KEY])
check("three of those still end the look", call is None)

TEMPLATE = '{"place": [{"symbol": "...", "action": "BUY", "quantity": "..."}], "cancel": ["tt-x", "..."], "notes": "..."}<|call|>'
call, trace = run([THINK + "<|channel|>commentary to=functions.submit json<|message|>" + TEMPLATE, SUBMIT])
check("a template is re-asked, not acted on", call and call["notes"] == "free")

call, trace = run([CALL] * (N + 2) + [TEMPLATE, BODY])
check("a template on a forced turn is another forced try", call and call["notes"] == "forced")

call, trace = run([PROSE, PROSE, TEMPLATE])
check("a template on the last forced try ends the look with nothing cancelled", call is None)

call, trace = run([PROSE, SUBMIT])
said = next((t["said"] for t in trace if t.get("tool") == "submit"), "")
check("prose with no decision: the next turn reasons and submits", call and call["notes"] == "free" and said.startswith("<|channel|>analysis"), said[:30])

call, trace = run([PROSE, PROSE, BODY])
check("a second such turn opens the submit", call and call["notes"] == "forced")

call, trace = run([PROSE, PROSE, PROSE])
check("three strikes end the look, prose too", call is None)

call, trace = run([CALL] * (N + 1) + [PROSE, SUBMIT])
check("prose after the limit's free turn is re-asked too", call and call["notes"] == "free")

seen = []
real_tokens = list(trader.tokenizer().encode(CALL.removesuffix("<|call|>") + "<|return|>", add_special_tokens=False))
turns = iter([real_tokens, list(trader.tokenizer().encode(SUBMIT, add_special_tokens=False))])


def sample_seen(sc, renew, model_input, params, trace, step):
    seen.append(trader.tokenizer().decode(model_input.to_ints()))
    return types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=next(turns), stop_reason="stop")]), sc


agent.sample_once = sample_seen
call, trace, _, _ = agent.gptoss_look(None, types.SimpleNamespace(), "system", "state", "2026-11-02", SPEC, "high")
check("a research call ending in <|return|> runs, and the next prompt reads it as a call",
      call and trace[0].get("tool") == "quote" and '{"symbols": ["A"]}<|call|><|start|>functions.quote' in seen[1])

tools.run, agent.sample_once = real_run, real_sample
print("ALL OK" if all(ok) else "FAILED")
sys.exit(0 if all(ok) else 1)
