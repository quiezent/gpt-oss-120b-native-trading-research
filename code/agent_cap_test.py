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
tools.run = lambda gw, name, args, ml_tool=None, sleeve_value=None, look=None: f"{name} result"   # 16 Nov: the forecast keywords
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

# ---------------------------------------------------------------- the context reserve (16 Nov 2026)
# Fake turns and tool results sized with the real tokenizer to walk the prompt up to the 32,768-token context. The fake
# model calls `script` in turn; once the loop opens the submit (the prompt ends in SUBMIT_OPEN), it writes BODY there.
import json  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

import checks  # noqa: E402

tok = trader.tokenizer()
enc = lambda s: list(tok.encode(s, add_special_tokens=False))
OPEN = agent.SUBMIT_OPEN()
CHECK1 = THINK + ('<|channel|>commentary to=functions.check_orders <|constrain|>json<|message|>{"place": [{"symbol": "SPY", '
                  '"action": "SELL", "quantity": 1, "type": "MKT", "tif": "DAY"}]}<|call|>')
CHECK20 = THINK + ('<|channel|>commentary to=functions.check_orders <|constrain|>json<|message|>{"place": ['
                   + ", ".join(['{"symbol": "SPY", "action": "SELL", "quantity": 1, "type": "MKT", "tif": "DAY"}'] * 20) + ']}<|call|>')
lines = lambda n: "\n".join(["line of text"] * (n // 4))                 # about n tokens, in lines
prompts = []


def sampler(script):
    it = iter(script)

    def sample_once(sc, renew, model_input, params, trace, step):
        ints = model_input.to_ints()
        prompts.append(list(ints))
        text = BODY if ints[-len(OPEN):] == OPEN else next(it)
        toks = enc(text)[:params.max_tokens]
        return types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=toks, stop_reason="stop")]), sc
    return sample_once


def first_prompt_len(state):
    seen = []
    agent.sample_once = lambda sc, renew, mi, params, trace, step: seen.append(len(mi.to_ints())) or (
        types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=enc(SUBMIT), stop_reason="stop")]), sc)
    agent.gptoss_look(None, types.SimpleNamespace(), "system", state, "2026-11-16", SPEC, "high")
    return seen[0]


BASE_LEN = first_prompt_len("")
state_at = lambda n: " the" * (n - BASE_LEN)                             # a state whose first prompt is n tokens
check("the padding gives prompts of the size asked", first_prompt_len(state_at(27700)) == 27700)


def look(n, script, result=None, checker=None, mod=agent):
    prompts.clear()
    tools.run = (lambda gw, name, args, **k: result) if result is not None else (lambda gw, name, args, **k: f"{name} result")
    mod.sample_once = sampler(script)
    call, trace, _, _ = mod.gptoss_look(None, types.SimpleNamespace(), "system", state_at(n), "2026-11-16", SPEC, "high",
                                        checker=checker)
    return call, trace


# (a) a long research result near the end: run, then the reserve opens the submit
call, trace = look(24000, [CALL, CALL], result=lines(7000))
check("(a) a long research result near the end runs; the reserve then opens the submit, and the look decides",
      call and call["notes"] == "forced" and trace[0]["tool"] == "quote" and trace[1].get("note") == "context nearly full: submit opened"
      and trace[-1]["tool"] == "submit", [(t.get("tool"), t.get("note"), t.get("error")) for t in trace])
call, trace = look(24000, [CALL, CALL], result=lines(12000))
cut = trace[0]["result"]
check("(a) a result larger than what is left: cut at a line break, then the cut line and the counter; the prompt leaves "
      "SUBMIT_ROOM; then the submit", call and call["notes"] == "forced"
      and cut.endswith("\n" + agent.CUT_LINE + "\n(research call 1 of 12)") and set(cut.split("\n")[:-2]) == {"line of text"}
      and max(len(p) for p in prompts) <= agent.CONTEXT - agent.SUBMIT_ROOM, (len(prompts[-1]), cut[-60:], trace[-1]))

# (b) checks on a prompt that starts at 27,700 tokens (the 90th percentile at the final turn)
served = []
big_check = lambda args: served.append(1) or (lines(1500), True, None)
call, trace = look(27700, [CHECK1, CHECK1, CHECK1], checker=big_check)
check("(b) three checks asked from 27,700 tokens: those that leave the reserve run, then the submit is opened; a decision",
      call and call["notes"] == "forced" and len(served) == 2 and any(t.get("note") for t in trace) and trace[-1]["tool"] == "submit",
      [(t.get("tool"), t.get("note"), str(t.get("result", ""))[:40]) for t in trace])
served.clear()
call, trace = look(29000, [CHECK20], checker=big_check)
skip = trace[0]["result"]
check("(b) a 20-order check whose result would not leave the reserve: one line, not run, not counted; the submit opened next",
      call and not served and skip.startswith("context nearly full (") and skip.endswith(" tokens left); no check is run")
      and int(skip.split("(")[1].split(" ")[0]) == agent.CONTEXT - len(prompts[0]) - len(enc(CHECK20))
      and trace[-1]["tool"] == "submit" and not any(t.get("note") for t in trace), skip)
call, trace = look(29600, [CALL], result=lines(1000))
check("(b) a research call that would not leave the reserve: 'no research call is run', not counted; the submit next",
      call and trace[0]["result"].endswith("tokens left); no research call is run") and trace[-1]["tool"] == "submit", trace[0]["result"])

# (c) a prompt already inside the reserve at step 0, and (d) one with under 300 tokens of room
call, trace = look(31000, [CALL])
check("(c) inside the reserve at step 0: the submit is opened at once (recorded with the room), and the look decides",
      call and call["notes"] == "forced" and trace[0] == {"step": 0, "room": agent.CONTEXT - 31000,
                                                          "note": "context nearly full: submit opened"}, trace[:1])
call, trace = look(32600, [CALL])
check("(d) under 300 tokens of room even with the submit opened: 'context full', no decision, no exception",
      call is None and "context full" in trace[-1]["error"] and not prompts, trace[-1])
TRUNC = THINK + '<|channel|>commentary to=functions.submit <|constrain|>json<|message|>{"place": [], "canc'
cut_once = iter([enc(TRUNC)])                                           # the opened submit cut off once, then written
agent.sample_once = lambda sc, renew, mi, params, trace, step: (types.SimpleNamespace(sequences=[types.SimpleNamespace(
    tokens=next(cut_once, enc(BODY)), stop_reason="stop")]), sc)
prompts.clear()
call, trace, _, _ = agent.gptoss_look(None, types.SimpleNamespace(), "system", state_at(31000), "2026-11-16", SPEC, "high")
check("a cut-off answer in the opened submit is re-asked with the submit still opened", call and call["notes"] == "forced"
      and "error" in trace[1] and trace[-1]["tool"] == "submit", [(t.get("note"), str(t.get("error"))[:30]) for t in trace])

# a look far from the limit (8,000 tokens) runs exactly as the committed agent.py of 0576413 (before the reserve) and of
# 23a7075 (before SUBMIT_ROOM): the fake model's turns are a few hundred tokens, under any max_tokens asked
def committed(rev):
    src = subprocess.run(["git", "show", f"{rev}:agent.py"], capture_output=True, text=True, encoding="utf-8",
                         cwd=Path(__file__).resolve().parent).stdout
    f = Path(tempfile.mkdtemp()) / f"agent_{rev}.py"
    f.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"agent_{rev}", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


small_check = lambda args: ("checked\nfine", True, {"x": 1})
for rev in ("0576413", "23a7075"):
    old_agent, same = committed(rev), []
    for script, chk in (([CALL, CALL, SUBMIT], None), ([CALL, CHECK1, CALL, CHECK1, SUBMIT], small_check),
                        ([CALL] * (N + 2) + [BAD, BODY], None), ([PROSE, PROSE, BODY], small_check)):
        c_new, t_new = look(8000, script, result=lines(400), checker=chk)
        p_new = [list(p) for p in prompts]
        c_old, t_old = look(8000, script, result=lines(400), checker=chk, mod=old_agent)
        same.append(c_new == c_old and t_new == t_old and p_new == prompts and c_new is not None)
    check(f"far from the limit (8,000 tokens): the same prompts, trace entries and decision as {rev}, step for step", all(same), same)

# ---------------------------------------------------------------- SUBMIT_ROOM (16 Nov 2026): the checker's case
# Their sampler (Trader-models mon2/verify/reserve_effect.py): every free turn is 3,000 tokens of reasoning and then a call;
# a turn longer than max_tokens is cut there (stop "length" unless it ended on the call); the opened submit writes the
# decision. Swept over starting prompts every 64 tokens from 20,000 to 30,500, link working and absent.
FIX = json.loads(Path("C:/Claude/TWS-Gateway-handoff/whatif_fixture_20261115.json").read_text(encoding="utf-8"))


class FixGW:                                                # a what-if gateway answering from the Gateway Owner's fixture
    def account(self):
        return dict(FIX["account"])

    def positions(self):
        return [{"symbol": "SPY", "quantity": 82}]

    def all_open_orders(self):
        return []

    def whatif_many(self, orders, timeout=10.0):
        return [dict(FIX["whatifs"]["R3"]["result"]) for _ in orders]


CTX = {"value": 90000.0, "positions": {}, "prices": {"SPY": [749.6, "delayed last"]}, "sleeves": 9, "looking": 9}
up = checks.Link(FixGW())
CHECKERS = {"working": lambda a: checks.check(up, a, CTX), "absent": lambda a: checks.check(None, a, CTX)}
HEAD_T = enc("<|channel|>analysis<|message|>")
FILL = enc(" step" * 3000)[:3000]
ORDER = {"symbol": "SPY", "action": "SELL", "quantity": 100, "type": "TRAIL", "trail_percent": 0.02, "tif": "GTC"}
TURN = {"check_orders": HEAD_T + FILL + enc('<|end|><|start|>assistant<|channel|>commentary to=functions.check_orders <|constrain|>'
                                            'json<|message|>' + json.dumps({"place": [ORDER]}) + "<|call|>"),
        "quote": HEAD_T + FILL + enc('<|end|><|start|>assistant<|channel|>commentary to=functions.quote <|constrain|>json'
                                     '<|message|>{"symbols": ["SPY"]}<|call|>')}
DECISION = enc(json.dumps({"place": [ORDER], "cancel": [], "notes": "decided"}) + "<|call|>")
forced_rooms, asked, free_asked = [], [], []


def their_sampler(kinds):
    k = {"n": 0}

    def sample_once(sc, renew, model_input, params, trace, step):
        ints = model_input.to_ints()
        if ints[-len(OPEN):] == OPEN:
            forced_rooms.append(agent.CONTEXT - len(ints))
            t = DECISION
        else:
            t = TURN[kinds[k["n"] % len(kinds)]]
            k["n"] += 1
            free_asked.append(params.max_tokens)
        asked.append(params.max_tokens)
        t = t[:params.max_tokens]
        stop = "length" if len(t) >= params.max_tokens and t[-1] != trader.CALL else "stop"
        return types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=t, stop_reason=stop)]), sc
    return sample_once


DROP_NOTE = "turn left too little room for the decision: dropped, submit opened"


def sweep(kinds, checker, mod=agent):
    ends = {"submit": 0, "context full": 0, "step limit": 0, "other": 0, "with a dropped turn": 0}
    for n in range(20000, 30501, 64):
        tools.run = lambda gw, name, args, **kw: lines(1500)
        mod.sample_once = their_sampler(kinds)
        call, trace, _, _ = mod.gptoss_look(None, types.SimpleNamespace(), "system", state_at(n), "2026-11-16", SPEC, "high",
                                              checker=checker)
        errs = " ".join(str(t.get("error")) for t in trace if t.get("error"))
        key = ("submit" if call and call.get("notes") == "decided" else "context full" if "context full" in errs
               else "step limit" if "step limit" in errs else "other")
        ends[key] += 1
        ends["with a dropped turn"] += any(t.get("note") == DROP_NOTE for t in trace)
    return ends


gap = sweep(["check_orders"], CHECKERS["working"], mod=committed("23a7075"))
print(f"     (23a7075, before SUBMIT_ROOM, the same sweep with only checks: {gap})")
forced_rooms.clear()
for label, kinds, chk in (("only checks, link working", ["check_orders"], CHECKERS["working"]),
                          ("only checks, link absent", ["check_orders"], CHECKERS["absent"]),
                          ("only research", ["quote"], None), ("research and checks alternating", ["quote", "check_orders"],
                                                               CHECKERS["working"])):
    ends = sweep(kinds, chk)
    check(f"3,000-token turns, {label}, from 20,000 to 30,500 tokens every 64 (165 looks): every look decides", ends["submit"] == 165,
          ends)
    print(f"     ({label}: {ends['with a dropped turn']} looks had a dropped turn)")
# The least room an opened submit can have after any turns from a prompt with room >= RESERVE: every accepted free
# tool-call turn leaves SUBMIT_ROOM; a result that runs is cut to leave SUBMIT_ROOM after the prefix; only the framework's
# one-line reply (a call not run, or a limit text), added whole, and the prefix can come out of it. A dropped turn, a turn
# cut at the room and a second malformed turn open the submit at a prompt whose room was >= RESERVE.
one_liners = ["context nearly full (9999 tokens left); no research call is run", "context nearly full (9999 tokens left); no check is run",
              f"research limit reached ({tools.MAX_CALLS} calls): call submit now",
              checks.cap_text(0).rstrip(".") + "; call submit now", checks.cap_text(3).rstrip(".") + "; call submit now"]
FLOOR = agent.SUBMIT_ROOM - max(len(agent.tool_result_tokens(nm, t)) for t in one_liners for nm in ("quote", "check_orders")) - len(OPEN)
check(f"the opened submit always had at least {FLOOR} tokens of room (SUBMIT_ROOM {agent.SUBMIT_ROOM} less the longest one-line "
      f"reply and the submit's prefix); least seen {min(forced_rooms)}", min(forced_rooms) >= FLOOR, min(forced_rooms))
check("a free turn is offered all the room (no cap): min(MAX_OUT, room)", all(a >= agent.RESERVE for a in free_asked), min(free_asked))

# a long decision that fits is kept: one sample, no note, the same as 23a7075 step for step
OLD = committed("23a7075")
SUB_TAIL = enc('<|end|><|start|>assistant<|channel|>commentary to=functions.submit <|constrain|>json<|message|>'
               + json.dumps({"place": [ORDER], "cancel": [], "notes": "long"}) + "<|call|>")
PAD_T = enc(" step" * 16000)[:16000]


def sized(tail, n):
    """A free turn of exactly n tokens: reasoning, then `tail` (a call)."""
    assert n >= len(HEAD_T) + len(tail), n
    return HEAD_T + PAD_T[:n - len(HEAD_T) - len(tail)] + tail


def scripted(turns, record):
    """A sampler answering free turns from `turns` (token lists, or callables of the prompt's length), the decision when the
    submit is opened; cut at max_tokens as the checker's sampler does. record gets (prompt, max_tokens, tokens)."""
    it = iter(turns)

    def sample_once(sc, renew, model_input, params, trace, step):
        ints = list(model_input.to_ints())
        t = DECISION if ints[-len(OPEN):] == OPEN else next(it, None)
        t = enc(PROSE) if t is None else t(len(ints)) if callable(t) else t     # past the script: prose (shows in the trace)
        t = t[:params.max_tokens]
        record.append((ints, params.max_tokens, list(t)))
        stop = "length" if len(t) >= params.max_tokens and t[-1] != trader.CALL else "stop"
        return types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=t, stop_reason=stop)]), sc
    return sample_once


def run_mod(mod, n, turns, checker=None, result=None):
    rec = []
    tools.run = (lambda gw, name, args, **k: result) if result is not None else (lambda gw, name, args, **k: f"{name} result")
    mod.sample_once = scripted(turns, rec)
    call, trace, _, _ = mod.gptoss_look(None, types.SimpleNamespace(), "system", state_at(n), "2026-11-16", SPEC, "high",
                                        checker=checker)
    return call, trace, rec


kept = []
for room in (3000, 5500, 9000):
    n = agent.CONTEXT - room
    for length in (room - 1500, room - 800, room - 50):
        turn = sized(SUB_TAIL, length)
        c_new, t_new, r_new = run_mod(agent, n, [turn])
        c_old, t_old, r_old = run_mod(OLD, n, [turn])
        kept.append(c_new and c_new["notes"] == "long" and len(r_new) == 1 and not any(t.get("note") for t in t_new)
                    and (c_new, t_new, r_new) == (c_old, t_old, r_old))
check("a long decision that fits (room 3,000 / 5,500 / 9,000; length room-1,500 to room-50) is kept: one sample, no note, "
      "the same as 23a7075", all(kept), kept)

# a dropped turn: a check turn that leaves under SUBMIT_ROOM is not added, not run, not counted; the submit is opened at
# the prompt as it was before the turn
QUOTE_TAIL = TURN["quote"][-len(enc('<|end|><|start|>assistant<|channel|>commentary to=functions.quote <|constrain|>json'
                                    '<|message|>{"symbols": ["SPY"]}<|call|>')):]
CHECK_TAIL = TURN["check_orders"][-len(enc('<|end|><|start|>assistant<|channel|>commentary to=functions.check_orders <|constrain|>'
                                           'json<|message|>' + json.dumps({"place": [ORDER]}) + "<|call|>")):]
ran = []
count_check = lambda a: ran.append(a) or ("checked", True, None)
n = agent.CONTEXT - 4000
long_check = lambda plen: sized(CHECK_TAIL, agent.CONTEXT - plen - 1000)     # leaves 1,000
call, trace, rec = run_mod(agent, n, [long_check], checker=count_check)
drop = next((t for t in trace if t.get("note") == DROP_NOTE), {})
check("a dropped turn: the next (opened) sample's prompt is the prompt before the turn plus the submit's prefix; nothing run "
      "or counted; the look decides", call and call["notes"] == "decided" and len(rec) == 2 and rec[1][0] == rec[0][0] + OPEN
      and not ran and drop.get("left") == 1000 and drop.get("dropped") == "check_orders" and "tool" not in drop and "result" not in drop
      and not any("order check" in str(t.get("result")) for t in trace), (len(rec), drop.get("left"), ran))

# the framework's own lines reach the prompt whole at the smallest left an accepted turn can leave (SUBMIT_ROOM), and the
# trace's result is what the prompt holds
exact = lambda tail: (lambda plen: sized(tail, agent.CONTEXT - plen - agent.SUBMIT_ROOM))      # leaves exactly SUBMIT_ROOM
SHORT_Q, SHORT_C = sized(QUOTE_TAIL, len(HEAD_T) + len(QUOTE_TAIL) + 40), sized(CHECK_TAIL, len(HEAD_T) + len(CHECK_TAIL) + 40)
cases = {"not run (check)": ([exact(CHECK_TAIL)], count_check, "no check is run"),
         "not run (research)": ([exact(QUOTE_TAIL)], None, "no research call is run"),
         "research limit": ([SHORT_Q] * N + [exact(QUOTE_TAIL)], None, "research limit reached"),
         "check limit": ([SHORT_C] * 3 + [exact(CHECK_TAIL)], count_check, "check limit reached (3 of 3)")}
whole = {}
for label, (turns, chk, words) in cases.items():
    call, trace, rec = run_mod(agent, 12000 if "limit" in label else 24000, turns, checker=chk)
    hit = next((t for t in trace if words in str(t.get("result"))), None)
    nxt = tok.decode(rec[-1][0])
    whole[label] = bool(call and hit and agent.CUT_LINE not in hit["result"] and hit["result"] in nxt
                        and agent.CUT_LINE not in nxt and agent.CONTEXT - len(rec[-1][0]) >= FLOOR)
    if not whole[label]:
        print("     ", label, [(t.get("tool"), t.get("note"), t.get("error"), str(t.get("result"))[-70:]) for t in trace][-4:])
check("the framework's lines (not run, check and research; the research-limit and check-limit texts) reach the prompt whole "
      "at left = SUBMIT_ROOM, the trace holds what the prompt holds, and the opened submit keeps the floor", all(whole.values()), whole)

# a turn cut at max_tokens is a malformed turn as at 23a7075: re-asked free at the same prompt; a second opens the submit
SUB_FREE = lambda n: sized(SUB_TAIL, n)
runaway = lambda plen: HEAD_T + PAD_T[:16000]                      # reasoning with no call: cut at whatever room is given
reask = []
for room in (8000, 3000):
    n = agent.CONTEXT - room
    for turns in ([runaway, SUB_FREE(700)], [runaway, runaway]):
        c_new, t_new, r_new = run_mod(agent, n, turns)
        c_old, t_old, r_old = run_mod(OLD, n, turns)
        same = (c_new, t_new, r_new) == (c_old, t_old, r_old)
        if turns[1] is runaway:                                     # two cuts: the submit opened, then the decision
            reask.append(same and c_new and c_new["notes"] == "decided" and len(r_new) == 3 and r_new[2][0][-len(OPEN):] == OPEN)
        else:                                                       # one cut: re-asked free at the same prompt, which decides
            reask.append(same and c_new and c_new["notes"] == "long" and r_new[1][0] == r_new[0][0] and r_new[1][1] == r_new[0][1]
                         and not any(t.get("note") for t in t_new))
check("a free turn cut at max_tokens (room 8,000 and 3,000) is re-asked free at the same prompt, as at 23a7075 step for step "
      "(prompts, max_tokens, trace, decision); two cuts in a row open the submit and the look decides", all(reask), reask)

# the reviewer's case: prompt about 24,745 tokens, a free turn runs to the room, the re-ask writes a reasoned submit
n = 24745
c_new, t_new, r_new = run_mod(agent, n, [runaway, SUB_FREE(1740)])
check("the reviewer's case (prompt 24,745): the decision is the re-asked free submit, in two samples, no note",
      c_new and c_new["notes"] == "long" and len(r_new) == 2 and r_new[0][1] == r_new[1][1] == agent.CONTEXT - len(r_new[0][0])
      and len(r_new[1][2]) == 1740 and not any(t.get("note") for t in t_new), [(len(p), mx, len(t)) for p, mx, t in r_new])

# a cut result keeps its counter as the last line, after CUT_LINE; the trace holds what the prompt holds
at_left = lambda tail, left: (lambda plen: sized(tail, agent.CONTEXT - plen - left))
cuts = {}
for label, tail, chk, res, counter in (
        ("research, many lines", QUOTE_TAIL, None, lines(20000), "(research call 1 of 12)"),
        ("check, many lines", CHECK_TAIL, lambda a: (lines(20000), True, None), None, "(order check 1 of 3)"),
        ("research, one 40,000-token line", QUOTE_TAIL, None, " the" * 40000, "(research call 1 of 12)")):
    call, trace, rec = run_mod(agent, 12000, [at_left(tail, 10000)], checker=chk, result=res)
    r = next(t for t in trace if t.get("tool") in ("quote", "check_orders"))["result"]
    body = r.split("\n" + agent.CUT_LINE + "\n")[0]
    nxt = tok.decode(rec[-1][0])
    cuts[label] = bool(call and r.endswith("\n" + agent.CUT_LINE + "\n" + counter) and len(enc(body)) > 5000 and r in nxt
                       and agent.CONTEXT - len(rec[-1][0]) >= agent.SUBMIT_ROOM >= FLOOR)
call, trace, rec = run_mod(agent, 12000, [at_left(QUOTE_TAIL, 10000), SUB_FREE(300)], result=lines(400))
cuts["uncut"] = trace[0]["result"] == lines(400) + "\n(research call 1 of 12)" and call
check("a cut result: the part that fits (whole lines, or the start of a single long line), CUT_LINE, then the counter; "
      "the trace equals the prompt; an uncut result unchanged; the opened submit keeps SUBMIT_ROOM", all(cuts.values()), cuts)

# property check: random scripted looks against 23a7075
import random  # noqa: E402

rmod_saved = sys.modules.get("tinker_cookbook.renderers.gpt_oss")
FILLER = enc(" the")[0]
BASE_P = []


class PromptR:                                              # the prompt as a list of n tokens: no tokenizing of 30,000 tokens a look
    def __init__(self, *a, **k):
        pass

    def create_conversation_prefix_with_tools(self, specs, system):
        return []

    def build_generation_prompt(self, msgs):
        return types.SimpleNamespace(to_ints=lambda: list(BASE_P))


pr = types.ModuleType("tinker_cookbook.renderers.gpt_oss")
pr.GptOssRenderer = PromptR
sys.modules["tinker_cookbook.renderers.gpt_oss"] = pr
rng = random.Random(1116)
lognorm = lambda med, p90: max(40, int(med * (p90 / med) ** (rng.gauss(0, 1) / 1.2816)))   # median and 90th percentile
TAILS = {"quote": QUOTE_TAIL, "check_orders": CHECK_TAIL, "submit": SUB_TAIL}
stats = {"looks": 0, "decided": 0, "context full": 0, "compared": 0, "same": 0, "dropped": 0, "cut": 0, "malformed": 0, "bad": []}
t0 = __import__("time").time()
for i in range(2400):
    n = rng.randint(11000, 31500)
    BASE_P[:] = [FILLER] * n
    specs = []
    for _ in range(24):
        tool = rng.choices(["quote", "check_orders", "submit", "runaway"], weights=[6, 3, 1, 0.4])[0]
        if tool == "runaway":                               # reasoning with no call: malformed, or cut at max_tokens
            specs.append(HEAD_T + PAD_T[:min(lognorm(6000, 14000), 15900)])
            continue
        L = lognorm(2974, 6336) if tool == "submit" else lognorm(178, 2678)
        specs.append(sized(TAILS[tool], min(L, 15000) + len(HEAD_T) + len(TAILS[tool])))
    sizes = [rng.choice([200, 800, 2000, 4000]) for _ in range(24)]
    chk = (lambda a: ("checked\n" + lines(600), True, None)) if rng.random() < 0.7 else None
    if chk is None:
        specs = [s for s in specs if s[-len(CHECK_TAIL):] != CHECK_TAIL] or [sized(SUB_TAIL, 500)]
    out = {}
    for tag, mod in (("new", agent), ("old", OLD)):
        it_sizes = iter(sizes * 2)
        tools.run = lambda gw, name, args, **k: lines(next(it_sizes))
        rec = []
        mod.sample_once = scripted(list(specs), rec)
        try:
            out[tag] = mod.gptoss_look(None, None, "s", "s", "2026-11-16", SPEC, "high", checker=chk)[:2] + (rec,)
        except StopIteration:
            out[tag] = None
    stats["looks"] += 1
    if out["new"] is None:
        stats["bad"].append((i, n, "ran out of scripted turns"))
        continue
    c, tr, rec = out["new"]
    if c is not None:
        stats["decided"] += 1
    elif any("context full" in str(t.get("error")) for t in tr) and agent.CONTEXT - n < agent.MIN_SUBMIT:
        stats["context full"] += 1
    else:
        stats["bad"].append((i, n, [t.get("note") or t.get("error") or t.get("tool") for t in tr]))
    stats["dropped"] += any(t.get("note") == DROP_NOTE for t in tr)
    stats["cut"] += any(agent.CUT_LINE in str(t.get("result")) for t in tr)
    stats["malformed"] += any(t.get("error") for t in tr)
    # every look where no tool-call turn left under SUBMIT_ROOM (a drop) and no result needed a cut under the new budget,
    # malformed and cut-off turns included: the same prompts asked, max_tokens, sampled turns, trace and decision
    if out["old"] is not None and not any(t.get("note") == DROP_NOTE or agent.CUT_LINE in str(t.get("result")) for t in tr):
        stats["compared"] += 1
        stats["same"] += (c, tr, rec) == out["old"]
secs = __import__("time").time() - t0
if rmod_saved is None:
    sys.modules.pop("tinker_cookbook.renderers.gpt_oss", None)
else:
    sys.modules["tinker_cookbook.renderers.gpt_oss"] = rmod_saved
check(f"{stats['looks']} random looks (prompts 11,000-31,500, recorded turn shapes): every look decides, or 'context full' "
      f"only when it starts under MIN_SUBMIT", not stats["bad"] and stats["decided"] + stats["context full"] == stats["looks"],
      stats["bad"][:2])
check(f"every look with no dropped turn and no cut result ({stats['compared']} looks, malformed and cut-off turns included): "
      f"the same prompts, max_tokens, turns, trace and decision as 23a7075", stats["compared"] > 1000 and stats["same"] == stats["compared"],
      (stats["same"], stats["compared"]))
print(f"     (random looks: {stats['looks']}; compared {stats['compared']}, identical {stats['same']}; {stats['dropped']} with a dropped "
      f"turn, {stats['cut']} with a cut result, {stats['malformed']} with a malformed or cut-off turn; {secs:.0f} s)")

tools.run, agent.sample_once = real_run, real_sample
print("ALL OK" if all(ok) else "FAILED")
sys.exit(0 if all(ok) else 1)
