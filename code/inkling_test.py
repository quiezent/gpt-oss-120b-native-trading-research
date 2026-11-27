"""Offline checks of inkling.look against a fake chat-completions endpoint ($0, no Gateway, no network), and of the
per-sleeve model and context keys in live.py (20 Nov 2026)."""

import copy
import http.client
import json
import sys

import agent
import checks
import inkling
import trader

FAILS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


class Fake:
    """Replies in order; each is a dict (a reply) or an exception to raise. Keeps a copy of every request body."""

    def __init__(self, replies):
        self.replies, self.bodies = list(replies), []

    def __call__(self, body):
        self.bodies.append(copy.deepcopy(body))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


N = [0]


def rep(calls=(), content="", reasoning="thinking", finish=None, pt=12000, ct=300, cached=0, ids=None):
    tc = []
    for k, (name, args) in enumerate(calls):
        N[0] += 1
        tc.append({"id": ids[k] if ids else f"call_{N[0]}", "type": "function",
                   "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}})
    N[0] += 1
    return {"id": f"resp-{N[0]}", "choices": [{"message": {"role": "assistant", "content": content or None, "reasoning_content": reasoning,
                                     "tool_calls": tc or None},
                         "finish_reason": finish or ("tool_calls" if tc else "stop")}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct,
                      "prompt_tokens_details": {"cached_tokens": cached}}}


SPECS = [{"name": n, "description": f"{n} tool", "parameters": {"type": "object", "properties": {}}}
         for n in ("tape", "quote", "news")]
SUB = {"place": [{"symbol": "AAPL", "action": "BUY", "quantity": 10, "type": "MKT"}], "cancel": [], "notes": "n"}
RAN = []
CTX = inkling.CONTEXT - inkling.MARGIN       # what the room is counted against


def run_tool(gw, name, args, **_):
    RAN.append((name, args))
    return f"result of {name} {json.dumps(args, sort_keys=True)}"


def go(replies, **kw):
    RAN.clear()
    f = Fake(replies)
    kw.setdefault("run_tool", run_tool)
    out = inkling.look(None, "SYSTEM TEXT", "STATE TEXT", SPECS, send=f, **kw)
    return out, f


def stats(trace):
    return trace[-1]["inkling"]


inkling.WAITS = (0, 0)                        # no sleeping in tests

# 1. A plain look: two research calls, then submit
(call, trace, pre, smp), f = go([rep([("tape", {})], pt=11000, ct=200), rep([("quote", {"symbols": ["AAPL"]})], pt=11500, ct=250),
                                 rep([("submit", SUB)], pt=12000, ct=400, cached=11000)], max_calls=15, final_line=True)
check("plain: decision is the submit", call == SUB, call)
check("plain: tools ran in order", [r[0] for r in RAN] == ["tape", "quote"], RAN)
check("plain: cost from usage", pre == 34500 and smp == 850, (pre, smp))
check("plain: said holds reasoning_content", all(e.get("said") == "thinking" for e in trace if "tokens" in e))
b = f.bodies[2]
check("plain: history has assistant turns with reasoning and tool results by id",
      [m["role"] for m in b["messages"]] == ["system", "user", "assistant", "tool", "assistant", "tool"]
      and b["messages"][3]["tool_call_id"] == b["messages"][2]["tool_calls"][0]["id"]
      and b["messages"][2]["reasoning_content"] == "thinking", [m["role"] for m in b["messages"]])
check("plain: counter line on research results", b["messages"][3]["content"].endswith("(research call 1 of 15)"), b["messages"][3]["content"])
check("plain: request fields", f.bodies[0]["tool_choice"] == "auto" and f.bodies[0]["reasoning_effort"] == 0.99
      and f.bodies[0]["temperature"] == 1.0 and f.bodies[0]["model"] == "thinkingmachines/Inkling-2-Small"
      and f.bodies[0]["max_completion_tokens"] == agent.MAX_OUT and "seed" not in f.bodies[0], f.bodies[0].keys())
check("plain: tools in OpenAI form, submit last", [t["function"]["name"] for t in f.bodies[0]["tools"]] == ["tape", "quote", "news", "submit"])
s = stats(trace)
check("plain: stats on the decision's entry", trace[-1].get("tool") == "submit" and s["requests"] == 3 and s["forced_turns"] == 0
      and s["http_422"] == 0 and s["cached_tokens"] == 11000 and s["context"] == 131072, s)
check("plain: estimate and server count recorded", all("est_prompt" in e and "prompt_tokens" in e for e in trace if "tokens" in e))
rids = [e.get("response_id") for e in trace if "tokens" in e]
check("plain: each reply's response id recorded (the only per-sleeve cost record; billing is one shared session)",
      len(rids) == 3 and all(rids) and len(set(rids)) == 3, rids)
import re as _re
got = [(e.get("prompt_tokens"), e.get("cached_tokens"), e.get("tokens")) for e in trace if "tokens" in e]
check("plain: each reply's usage recorded (prompt, cached, completion)",
      got == [(11000, 0, 200), (11500, 0, 250), (12000, 11000, 400)], got)
import calendar as _cal
import time as _time


def utc_now(u):                           # a "utc" stamp of the form billing's hours read, and UTC (not local) time now
    if not _re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", u or ""):
        return False
    return abs(_cal.timegm(_time.strptime(u, "%Y-%m-%dT%H:%M:%SZ")) - _time.time()) < 120


check("plain: each reply's UTC time recorded, in UTC", all(utc_now(e.get("utc")) for e in trace if "tokens" in e), trace)
(call, trace, *_), f = go([rep([("tape", {})], pt=11000, ct=200, cached=5000), rep([("quote", {"symbols": ["AAPL"]})], pt=11500, ct=250, cached=6000),
                           rep([("submit", SUB)], pt=12000, ct=400, cached=7000)], max_calls=15, final_line=True)
got = [e.get("cached_tokens") for e in trace if "tokens" in e]
check("cached: each reply's own cached tokens, not a running total; the look's total in its stats",
      got == [5000, 6000, 7000] and stats(trace)["cached_tokens"] == 18000, (got, stats(trace)))

# 2. Two malformed turns open the submit; the last try is plain (effort 0.0), as gpt-oss's plain opened submit
(call, trace, *_), f = go([rep(content="I think..."), rep([("quote", "{bad json")]), rep([("submit", SUB)], reasoning="plan")])
check("malformed x2 then forced submit: decision", call == SUB)
check("malformed: third request names submit", f.bodies[2]["tool_choice"] == inkling.SUBMIT_CHOICE and f.bodies[1]["tool_choice"] == "auto")
check("malformed: the last try runs at the plain effort", f.bodies[2]["reasoning_effort"] == inkling.FORCED_PLAIN_EFFORT
      and trace[2].get("effort") == inkling.FORCED_PLAIN_EFFORT, f.bodies[2]["reasoning_effort"])
check("malformed: forced entry has forced and reasoned", trace[2].get("forced") is True and trace[2].get("reasoned") is True, trace[2])
check("malformed: errors traced", trace[0]["error"].startswith("MALFORMED: output ended without a tool call") and "bad JSON" in trace[1]["error"])
check("malformed: no tool ran", RAN == [])
s = stats(trace)
check("malformed: stats", s["forced_turns"] == 1 and s["forced_reasoned"] == 1 and s["forced_plain"] == 1, s)

# 3. Three 422s: no decision, counted
e422 = lambda: inkling.HttpError(422, "reply broke tool_choice")
(call, trace, *_), f = go([e422(), e422(), e422()])
check("422 x3: no decision", call is None)
check("422 x3: counted", stats(trace)["http_422"] == 3 and all(t.get("http") == 422 for t in trace[:3]), trace)
check("422 x3: each attempt's UTC time recorded", all(t.get("utc", "").endswith("Z") for t in trace[:3]), trace)
check("422 x3: second 422 opens the submit", f.bodies[2]["tool_choice"] == inkling.SUBMIT_CHOICE)

# 4. The research limit, told and then forced (a turn apart), with the final words; the forced turn at the sleeve's effort
(call, trace, *_), f = go([rep([("tape", {})]), rep([("quote", {"symbols": ["A"]})]), rep([("news", {})]), rep([("news", {})]),
                           rep([("submit", SUB)], reasoning="")], max_calls=2, final_line=True)
results = [t.get("result") for t in trace if t.get("tool") and t["tool"] != "submit"]
check("limit: third call told", results[2] == "research limit reached (2 calls): call submit now", results)
check("limit: fourth call's line has the final words", results[3] == "research limit reached (2 calls): " + agent.final_words(False), results[3])
check("limit: next request forced", f.bodies[4]["tool_choice"] == inkling.SUBMIT_CHOICE and f.bodies[3]["tool_choice"] == "auto")
check("limit: forced turn at the sleeve's effort, its lack of reasoning recorded", f.bodies[4]["reasoning_effort"] == 0.99
      and trace[-1].get("reasoned") is False and stats(trace)["forced_reasoned"] == 0 and stats(trace)["forced_plain"] == 0, trace[-1])
check("limit: only two tools ran", len(RAN) == 2)

# 4b. Several past-limit calls in one reply: the warning only; the next turn stays free (gpt-oss's turn after the warning)
(call, trace, *_), f = go([rep([("tape", {}), ("quote", {}), ("news", {}), ("news", {})]), rep([("submit", SUB)])], max_calls=2, final_line=True)
res = [x["content"] for x in f.bodies[1]["messages"] if x["role"] == "tool"]
check("per reply: both past-limit calls told, none opens the submit", res[2:] == ["research limit reached (2 calls): call submit now"] * 2
      and f.bodies[1]["tool_choice"] == "auto", (res, f.bodies[1]["tool_choice"]))

# 4c. Once a line in a reply opens the submit, the later calls are not run and get that line
def checker0(args):
    raise AssertionError("the check ran")
checker0.own_limit = None
(call, trace, *_), f = go([rep([("tape", {})]), rep([("news", {})]), rep([("news", {}), ("check_orders", {"place": []})]),
                           rep([("submit", SUB)])], max_calls=1, final_line=True, checker=checker0)
res = [x["content"] for x in f.bodies[3]["messages"] if x["role"] == "tool"]
final_l = "research limit reached (1 calls): " + agent.final_words(True)
check("after the opening line: the check is not run and reads the same line", res[-2:] == [final_l, final_l]
      and trace[3].get("not_run") and f.bodies[3]["tool_choice"] == inkling.SUBMIT_CHOICE, (res, trace[3]))

# 5. Several calls in one reply run in order, each answered by its id
(call, trace, *_), f = go([rep([("quote", {"symbols": ["A"]}), ("news", {"symbol": "A"})]), rep([("submit", SUB)])], max_calls=15)
m = f.bodies[1]["messages"]
check("parallel: both ran", [r[0] for r in RAN] == ["quote", "news"])
check("parallel: two tool messages matched to the two ids",
      [x["tool_call_id"] for x in m if x["role"] == "tool"] == [c["id"] for c in m[2]["tool_calls"]], m)
check("parallel: counted as two research calls", m[4]["content"].endswith("(research call 2 of 15)"), m[4]["content"])

# 5b. Repeated or missing ids are numbered by the loop; arguments are re-sent as a JSON string
r5 = rep([("quote", {"symbols": ["A"]}), ("news", {"symbol": "A"})], ids=["dup", "dup"])
r5["choices"][0]["message"]["tool_calls"][1]["function"]["arguments"] = {"symbol": "A"}   # a dict, not a string
(call, trace, *_), f = go([r5, rep([("submit", SUB)])])
m = f.bodies[1]["messages"]
ids = [c["id"] for c in m[2]["tool_calls"]]
check("ids: repeated ids renumbered, each answered", len(set(ids)) == 2 and [x["tool_call_id"] for x in m[3:5]] == ids, ids)
check("ids: arguments re-sent as strings", all(isinstance(c["function"]["arguments"], str) for c in m[2]["tool_calls"]))

# 6. A submit beside another call: the submit is taken, the other not run; two submits are malformed
(call, trace, *_), f = go([rep([("quote", {"symbols": ["A"]}), ("submit", SUB)])])
check("submit beside a call: decision", call == SUB and RAN == [])
check("submit beside a call: not_run named", trace[0].get("not_run") == ["quote"], trace[0])
(call, trace, *_), f = go([rep([("submit", SUB), ("submit", {"place": [], "cancel": []})]), rep([("submit", SUB)])])
check("two submits: one malformed turn, then the decision", "two submit calls" in trace[0].get("error", "") and call == SUB, trace[0])

# 7. The missing-price re-ask, once a look
lmt = {"place": [{"symbol": "AAPL", "action": "BUY", "quantity": 10, "type": "LMT"}], "cancel": [], "notes": "x"}
(call, trace, *_), f = go([rep([("submit", lmt)]), rep([("submit", lmt)])])
check("missing price: re-asked once, then taken as written", call == lmt and trace[0].get("reask") == "submit" and trace[1].get("tool") == "submit", trace[:2])
m = f.bodies[1]["messages"]
check("missing price: answered on the submit's id", m[-1]["role"] == "tool" and m[-1]["tool_call_id"] == m[-2]["tool_calls"][0]["id"], m[-2:])
(call, trace, *_), f = go([rep([("submit", lmt)]), inkling.HttpError(503, "x"), inkling.HttpError(503, "y"), inkling.HttpError(503, "z")])
check("missing price: failed request after the re-ask falls back to the held submit",
      call == lmt and any(t.get("fallback") == "reask" for t in trace), trace)
check("missing price: the failed request after the re-ask has its UTC time",
      [utc_now(t.get("utc")) for t in trace if "after the re-ask" in t.get("error", "")] == [True], trace)
(call, trace, *_), f = go([rep([("submit", lmt)]), TimeoutError(), TimeoutError()])
check("missing price: a request that times out twice after the re-ask falls back, with its UTC time",
      call == lmt and [utc_now(t.get("utc")) for t in trace if "after the re-ask: TimeoutError" in t.get("error", "")] == [True], trace)
(call, trace, *_), f = go([rep(content=json.dumps(lmt)), rep([("submit", SUB)])])
m = f.bodies[1]["messages"]
check("missing price: a text decision is re-asked in a user message", call == SUB and trace[0].get("answered_as") == "user"
      and m[-1]["role"] == "user" and m[-1]["content"].startswith("submit not accepted yet") and m[-2]["role"] == "assistant", m[-2:])

# 8. The context reserve opens the submit
long_tool = lambda gw, name, args, **_: "word " * 1800        # about 2,300 estimated: run (4,500 - 1,907 >= RESERVE), then
(call, trace, *_), f = go([rep([("tape", {})], pt=CTX - 4500 - 100 - inkling.FRAME, ct=100),   # the room is under it
                           rep([("submit", SUB)])], run_tool=long_tool)
last = f.bodies[1]["messages"][-1]["content"]
check("reserve: the result ran and was kept whole", "word word" in last and agent.CUT_LINE not in last, last[-80:])
check("reserve: room under RESERVE, submit opened", f.bodies[1]["tool_choice"] == inkling.SUBMIT_CHOICE
      and any(t.get("note") == "context nearly full: submit opened" for t in trace), [t.get("note") for t in trace])
check("reserve: max_completion_tokens is the room", f.bodies[1]["max_completion_tokens"] < agent.RESERVE, f.bodies[1]["max_completion_tokens"])
check("reserve: room over SUBMIT_ROOM + MIN_REASON keeps the sleeve's effort",
      (f.bodies[1]["max_completion_tokens"] >= agent.SUBMIT_ROOM + agent.MIN_REASON) == (f.bodies[1]["reasoning_effort"] == 0.99),
      (f.bodies[1]["max_completion_tokens"], f.bodies[1]["reasoning_effort"]))
(call, trace, *_), f = go([rep([("tape", {})], pt=CTX - 2000, ct=600), rep([("submit", SUB)])])   # dropped: room 2,000 after
check("reserve: under SUBMIT_ROOM + MIN_REASON the forced turn is plain", f.bodies[1]["max_completion_tokens"] < agent.SUBMIT_ROOM + agent.MIN_REASON
      and f.bodies[1]["reasoning_effort"] == inkling.FORCED_PLAIN_EFFORT and stats(trace)["forced_plain"] == 1,
      (f.bodies[1]["max_completion_tokens"], f.bodies[1]["reasoning_effort"]))

# 8b. A forced reply cut at max_tokens: the next forced try is plain
(call, trace, *_), f = go([rep([("news", {})]), rep([("news", {})]), rep([("news", {})]), rep([("submit", SUB)], finish="length"),
                           rep([("submit", SUB)])], max_calls=1)
check("cut forced reply after the limit: first forced try full, the next plain (one strike only)",
      f.bodies[3]["tool_choice"] == inkling.SUBMIT_CHOICE and f.bodies[3]["reasoning_effort"] == 0.99
      and f.bodies[4]["reasoning_effort"] == inkling.FORCED_PLAIN_EFFORT and call == SUB and stats(trace)["forced_plain"] == 1,
      (f.bodies[3]["reasoning_effort"], f.bodies[4]["reasoning_effort"]))

# 9. A free tool-call turn that leaves under SUBMIT_ROOM is dropped and not run
(call, trace, *_), f = go([rep([("tape", {})], pt=CTX - 5000, ct=4000), rep([("submit", SUB)])])
check("dropped: not run, submit opened", RAN == [] and trace[0].get("dropped") == ["tape"]
      and f.bodies[1]["tool_choice"] == inkling.SUBMIT_CHOICE and len(f.bodies[1]["messages"]) == 2, trace[0])

# 10. A call whose result would leave under RESERVE is not run, with the not-run line
(call, trace, *_), f = go([rep([("tape", {})], pt=CTX - 3700 - 100 - inkling.FRAME, ct=100), rep([("submit", SUB)])], final_line=True)
r0 = trace[0].get("result", "")
check("not run: line starts with live.NOT_RUN's text", r0.startswith("context nearly full (") and "no research call is run" in r0
      and r0.endswith(agent.final_words(False)) and RAN == [], r0)

# 11. A forced reply calling another tool is a malformed turn
(call, trace, *_), f = go([rep(content="x"), rep(content="y"), rep([("tape", {})]), rep([("submit", SUB)])])
check("forced non-submit: malformed, tool not run", RAN == [] and "forced turn without a submit" in trace[2].get("error", ""), trace[2])
check("forced non-submit: three strikes, no decision", call is None)

# 12. Cut at max_tokens is malformed; a text decision is taken
(call, trace, *_), f = go([rep([("tape", {})], finish="length"), rep(content=json.dumps(SUB))])
check("length: malformed", "cut at max_tokens" in trace[0].get("error", ""))
check("text decision taken", call == SUB)

# 13. Retries: a 503, a 524, a body that is not JSON and a cut transfer are retried; a third failure raises
(call, trace, *_), f = go([inkling.HttpError(503, "busy"), rep([("submit", SUB)])])
check("retry: one 503 retried", call == SUB and "retried" in trace[0].get("error", ""), trace[0])
check("retry: the retried attempt's UTC time recorded", trace[0].get("utc", "").endswith("Z"), trace[0])
(call, trace, *_), f = go([inkling.HttpError(524, "origin timeout"), inkling.BadBody("html"), rep([("submit", SUB)])])
check("retry: 524 and a non-JSON body retried", call == SUB and len(f.bodies) == 3)
(call, trace, *_), f = go([http.client.IncompleteRead(b"x"), rep([("submit", SUB)])])
check("retry: a cut transfer retried", call == SUB)
try:
    go([inkling.HttpError(503, "a"), inkling.HttpError(503, "b"), inkling.HttpError(503, "c")])
    check("retry: three failures raise", False)
except inkling.HttpError as e:
    end = e.inkling_trace[-1] if e.inkling_trace else {}
    check("retry: three failures raise, the trace kept on the exception", e.status == 503 and len(e.inkling_trace) == 3
          and all("retried" in t.get("error", "") for t in e.inkling_trace[:2]), e.inkling_trace)
    check("retry: the attempt that ended the look has its UTC time and the look's counts",
          end.get("error", "").startswith("look ended: HttpError") and utc_now(end.get("utc"))
          and end.get("inkling", {}).get("attempts") == 3 and end["inkling"].get("lost_attempts") == 3
          and end["inkling"].get("model") == inkling.MODEL, end)
try:
    go([TimeoutError(), TimeoutError()])
    check("retry: a second timeout raises", False)
except TimeoutError:
    check("retry: a second timeout raises", True)

# 13b. A 400 raises with what the look had done; a 400 naming the context is one malformed turn
try:
    go([rep([("tape", {})], pt=20000, ct=500), inkling.HttpError(400, "unknown model")])
    check("400 raises", False)
except inkling.HttpError as e:
    check("400 raises, with the look's trace and cost", e.status == 400 and e.inkling_cost == (20000, 500)
          and e.inkling_trace and e.inkling_trace[0].get("tool") == "tape", (e.inkling_cost, e.inkling_trace))
(call, trace, *_), f = go([inkling.HttpError(400, "This model's maximum context length is 131072 tokens"), rep([("submit", SUB)])])
check("400 context: malformed turn, then the decision", call == SUB and trace[0].get("http") == 400, trace[0])

# 14. Seeds, effort names, ids, context
(call, trace, *_), f = go([rep([("tape", {})]), rep([("submit", SUB)])], seed=7)
check("seed per step", f.bodies[0]["seed"] == 700 and f.bodies[1]["seed"] == 701)
check("effort names", inkling.effort_value("max") == 0.99 and inkling.effort_value("high") == 0.9 and inkling.effort_value(0.5) == 0.5)
for bad_e in (1.0, -0.1, "ultra", True, float("nan")):
    try:
        inkling.effort_value(bad_e)
        check(f"effort {bad_e!r} refused", False)
    except ValueError:
        check(f"effort {bad_e!r} refused", True)
check("is_inkling", inkling.is_inkling("thinkingmachines/Inkling-2-Small") and inkling.is_inkling("thinkingmachines/Inkling-2-Small:peft:262144")
      and not inkling.is_inkling("base") and not inkling.is_inkling("claude-opus-5-5") and not inkling.is_inkling(None))
check("context_of", inkling.context_of({"context": 65536}) == 65536 and inkling.context_of({}) == 131072
      and inkling.context_of({"context": "131072"}) == 131072 and inkling.context_of({"context": 10 ** 9}) == 131072)
(call, trace, *_), f = go([rep([("submit", SUB)], pt=60000)], context=65536)
check("a smaller context bounds the room", f.bodies[0]["max_completion_tokens"] == agent.MAX_OUT and stats(trace)["context"] == 65536)

# 15. check_orders outside the research count, counted against its own limit
def checker(args):
    return "check result", True, {"raw": 1}
checker.own_limit = 3.0
(call, trace, *_), f = go([rep([("check_orders", {"place": []})]), rep([("submit", SUB)])], checker=checker, max_calls=15)
m = f.bodies[1]["messages"]
check("check_orders: offered and counted", any(t["function"]["name"] == "check_orders" for t in f.bodies[0]["tools"])
      and m[-1]["content"].endswith(f"(order check 1 of {checks.MAX_CHECKS})") and RAN == [], m[-1]["content"])

# 16. live.py: the sleeve keys
import live
meta = {"model": "thinkingmachines/Inkling-2-Small", "final_line": True, "context": 131072}
check("live.effort: max for Inkling", live.effort(meta) == "max" and inkling.effort_value(live.effort(meta)) == 0.99)
check("live.forced_mode_of: None for Inkling", live.forced_mode_of(meta) is None)
check("live.effort: gpt-oss unchanged", live.effort({"model": "base"}) == "high" and live.forced_mode_of({"model": "base", "forced_mode": "reasoning"}) == "reasoning")
check("live.final_line_of: as the sleeve says", live.final_line_of(meta) is True)
check("live.context_of: Inkling", live.context_of(meta) == 131072 and live.context_of({"model": "thinkingmachines/Inkling-2-Small"}) == 131072)
check("live.context_of / base_model_of: gpt-oss as before without the keys",
      live.context_of({"model": "base"}) == agent.CONTEXT and live.base_model_of({"model": "base"}) == live.BASE
      and live.context_of({"model": "tinker://x/sampler_weights/y"}) == agent.CONTEXT and live.base_model_of({"model": "tinker://x"}) is None
      and live.context_of({"model": "claude-fable-5-2"}) is None)
v = {"model": "base", "base_model": "openai/gpt-oss-120b:peft:131072", "context": 131072}
check("live: the 131,072 variant", live.base_model_of(v) == "openai/gpt-oss-120b:peft:131072" and live.context_of(v) == 131072
      and live.context_of(dict(v, context=None)) == 131072 and live.context_of(dict(v, context=200000)) == 131072
      and live.context_of(dict(v, context=65536)) == 65536)
check("live: a context without a variant changes nothing, and says so", live.context_of({"model": "base", "context": 131072}) == agent.CONTEXT
      and any("context" in m for m in live.switch_bad({"model": "base", "context": 131072})))
check("live: a base_model on a checkpoint is ignored, and says so", live.base_model_of({"model": "tinker://x", "base_model": "openai/gpt-oss-120b:peft:131072"}) is None
      and any("base_model" in m for m in live.switch_bad({"model": "tinker://x", "base_model": "openai/gpt-oss-120b:peft:131072"})))
check("live: a base_model that is not gpt-oss is ignored", live.base_model_of({"model": "base", "base_model": "Qwen/Qwen3"}) == live.BASE)
check("live: switch_bad quiet on the planned sleeves", live.switch_bad(meta) == [] and live.switch_bad(v) == [])

# 17. agent.gptoss_look's per-look context (the 131,072 variant): a long prompt gets the room of the larger context
class _Seq:
    def __init__(self, toks):
        self.tokens, self.stop_reason = toks, "stop"


class _SC:
    def __init__(self):
        self.params = []

    def sample(self, model_input, n, params):
        self.params.append(params)
        tok = trader.tokenizer()
        toks = list(tok.encode("<|channel|>commentary to=functions.submit <|constrain|>json<|message|>" + json.dumps(SUB),
                               add_special_tokens=False)) + [trader.CALL]
        res = type("R", (), {"sequences": [_Seq(toks)]})()
        return type("F", (), {"result": lambda self, timeout=None: res})()


big = "word " * 20000                        # about 20,000 tokens of system text
for ctxv, want in ((None, agent.CONTEXT), (131072, 131072)):
    sc = _SC()
    call, trace, prefill, sampled = agent.gptoss_look(sc, None, big, "STATE", "2026-11-23", SPECS, "high", context=ctxv)
    mt = sc.params[0].max_tokens
    plen = prefill
    check(f"gptoss_look context={ctxv}: decision, and max_tokens is min(MAX_OUT, {want} - prompt)",
          call == SUB and mt == min(agent.MAX_OUT, want - plen), (call is not None, mt, plen))
try:
    agent.gptoss_look(_SC(), None, "S", "STATE", "2026-11-23", SPECS, "high", context="131072")
    check("gptoss_look: a context that is not a number raises", False)
except ValueError:
    check("gptoss_look: a context that is not a number raises", True)

# 18. The second review (20 Nov)
import urllib.error
# 18a. A cut result followed by many calls in one reply still leaves the decision SUBMIT_ROOM
long_tape = lambda gw, name, args, **_: ("line of tape text\n" * 3000) if name == "tape" else f"result of {name}"
(call, trace, *_), f = go([rep([("tape", {})] + [("quote", {"symbols": [f"S{k}"]}) for k in range(15)], pt=CTX - 6500, ct=100),
                           rep([("submit", SUB)])], run_tool=long_tape, final_line=True, max_calls=15)
mct = f.bodies[1]["max_completion_tokens"]
check("many calls after a cut result: the decision keeps SUBMIT_ROOM", mct >= agent.SUBMIT_ROOM and call == SUB, mct)
m = f.bodies[1]["messages"]
check("many calls: every call id answered", len([x for x in m if x["role"] == "tool"]) == 16
      and [x["tool_call_id"] for x in m if x["role"] == "tool"] == [c["id"] for c in m[2]["tool_calls"]])

# 18b. A 400 naming the context near the end leaves room for another try; an unrelated 400 raises
(call, trace, *_), f = go([rep([("tape", {})], pt=CTX - 3700 - 100 - inkling.FRAME, ct=100),
                           inkling.HttpError(400, "This model's maximum context length is 131072 tokens"), rep([("submit", SUB)])])
check("400 context at the end: another try runs, decision", call == SUB and len(f.bodies) == 3
      and f.bodies[2]["max_completion_tokens"] >= agent.MIN_SUBMIT, [b["max_completion_tokens"] for b in f.bodies])
try:
    go([inkling.HttpError(400, "Invalid tools: Expected maximum length 128")])
    check("400 not about the context raises", False)
except inkling.HttpError as e:
    check("400 not about the context raises", e.status == 400)

# 18c. A valid submit beside a malformed call is the decision
(call, trace, *_), f = go([rep([("think", {"x": 1}), ("submit", SUB)])])
check("valid submit beside an undeclared call: the decision", call == SUB and trace[0].get("not_run") == ["think"], trace[0])
(call, trace, *_), f = go([rep([("quote", "{bad"), ("submit", SUB)])])
check("valid submit beside bad JSON: the decision", call == SUB)
(call, trace, *_), f = go([rep([("think", {"x": 1}), ("tape", {})]), rep([("submit", SUB)])])
check("no submit and a malformed call: one malformed turn, nothing run", RAN == [] and "undeclared" in trace[0].get("error", ""), trace[0])

# 18d. The missing-price re-ask answers the submit itself when ids repeat, with arguments as a string
r = rep([("quote", {"symbols": ["A"]}), ("submit", lmt)], ids=["x", "x"])
(call, trace, *_), f = go([r, rep([("submit", SUB)])])
m = f.bodies[1]["messages"]
check("re-ask with repeated ids: the history holds the submit and answers it", m[-2]["tool_calls"][0]["function"]["name"] == "submit"
      and m[-1]["tool_call_id"] == m[-2]["tool_calls"][0]["id"] and isinstance(m[-2]["tool_calls"][0]["function"]["arguments"], str)
      and call == SUB, m[-2:])
r = rep([("submit", lmt)])
r["choices"][0]["message"]["tool_calls"][0].pop("id")
(call, trace, *_), f = go([r, rep([("submit", SUB)])])
m = f.bodies[1]["messages"]
check("re-ask of a submit call with no id: answered as a call, not as text", m[-1]["role"] == "tool" and m[-2]["tool_calls"]
      and "answered_as" not in trace[0], m[-2:])

# 18e. A connect timeout wrapped in URLError counts as a timeout
try:
    go([urllib.error.URLError(TimeoutError("timed out")), urllib.error.URLError(TimeoutError("timed out")), rep([("submit", SUB)])])
    check("URLError(TimeoutError) twice raises", False)
except urllib.error.URLError:
    check("URLError(TimeoutError) twice raises", True)
(call, trace, *_), f = go([urllib.error.URLError(ConnectionRefusedError("refused")), rep([("submit", SUB)])])
check("URLError of a refused connection is retried", call == SUB)
check("URLError of a refused connection: the retried attempt has its UTC time", "URLError" in trace[0].get("error", "")
      and utc_now(trace[0].get("utc")), trace[0])

# 18f. live.py: the variant's form, the Inkling forced_mode note, the lanes, research_summary
check("variant_context: only <BASE>:peft:<n> in range", live.variant_context("openai/gpt-oss-120b:peft:131072") == 131072
      and live.variant_context("openai/gpt-oss-120b:peft:131k") is None and live.variant_context("openai/gpt-oss-120b:peft:2000000") is None
      and live.variant_context("openai/gpt-oss-120b:131072") is None and live.variant_context("openai/gpt-oss-120b") is None)
check("base_model_of: a mistyped variant falls back to BASE and switch_bad says so",
      live.base_model_of({"model": "base", "base_model": "openai/gpt-oss-120b:peft:131k"}) == live.BASE
      and any("base_model" in x for x in live.switch_bad({"model": "base", "base_model": "openai/gpt-oss-120b:peft:131k"})))
check("switch_bad: forced_mode on an Inkling sleeve is reported",
      any("Inkling" in x for x in live.switch_bad({"model": "thinkingmachines/Inkling-2-Small", "forced_mode": "reasoning"})))
check("GPTOSS_LANES covers 16 sleeves", live.GPTOSS_LANES >= 16)
summ = live.research_summary([{"tool": "news", "args": {"symbol": "A"}, "result": "x"},
                              {"tool": "news", "args": {"symbol": "B"}, "result": "line", "not_run": "after the line that opened the submit in this reply"},
                              {"tool": "submit", "args": SUB, "not_run": ["quote"]}], SUB, [])
check("research_summary: a call not run after the opening line is not research", summ["research_calls"] == 1, summ)

# 19. The Researcher's read (20 Nov)
import ml_tools
import tools as _tools
# 19a. Inkling reads the tool words gpt-oss reads, no more
live_specs = list(_tools.SPECS) + [ml_tools.module("ML4").SPEC, checks.CHECK_SPEC, trader.TOOL_LIVE, trader.TOOL_LIVE_PROBS]
diff = [x["name"] for x in live_specs if sorted(inkling.harmony_words(x)) != sorted(inkling.json_words(inkling.oai_tools([x])[0]))]
check("tool words: the same descriptive words as gpt-oss's renderer, every live spec", diff == [], diff)
from tinker_cookbook.renderers import gpt_oss as _g
rendered = "\n".join(_g._format_tool_definition(x) for x in agent.harmony_specs(live_specs))
missing = [w for x in live_specs for w in inkling.json_words(inkling.oai_tools([x])[0])
           if isinstance(json.loads(w), str) and json.loads(w).split("\n")[0] not in rendered]
check("tool words: every word Inkling reads is in gpt-oss's rendered tools", missing == [], missing[:3])
sub = inkling.oai_tools([trader.TOOL_LIVE])[0]["function"]["parameters"]
items = sub["properties"]["place"]["items"]
check("tool words: no description under place[] items, types and enums kept",
      not any("description" in v for v in items.get("properties", {}).values())
      and items["properties"]["type"].get("enum") == trader.TOOL_LIVE["parameters"]["properties"]["place"]["items"]["properties"]["type"].get("enum"))
odd = {"name": "x", "description": "d", "parameters": {"type": "object", "properties": {"o": {"type": "object", "description": "kept",
       "properties": {"title": {"type": "string", "description": "gone"}, "description": {"type": "number"}}}}}}
po = inkling.oai_tools([odd])[0]["function"]["parameters"]["properties"]["o"]
check("tool words: properties named title/description survive, their own words do not",
      po["description"] == "kept" and set(po["properties"]) == {"title", "description"} and "description" not in po["properties"]["title"], po)
check("words_sha: equal for the same specs, different when a word changes", inkling.words_sha(live_specs) == inkling.words_sha(list(live_specs))
      and inkling.words_sha([dict(trader.TOOL_LIVE, description="other")]) != inkling.words_sha([trader.TOOL_LIVE]))

# 19b. Attempts and lost attempts; a 400 about tokens is an overflow
(call, trace, *_), f = go([inkling.HttpError(503, "busy"), e422(), rep([("submit", SUB)])])
s19 = stats(trace)
check("attempts counted, lost ones too", s19["attempts"] == 3 and s19["lost_attempts"] == 2 and s19["requests"] == 2, s19)
(call, trace, *_), f = go([inkling.HttpError(400, "prompt + max tokens exceeds the limit"), rep([("submit", SUB)])])
check("a 400 naming tokens is an overflow: one malformed turn, then the decision", call == SUB and trace[0].get("http") == 400)

# 19c. switch_bad for an Inkling entry copied from its twin
ink = {"model": "thinkingmachines/Inkling-2-Small", "effort": "high"}
w = live.switch_bad(ink)
check("switch_bad: an Inkling sleeve under max effort, and without final_line, is reported",
      any("under its maximum" in m for m in w) and any("final_line" in m for m in w), w)
check("switch_bad: quiet on a right Inkling entry", live.switch_bad({"model": "thinkingmachines/Inkling-2-Small", "effort": "max",
                                                                      "final_line": True, "context": 131072}) == [])

# 19d. The record's words hashes
check("words_of: hashes of the system text and the tools offered", live.words_of(("S", live_specs))[0] and
      live.words_of(("S", live_specs))[1] == inkling.words_sha(live_specs) and live.words_of(None) == (None, None))

# 19e. The smoke's replay: framework lines not served, counters and cut lines taken off
import inkling_smoke as sm
rp = sm.Replay([{"tool": "news", "args": {"symbol": "A"}, "result": "story\n(research call 3 of 15)"},
                {"tool": "news", "args": {"symbol": "B"}, "result": "research limit reached (15 calls): call submit now"},
                {"tool": "tape", "args": {}, "result": "context nearly full (3928 tokens left); no research call is run"},
                {"tool": "quote", "args": {"symbols": ["A"]}, "result": "row\n" + agent.CUT_LINE + "\n(research call 4 of 15)"},
                {"tool": "check_orders", "args": {}, "result": "checked\n(order check 1 of 3)"},
                {"tool": "check_orders", "args": {}, "result": checks.cap_text(3)}])
check("smoke replay: a stored result without its counter", rp.answer("news", {"symbol": "A"}) == ("story", True))
check("smoke replay: a limit line is never served", rp.answer("news", {"symbol": "B"}) == (sm.NONE_LINE, False))
check("smoke replay: a not-run line is never served", rp.answer("tape", {}) == (sm.NONE_LINE, False))
check("smoke replay: the cut line and counter off", rp.answer("quote", {"symbols": ["A"]}) == ("row", True))
check("smoke replay: a check without its counter, the cap text never", rp.answer("check_orders", {}) == ("checked", True)
      and rp.answer("check_orders", {}) == (sm.NONE_LINE, False))

# 20. A 400 that names the server's context limit: the look counts against it from then on
(call, trace, *_), f = go([rep([("tape", {})], pt=60000, ct=500),
                           inkling.HttpError(400, "This model's maximum context length is 65536 tokens. However, you requested 76590"),
                           rep([("submit", SUB)])], context=131072)
check("a 400 naming 65,536: the next request fits it", call == SUB and f.bodies[2]["max_completion_tokens"] <= 65536 - inkling.MARGIN - 60000,
      [b["max_completion_tokens"] for b in f.bodies])

print("inkling_test.py:", "ALL OK" if not FAILS else f"FAILED ({len(FAILS)})")
sys.exit(1 if FAILS else 0)
