"""Two changes of Mon 16 Nov 2026 evening (Clayton's answers): the missing-price re-ask (tools.missing_prices and
missing_text; agent.gptoss_look; mcp_server.submit) and IBKR's margin warning to the more levered sleeves
(trader.margin_warning in the whole-account line). Offline: scripted turns on the real tokenizer, the MCP server in a
subprocess with temporary files, fixed account figures. Nothing under runs/ is written.

    python reask_margin_test.py
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import agent  # noqa: E402
import checks  # noqa: E402
import tools  # noqa: E402
import trader  # noqa: E402

ok = []


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail and not good else ""))


# ================================================================ 1. which orders lack a price field
O = lambda **k: dict({"symbol": "AMD", "action": "BUY", "quantity": 100, "tif": "GTC"}, **k)
T = [(O(type="STP", limit=506), [(1, "BUY 100 AMD STP", ["stop"], ["limit"])]),       # cand2 15:30 Mon
     (O(type="STP", stop=506), []), (O(type="LMT"), [(1, "BUY 100 AMD LMT", ["limit"], [])]), (O(type="LMT", limit=5), []),
     (O(type="STP LMT", stop=5), [(1, "BUY 100 AMD STP LMT", ["limit"], [])]),
     (O(type="stp  lmt", limit=5), [(1, "BUY 100 AMD STP LMT", ["stop"], [])]), (O(type="STP LMT", stop=5, limit=4), []),
     (O(type="TRAIL"), [(1, "BUY 100 AMD TRAIL", ["trail_percent"], [])]), (O(type="TRAIL", trail_percent=2), []),
     (O(type="LOC"), [(1, "BUY 100 AMD LOC", ["limit"], [])]), (O(type="LOC", limit="12.5"), []),
     (O(type="MKT"), []), (O(type="MOO"), []), (O(type="MOC"), []), (O(type="STP", stop=None), [(1, "BUY 100 AMD STP", ["stop"], [])]),
     (O(type="STP", stop=""), [(1, "BUY 100 AMD STP", ["stop"], [])]), (O(type="STP", stop="abc"), []),
     (O(type="STP", stop=True), [(1, "BUY 100 AMD STP", ["stop"], [])]), (O(type="XYZ"), []), (O(), []),
     (O(type="STP_LMT", limit=5), [(1, "BUY 100 AMD STP LMT", ["stop"], [])])]   # "_" read as a space, as clean_order
wrong = [(o, tools.missing_prices([o]), want) for o, want in T if tools.missing_prices([o]) != want]
check(f"missing_prices on {len(T)} orders: LMT/LOC need limit, STP stop, STP LMT both, TRAIL trail_percent; MKT, MOO, MOC "
      "none; case, spaces and underscores in the type read as clean_order reads them; a non-number is not 'missing' here; an unused field given is named",
      not wrong, wrong[:3])
check("missing_prices: not a list, or entries that are not objects, give nothing; indices count from 1",
      tools.missing_prices(None) == [] and tools.missing_prices("x") == [] and tools.missing_prices([1, O(type="STP")])
      == [(2, "BUY 100 AMD STP", ["stop"], [])])
txt = tools.missing_text(tools.missing_prices([O(type="STP", limit=506), O(type="MKT"), O(symbol="MU", type="LMT")]))
check("missing_text names each order, its missing field and the unused one, and says the next submit is sent as written",
      txt == ('submit not accepted yet: order 1 (BUY 100 AMD STP) has no "stop" (its stop (trigger) price); it gives "limit", '
              'which its type does not use; order 3 (BUY 100 MU LMT) has no "limit" (its limit price). IBKR refuses an order '
              'of that type without it. Call submit again with your whole decision (every order and cancel); this is asked '
              'once a look, and the next submit is sent as written.'), txt)

# ================================================================ 2. agent.gptoss_look on scripted turns
tok = trader.tokenizer()
enc = lambda s: list(tok.encode(s, add_special_tokens=False))
OPEN, A = agent.SUBMIT_OPEN(), agent.ANALYSIS_OPEN()
FILLER = enc(" the")[0]
BASE_P = []


class PromptR:
    def __init__(self, *a, **k):
        pass

    def create_conversation_prefix_with_tools(self, specs, system):
        return []

    def build_generation_prompt(self, msgs):
        return types.SimpleNamespace(to_ints=lambda: list(BASE_P))


saved_mod = sys.modules.get("tinker_cookbook.renderers.gpt_oss")
pr = types.ModuleType("tinker_cookbook.renderers.gpt_oss")
pr.GptOssRenderer = PromptR
sys.modules["tinker_cookbook.renderers.gpt_oss"] = pr
SPEC = [{"name": "quote", "description": "q", "parameters": {"type": "object", "properties": {"symbols": {"type": "array", "items": {"type": "string"}}}}}]
PAD = enc(" step" * 400)[:400]
tail = lambda name, args: enc(f"<|end|><|start|>assistant<|channel|>commentary to=functions.{name} <|constrain|>json<|message|>"
                              + json.dumps(args) + "<|call|>")
Q = A + PAD[:150] + tail("quote", {"symbols": ["SPY"]})
P = enc("<|channel|>analysis<|message|>Reasoning.<|end|><|start|>assistant<|channel|>final<|message|>I would hold.<|return|>")
BAD_ORDERS = {"place": [O(type="STP", limit=506), O(symbol="MU", type="MKT", action="SELL")], "cancel": [], "notes": "first"}
GOOD_ORDERS = {"place": [O(type="STP", stop=506), O(symbol="MU", type="MKT", action="SELL")], "cancel": [], "notes": "second"}
FREE_BAD = A + PAD[:60] + tail("submit", BAD_ORDERS)
FREE_GOOD = A + PAD[:60] + tail("submit", GOOD_ORDERS)
FORCED_BAD = enc(json.dumps(BAD_ORDERS) + "<|call|>")
FORCED_GOOD = enc(json.dumps(GOOD_ORDERS) + "<|call|>")


def fake(free, forced, rec):
    its, fts = iter(free), iter(forced)

    def sample_once(sc, renew, model_input, params, trace, step):
        ints = list(model_input.to_ints())
        kind = "forced" if ints[-len(OPEN):] == OPEN else "free"
        t = next(fts, FORCED_GOOD) if kind == "forced" else next(its, P)
        t = t[:params.max_tokens]
        rec.append({"prompt": ints, "max": params.max_tokens, "kind": kind, "step": step})
        stop = "length" if len(t) >= params.max_tokens and t[-1] not in params.stop else "stop"
        return types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=t, stop_reason=stop)]), sc
    return sample_once


def run(mod, n, free, forced=(), **kw):
    BASE_P[:] = [FILLER] * n
    rec = []
    tools.run = lambda gw, name, args, **k: "\n".join(["line of text"] * 100)
    mod.sample_once = fake(list(free), list(forced), rec)
    call, trace, prefill, sampled = mod.gptoss_look(None, None, "s", "s", "2026-11-17", SPEC, "high", seed=5,
                                                    forced_mode="plain", final_line=False, **kw)
    return call, trace, rec


def committed(rev):
    src = subprocess.run(["git", "show", f"{rev}:agent.py"], capture_output=True, text=True, encoding="utf-8", cwd=ROOT).stdout
    f = Path(tempfile.mkdtemp()) / f"agent_{rev}.py"
    f.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"agent_{rev}", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


real_run, real_sample = tools.run, agent.sample_once
c, trace, rec = run(agent, 6000, [Q, FREE_BAD, FREE_GOOD])
rk = [t for t in trace if t.get("reask")]
check("a free submit with an STP that has no stop: answered once with missing_text (trace 'reask', no 'tool'), and the next "
      "free submit is the decision", c == GOOD_ORDERS and len(rk) == 1 and "tool" not in rk[0]
      and rk[0]["result"] == tools.missing_text(tools.missing_prices(BAD_ORDERS["place"])) and rk[0]["args"]["notes"] == "first"
      and rk[0]["missing"] == [{"order": 1, "fields": ["stop"]}] and trace[-1].get("tool") == "submit", (c, trace[-1]))
check("the model reads the answer: the next request's prompt ends with the submit call and that tool result",
      rec[-1]["prompt"][-len(agent.tool_result_tokens("submit", rk[0]["result"])):] == agent.tool_result_tokens("submit", rk[0]["result"]))
c, trace, rec = run(agent, 6000, [FREE_BAD, FREE_BAD])
check("asked once a look: a second incomplete submit is taken as written", c == BAD_ORDERS and len([t for t in trace if t.get("reask")]) == 1)
c, trace, rec = run(agent, 6000, [Q] * 14, forced=[FORCED_BAD, FORCED_GOOD])
check("an opened submit (after the research limit) that lacks a stop is asked once; the next turn is the opened submit again",
      c == GOOD_ORDERS and [r["kind"] for r in rec[-2:]] == ["forced", "forced"] and len([t for t in trace if t.get("reask")]) == 1)
c, trace, rec = run(agent, 6000, [FREE_GOOD])
check("a complete submit is taken at once (no re-ask)", c == GOOD_ORDERS and not [t for t in trace if t.get("reask")])
# room: a free turn runs only with RESERVE (2,500) of room, so the answer always fits after a free submit; an opened submit
# (under RESERVE) is asked only while the answer leaves SUBMIT_ROOM after the submit's prefix, else taken as written
asked, taken, odd = 0, 0, []
for n in range(30000, 32400, 7):
    c, trace, rec = run(agent, n, [], forced=[FORCED_BAD, FORCED_GOOD])
    r = [t for t in trace if t.get("reask")]
    if c == GOOD_ORDERS and len(r) == 1:
        asked += 1
        room_after = agent.CONTEXT - len(rec[-1]["prompt"])          # the re-asked turn's room; its prompt holds the prefix
        if room_after < agent.SUBMIT_ROOM:
            odd.append((n, "asked with too little room", room_after))
    elif c == BAD_ORDERS and not r:
        taken += 1
    elif c is not None or n < 32100:
        odd.append((n, c and c.get("notes"), len(r)))
check(f"near the end of the context: an opened incomplete submit is asked while the answer leaves SUBMIT_ROOM ({asked} looks), "
      f"and taken as written when it would not ({taken} looks); never asked without that room", asked > 20 and taken > 20 and not odd,
      (asked, taken, odd[:3]))
OLD = committed("88d1be78")
same = True
for free, forced in (([Q, FREE_GOOD], []), ([Q] * 14, [FORCED_GOOD]), ([P, Q, FREE_GOOD], []), ([Q] * 13, [])):
    same = same and run(OLD, 6000, free, forced) == run(agent, 6000, free, forced)
check("looks with complete submits are the same as at 88d1be78: requests, trace, decision", same)
worst = []
for n in (1, 12, 15, tools.MAX_CALLS_TOP):
    BADF = enc("not json<|call|>")
    c, trace, rec = run(agent, 4000, [Q] * (n + 2), forced=[BADF, BADF, FORCED_BAD, FORCED_GOOD], max_calls=n)
    steps = max(t.get("step", -1) for t in trace) + 1
    if c != GOOD_ORDERS or steps != n + 6:
        worst.append((n, steps, c is not None, trace[-1]))
check("the longest look with a re-ask (limit, told, opened, two malformed, the re-ask, the decision) decides on the range's "
      f"last step for limits 1 to {tools.MAX_CALLS_TOP}", not worst, worst)
# the held-back submit is the decision when no later one comes (the review of 16 Nov evening)
BADF = enc("not json<|call|>")
c, trace, rec = run(agent, 6000, [P, P], forced=[FORCED_BAD, BADF])
check("two malformed turns, then an opened incomplete submit (re-asked), then a malformed one: the re-asked submit is the "
      "decision, and the trace says so", c == BAD_ORDERS and trace[-1].get("fallback") == "reask" and trace[-1].get("tool") == "submit",
      trace[-2:])
c, trace, rec = run(agent, 6000, [FREE_BAD, P, P], forced=[BADF])      # the third is the opened submit after the second
check("a free submit re-asked, then three malformed turns: the re-asked submit is the decision", c == BAD_ORDERS
      and trace[-1].get("fallback") == "reask")
calls_n = [0]


def boom(sc, renew, model_input, params, trace, step):
    calls_n[0] += 1
    if calls_n[0] == 1:
        return types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=FREE_BAD, stop_reason="stop")]), sc
    raise TimeoutError("Tinker did not answer")


BASE_P[:] = [FILLER] * 6000
agent.sample_once = boom
c, trace, _, _ = agent.gptoss_look(None, None, "s", "s", "2026-11-17", SPEC, "high", seed=5, forced_mode="plain", final_line=False)
raised = False
calls_n[0] = 5                                              # every sample raises: no re-ask, so the error leaves the look as before
try:
    agent.gptoss_look(None, None, "s", "s", "2026-11-17", SPEC, "high", seed=5, forced_mode="plain", final_line=False)
except TimeoutError:
    raised = True
check("a sample that fails after a re-ask: the re-asked submit is the decision; with no re-ask the error leaves the look as "
      "before", c == BAD_ORDERS and trace[-1].get("fallback") == "reask" and "sample failed after the re-ask" in str(trace[-2])
      and raised, (c, trace[-2:]))
tools.run, agent.sample_once = real_run, real_sample
if saved_mod is None:
    sys.modules.pop("tinker_cookbook.renderers.gpt_oss", None)
else:
    sys.modules["tinker_cookbook.renderers.gpt_oss"] = saved_mod

import live  # noqa: E402

base = [{"step": 0, "tool": "quote", "args": {"symbols": ["SPY"]}, "result": "x\n(research call 1 of 15)"}]
with_rk = base + [{"step": 1, "reask": "submit", "args": BAD_ORDERS, "result": "submit not accepted yet: ..."}]
check("live.research_summary counts a re-ask as nothing (no research call, no check, nothing not run)",
      live.research_summary(with_rk, GOOD_ORDERS, ["SPY"]) == live.research_summary(base, GOOD_ORDERS, ["SPY"]))

# ================================================================ 3. the Fable path: mcp_server.submit
PROBE = """
import json, sys, asyncio
sys.path.insert(0, sys.argv[1])
import mcp_server
bad = json.loads(sys.argv[2]); good = json.loads(sys.argv[3])
out = [mcp_server.submit(bad["place"], bad["cancel"], bad["notes"]), json.loads(mcp_server.SUBMIT.read_text())["notes"]]
out += [mcp_server.submit(bad["place"], bad["cancel"], "again"), json.loads(mcp_server.SUBMIT.read_text())["notes"]]
out += [mcp_server.submit(good["place"], good["cancel"], "third"), json.loads(mcp_server.SUBMIT.read_text())["notes"]]
print(json.dumps(out))
"""
d = Path(tempfile.mkdtemp())
env = dict(os.environ, TRADER_SUBMIT_FILE=str(d / "submit.json"), TRADER_LOG_FILE=str(d / "research.jsonl"), TRADER_MAX_GROSS="none")
p = subprocess.run([sys.executable, "-c", PROBE, str(ROOT), json.dumps(BAD_ORDERS), json.dumps(GOOD_ORDERS)],
                   capture_output=True, text=True, encoding="utf-8", env=env, cwd=d)
res = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 and p.stdout.strip() else ["error", p.stderr[-500:]]
logged = [json.loads(l) for l in (d / "research.jsonl").read_text(encoding="utf-8").splitlines()] if (d / "research.jsonl").exists() else []
check("Fable's submit: the first incomplete one is answered with missing_text and written as the fallback decision; the next "
      "replaces it, once; a third is refused; the re-ask is logged as 'reask' (no 'tool')",
      res[:2] == [tools.missing_text(tools.missing_prices(BAD_ORDERS["place"])), "first"] and res[2].startswith("submitted:")
      and res[3] == "again" and res[4].startswith("already submitted") and res[5] == "again" and len(logged) == 1
      and logged[0].get("reask") == "submit" and "tool" not in logged[0], (res, logged))

# ================================================================ 4. IBKR's margin warning in the whole-account line
ACC = {"NetLiquidation": "1385313", "EquityWithLoanValue": "1384000", "InitMarginReq": "1352582", "ExcessLiquidity": "39597",
       "Cushion": "0.029", "SMA": "-832613", "GrossPositionValue": "4700872", "RegTMargin": "2350436", "read_time": "2026-11-16T20:30:24Z"}
st = lambda value, gross: {"value": value, "positions": [{"symbol": "X", "quantity": gross / 10, "close": 10.0}]}
info = {"account": ACC, "sleeves": 10, "total": 720000.0}
hi = trader.account_line(info, st(90000.0, 90000.0 * 14.4))       # base-90k-b at 15:30
lo_ = trader.account_line(info, st(30000.0, 30000.0 * 1.27))      # fable at 15:30
WARN = (" IBKR margin warning (SMA $-832,613; excess liquidity $39,597): IBKR closes positions in this account, choosing which "
        "itself, when excess liquidity is below zero, and near the close (from 15:50 ET on a full trading day) when SMA is below "
        "zero; it did so at 15:51 ET on Mon 16 Nov 2026, in every sleeve. Your gross exposure is 14.4x its value; your "
        "account's maximum is 3.4x its net liquidation.")
check("Monday 15:30's figures: a sleeve at 14.4x (above the account's 3.4x) gets the warning at the end of its account line, "
      "word for word", hi.endswith(WARN), hi[-400:])
check("a sleeve at 1.27x (below the account's multiple) gets the line without the warning", "margin warning" not in lo_
      and lo_ == hi.replace(WARN, "").replace("value (at last close) is 6.5%", "value (at last close) is 2.2%").replace(
          "its gross exposure is 27.6% of the account's gross.", "its gross exposure is 0.8% of the account's gross.") or "margin warning" not in lo_)
safe = dict(ACC, SMA="250000", ExcessLiquidity="500000")
check("no warning when SMA is above zero and excess liquidity is at or above 10% of net liquidation, however levered the sleeve",
      "margin warning" not in trader.account_line({"account": safe, "sleeves": 10}, st(90000.0, 90000.0 * 14.4)))
xl_only = dict(ACC, SMA="250000", ExcessLiquidity="100000")       # under 10% of 1,385,313
check("excess liquidity under 10% of net liquidation alone (SMA positive) warns a sleeve above the account's multiple",
      "IBKR margin warning (SMA $250,000; excess liquidity $100,000)" in trader.account_line({"account": xl_only, "sleeves": 10},
                                                                                           st(90000.0, 90000.0 * 5)))
neg = trader.account_line(info, {"value": -500.0, "positions": [{"symbol": "X", "quantity": 100, "close": 10.0}]})
check("a sleeve whose value is at or below zero with gross gets the warning (the most levered of all)",
      "Your gross exposure is $1,000 with its value at or below zero; your account's maximum is 3.4x" in neg, neg[-300:])
check("figures not read, or no net liquidation: no warning and no error",
      "margin warning" not in trader.account_line({"account": None}, st(90000.0, 9e5))
      and "margin warning" not in trader.account_line({"account": dict(ACC, NetLiquidation="0")}, st(90000.0, 9e5)))

# ================================================================ 5. ML4.1's plain wording only for the open-specific error
import forecast4  # noqa: E402
L1, L2 = forecast4._PLAIN[0][0], forecast4._PLAIN[1][0]
t_open = "\n".join([f"as of the close of 2026-11-13; {L1}", "x", L2])
t_noon = "\n".join([f"as of the close of 2026-11-09; {L1}", "x", "market now: SPY 1.0% since the open"])
check("ML4.1: both lines reworded when the open-specific error is there; the session-ranges line alone (e.g. a bars timeout "
      "at 11:30) is left as ML4.0 wrote it", forecast4._plain(t_open) == t_open.replace(L1, forecast4._PLAIN[0][1]).replace(L2, forecast4._PLAIN[1][1])
      and forecast4._plain(t_noon) == t_noon)

print("ALL OK" if all(ok) else f"FAILED ({ok.count(False)})")
sys.exit(0 if all(ok) else 1)
