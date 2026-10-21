"""Run base gpt-oss-120b, NAV4 and NAV9 on the P1 interface they were trained on.

    python nav_runner.py replay --arms base nav4 nav9 --horizons 20 63 --every 10 --seeds 2 --tag oos1

Each episode is one Harmony conversation: the model's sampled tokens (analysis included) and the
tool results are appended verbatim, as in training. A sample must end in <|call|> (one pa_tws
call) or <|return|> (a final answer); anything else ends the episode as MALFORMED, unrepaired.
After the model stops, the evaluator advances the rest of the horizon holding inventory (Codex's
terminal valuation); outcome() also reports realized P&L and the return after liquidation.
"""

import argparse
import asyncio
import json
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path

import p1

ROOT = Path(__file__).resolve().parent
PROJECT_ID = os.environ.get("TINKER_PROJECT_ID")   # your own Tinker project
BASE = "openai/gpt-oss-120b"
MODELS = {
    "base": None,
    "nav4": "tinker://6b29aa84-9224-5771-aace-a2af9c7d0eb0:train:0/sampler_weights/nav3s-73cbd90ec9a9-final-sampler",
    "nav9": "tinker://a75cb7d0-b32a-5e1a-8d69-8cebdc7af78b:train:0/sampler_weights/mixnav-b9578d5fa6c7-final-sampler",
}
CALL, RET = 200012, 200002
MAX_OUT, CONTEXT = 8192, 32768
PRICE = {"prefill": 0.33, "sample": 0.84}   # USD per million tokens (uncached prefill: an upper bound)
_tok = None


def get_tok():
    global _tok
    if _tok is None:
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        _tok = get_tokenizer(BASE)
    return _tok


def renderer(current_date):
    from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
    return GptOssRenderer(get_tok(), use_system_prompt=True, reasoning_effort="medium", current_date=current_date)


def decision_date(session):
    """Codex's decision time: midnight New York on the calendar day after the start session."""
    return (date.fromisoformat(session) + timedelta(days=1)).isoformat()


def render_prompt(host, start):
    r = renderer(decision_date(start))
    msgs = r.create_conversation_prefix_with_tools([p1.tool_spec()], p1.system_content(host.horizon))
    msgs.append({"role": "user", "content": "Choose a strategy using this initial observation: "
                 + p1.canonical(host.initial_observation())})
    return list(r.build_generation_prompt(msgs).to_ints())


def tool_message_tokens(result):
    text = ("<|start|>functions.pa_tws to=assistant<|channel|>commentary<|message|>" + p1.canonical(result)
            + "<|end|><|start|>assistant")
    return list(get_tok().encode(text, add_special_tokens=False))


SEG = re.compile(r"^(?: to=(?P<r1>[^\s<]+))?<\|channel\|>(?P<ch>analysis|commentary|final)(?: to=(?P<r2>[^\s<]+))?"
                 r"\s*(?:<\|constrain\|>(?P<con>\w+))?<\|message\|>(?P<body>.*?)(?P<term><\|end\|>|<\|call\|>|<\|return\|>)$",
                 re.S)


def _strict_json(body):
    def pairs(kv):
        d = {}
        for k, v in kv:
            if k in d:
                raise ValueError(f"duplicate key {k}")
            d[k] = v
        return d

    def bad_constant(c):
        raise ValueError(f"non-finite {c}")
    obj = json.loads(body, object_pairs_hook=pairs, parse_constant=bad_constant)
    if not isinstance(obj, dict):
        raise ValueError("arguments must be a JSON object")
    return obj


def parse(tokens, stop_reason):
    """('TOOL', args) | ('FINAL', text) | ('MALFORMED', reason)."""
    if str(stop_reason).split(".")[-1].lower() != "stop" or not tokens or tokens[-1] not in (CALL, RET):
        return "MALFORMED", f"no terminal token (stop_reason={stop_reason})"
    if any(t in (CALL, RET) for t in tokens[:-1]):
        return "MALFORMED", "terminal token before the end"
    segs = get_tok().decode(tokens).split("<|start|>assistant")
    for k, seg in enumerate(segs):
        m = SEG.match(seg)
        if not m:
            return "MALFORMED", f"unparseable segment {k}: {seg[:120]!r}"
        if "<|" in m["body"]:
            return "MALFORMED", "special token inside a message body"
        last = k == len(segs) - 1
        rcpt = m["r1"] or m["r2"]
        if not last:
            if m["term"] != "<|end|>" or rcpt or m["ch"] == "final":
                return "MALFORMED", f"segment {k} must be analysis/commentary ending in <|end|>"
            continue
        if m["term"] == "<|call|>":
            if m["ch"] != "commentary" or rcpt != "functions.pa_tws" or m["con"] != "json":
                return "MALFORMED", f"tool call to {rcpt!r} on channel {m['ch']} (constrain={m['con']})"
            try:
                return "TOOL", _strict_json(m["body"])
            except ValueError as e:
                return "MALFORMED", f"bad JSON: {e}"
        if m["term"] == "<|return|>" and m["ch"] == "final" and not rcpt:
            return "FINAL", m["body"]
        return "MALFORMED", "last segment is neither a tool call nor a final answer"
    return "MALFORMED", "empty"


async def episode(sampler, host, start, temperature, seed, max_steps=12, full_reserve=True):
    import tinker
    prompt = render_prompt(host, start)
    steps, status, prefill, sampled = [], "STEP_LIMIT", 0, 0
    first_len = len(prompt)
    for n in range(1, max_steps + 1):
        if full_reserve and len(prompt) + MAX_OUT > CONTEXT:
            status = "CONTEXT_BUDGET"
            break
        cap = min(MAX_OUT, CONTEXT - len(prompt))
        res = await sampler.sample_async(tinker.ModelInput.from_ints(prompt), 1,
                                         tinker.SamplingParams(temperature=temperature, seed=seed, max_tokens=cap, stop=[RET, CALL]))
        seq = res.sequences[0]
        toks = list(seq.tokens)
        prefill += len(prompt)
        sampled += len(toks)
        kind, val = parse(toks, seq.stop_reason)
        text = get_tok().decode(toks)
        step = {"step": n, "session": host.session, "prompt_tokens": len(prompt), "sample_tokens": len(toks), "text": text}
        if kind == "MALFORMED":
            status = "MALFORMED"
            step["error"] = val
            steps.append(step)
            break
        if kind == "FINAL":
            status = "FINAL"
            steps.append(step)
            break
        result = host.execute(val)
        step["action"], step["result_ok"] = val, result["result"].get("ok", True) if isinstance(result["result"], dict) else True
        if not step["result_ok"]:
            step["error"] = result["result"].get("error")
        steps.append(step)
        prompt = prompt + toks + tool_message_tokens(result)
    model_sessions = host.model_advanced
    evaluator_sessions = host.terminal()
    out = p1.outcome(host)
    return {**out, "status": status, "steps": len(steps), "evaluator_sessions": evaluator_sessions,
            "model_sessions": model_sessions, "first_prompt_tokens": first_len, "prefill_tokens": prefill,
            "sampled_tokens": sampled, "cost_usd_upper": (prefill * PRICE["prefill"] + sampled * PRICE["sample"]) / 1e6,
            "trace": steps, "fills": host.fills}


def passive(bars, start, horizon, symbols=p1.SYMBOLS):
    """No-skill benchmark in the same world: at the start, a $2,500 limit buy (limit = close x 1.02,
    sized at that limit) of each symbol, held to the horizon. Uses the same Host, so same costs and caps."""
    h = p1.Host(bars, start, horizon)
    for s in symbols:
        lim = round(float(h.close(s)) * 1.02, 2)
        q = int(2500 // lim)
        h.execute({"command": "submit-stock-order", "conid": p1.CONIDS[s], "symbol": s, "side": "BUY", "quantity": q,
                   "order_type": "LMT", "limit": lim, "tif": "DAY", "order_ref": f"pa:passive{s}"})
    h.terminal()
    return p1.outcome(h)


def starts_for(bars, first, horizon, every, last_data=None):
    dates = [b["session"] for b in bars[p1.SYMBOLS[0]]]
    i0 = next(i for i, d in enumerate(dates) if d >= first)
    return [dates[i] for i in range(i0, len(dates) - horizon - 1, every)]


async def replay(args):
    import tinker
    bars = {k: [b for b in v if b["session"] <= args.last] for k, v in p1.load_bars("ibkr").items()}
    service = tinker.ServiceClient(project_id=PROJECT_ID)
    samplers = {a: (service.create_sampling_client(base_model=BASE) if MODELS[a] is None
                    else service.create_sampling_client(model_path=MODELS[a])) for a in args.arms}
    out_dir = ROOT / "runs" / "navtest" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for h in args.horizons:
        for st in starts_for(bars, args.first, h, args.every):
            for k in range(args.seeds):
                seed = 1000 + k
                for a in args.arms:
                    f = out_dir / f"{a}_{st}_h{h}_s{seed}.json"
                    if not f.exists():
                        jobs.append((a, st, h, seed, f))
    print(f"{len(jobs)} episodes to run", flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    done = 0

    async def one(a, st, h, seed, f):
        nonlocal done
        async with sem:
            host = p1.Host(bars, st, h)
            try:
                r = await episode(samplers[a], host, st, args.temperature, seed)
            except Exception as e:
                print("ERROR", a, st, h, seed, repr(e)[:300], flush=True)
                return
            r.update({"arm": a, "start": st, "horizon": h, "seed": seed, "temperature": args.temperature,
                      "model": MODELS[a] or BASE})
            f.write_text(json.dumps(r, default=str), encoding="utf-8")
            done += 1
            if done % 10 == 0:
                print(f"{done}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)
    await asyncio.gather(*(one(*j) for j in jobs))
    print("finished", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--arms", nargs="+", default=["base", "nav4", "nav9"])
    r.add_argument("--horizons", nargs="+", type=int, default=[20, 63])
    r.add_argument("--first", default="2025-11-10")   # after every NAV training outcome (last 2025-11-07)
    r.add_argument("--last", default="2026-10-12")   # last completed session in the data
    r.add_argument("--every", type=int, default=10)
    r.add_argument("--seeds", type=int, default=2)
    r.add_argument("--temperature", type=float, default=1.0)
    r.add_argument("--concurrency", type=int, default=24)
    r.add_argument("--tag", required=True)
    a = ap.parse_args()
    if a.cmd == "replay":
        asyncio.run(replay(a))


if __name__ == "__main__":
    main()
