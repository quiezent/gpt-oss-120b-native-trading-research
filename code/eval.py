"""Paired validation: every model scored on the same starting books, with the training reward.

    python eval.py states                       # build the validation books once, with base (cached)
    python eval.py score --model base --tag base
    python eval.py score --model tinker://... --tag rl1-step010
    python eval.py compare base rl1-step010     # mean paired difference, block-bootstrap interval

A starting book for date d is made by base trading a fresh $50k sleeve for k = (d mod 10) sessions
before d; its decisions are cached and replayed, so every model sees identical books.
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

import tinker

import reward
import trader

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "eval"
SPLITS = {"val": ("20260102", "20260514"), "test": ("20260601", "20260925")}
PROJECT_ID = "51e502c3-a670-40ad-b14a-c9eb569d4363"
BASE = "openai/gpt-oss-120b"
SIZE = 50000.0


def sampler(service, model):
    return service.create_sampling_client(base_model=BASE) if model == "base" else service.create_sampling_client(model_path=model)


def split_dates(data, split):
    a, b = SPLITS[split]
    return [i for i, x in enumerate(data.dates) if a <= x <= b]


def states_file(split):
    return OUT / ("val_states.json" if split == "val" else f"{split}_states.json")


def result_file(tag, split):
    return OUT / (f"{tag}.json" if split == "val" else f"{split}-{tag}.json")


def rebuild(data, d, calls):
    k = len(calls)
    w = trader.SimWorld(data, d - k, SIZE)
    for c in calls:
        w.events.clear()        # as in training, events are consumed at each look
        if c:
            w.act(c)
        w.advance()
    return w


async def build_states(args):
    data = trader.Data(last="20261012")
    sc = sampler(tinker.ServiceClient(project_id=PROJECT_ID, user_metadata={"purpose": f"eval:{args.split}:states"}), "base")
    STATES = states_file(args.split)
    states = json.loads(STATES.read_text()) if STATES.exists() else {}

    async def one(d):
        key = data.dates[d]
        if key in states:
            return
        k = d % 10
        w = trader.SimWorld(data, d - k, SIZE)
        calls = []
        for i in range(k):
            text, _, _ = trader.state_text(w, data, d - k + i)
            p = trader.render(text, SIZE, data.iso(d - k + i))
            seq = (await sc.sample_async(p, 1, tinker.SamplingParams(temperature=1.0, seed=d * 100 + i, max_tokens=2048,
                                                                      stop=[trader.RET, trader.CALL]))).sequences[0]
            call, _ = trader.parse(list(seq.tokens), seq.stop_reason)
            calls.append(call)
            if call:
                w.act(call)
            w.advance()
        states[key] = calls
    await asyncio.gather(*(one(d) for d in split_dates(data, args.split)))
    OUT.mkdir(parents=True, exist_ok=True)
    STATES.write_text(json.dumps(states))
    print(f"{len(states)} validation books")


async def score(args):
    data = trader.Data(last="20261012")
    states = json.loads(states_file(args.split).read_text())
    sc = sampler(tinker.ServiceClient(project_id=PROJECT_ID, user_metadata={"purpose": f"eval:{args.split}:{args.tag}"}), args.model)
    sem = asyncio.Semaphore(32)

    async def one(d):
        async with sem:
            w = rebuild(data, d, states[data.dates[d]])
            text, st, _ = trader.state_text(w, data, d)
            feats, sm = data.features(d)
            p = trader.render(text, SIZE, data.iso(d))
            res = await sc.sample_async(p, args.samples, tinker.SamplingParams(temperature=1.0, max_tokens=2048,
                                                                               stop=[trader.RET, trader.CALL]))
            rows = []
            for seq in res.sequences:
                call, err = trader.parse(list(seq.tokens), seq.stop_reason)
                try:
                    parts = reward.branch(w.clone(), call, data, d, feats, sm)
                except Exception as e:
                    call, err = None, f"simulation error: {e!r}"[:200]
                    parts = reward.branch(w.clone(), None, data, d, feats, sm)
                rows.append({**parts, "call": call, "error": err, "tokens": len(seq.tokens)})
            return {"date": data.dates[d], "samples": rows}
    res = await asyncio.gather(*(one(d) for d in split_dates(data, args.split)))
    OUT.mkdir(parents=True, exist_ok=True)
    result_file(args.tag, args.split).write_text(json.dumps({"model": args.model, "results": res}, default=float))
    allr = [s for r in res for s in r["samples"]]
    m = lambda k: sum(float(s[k]) for s in allr) / len(allr)
    print(f"{args.tag}: {len(res)} dates x {args.samples}: R {m('R'):+.3f} | book {m('book_pp'):+.3f} hedge {m('hedge_pp'):+.3f} "
          f"premium {m('premium_pp'):+.3f} risk {m('risk_pp'):+.3f} | malformed {m('malformed'):.3f} gross {m('avg_gross'):.2f}")


def compare(args):
    a = json.loads(result_file(args.a, args.split).read_text())["results"]
    b = json.loads(result_file(args.b, args.split).read_text())["results"]
    ma = {r["date"]: sum(s["R"] for s in r["samples"]) / len(r["samples"]) for r in a}
    mb = {r["date"]: sum(s["R"] for s in r["samples"]) / len(r["samples"]) for r in b}
    dates = sorted(set(ma) & set(mb))
    diff = [mb[x] - ma[x] for x in dates]
    mean = sum(diff) / len(diff)
    rng = random.Random(0)
    L = 10
    blocks = [diff[i:i + L] for i in range(0, len(diff) - L + 1)]
    boots = []
    for _ in range(20000):
        xs = []
        while len(xs) < len(diff):
            xs += rng.choice(blocks)
        boots.append(sum(xs[:len(diff)]) / len(diff))
    boots.sort()
    lo, hi = boots[1000], boots[19000]
    parts = {}
    for k in ("book_pp", "hedge_pp", "premium_pp", "risk_pp", "malformed", "avg_gross"):
        pa = sum(s[k] for r in a for s in r["samples"]) / sum(len(r["samples"]) for r in a)
        pb = sum(s[k] for r in b for s in r["samples"]) / sum(len(r["samples"]) for r in b)
        parts[k] = round(pb - pa, 3)
    print(f"{args.b} - {args.a}: {mean:+.3f} pp per decision over {len(dates)} dates; 90% block-bootstrap [{lo:+.3f}, {hi:+.3f}]")
    print("part differences:", parts)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("states")
    st.add_argument("--split", default="val", choices=list(SPLITS))
    s = sub.add_parser("score")
    s.add_argument("--split", default="val", choices=list(SPLITS))
    s.add_argument("--model", required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--samples", type=int, default=4)
    c = sub.add_parser("compare")
    c.add_argument("--split", default="val", choices=list(SPLITS))
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "states":
        asyncio.run(build_states(a))
    elif a.cmd == "score":
        asyncio.run(score(a))
    else:
        compare(a)


if __name__ == "__main__":
    main()
