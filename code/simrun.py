"""Run a model day by day on SimWorld windows (smoke tests and end-to-end evaluation).

    python simrun.py --model base --starts 2025-03-03 2025-06-02 --days 10 --seeds 2 --tag smoke1
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import tinker

import trader

ROOT = Path(__file__).resolve().parent
PROJECT_ID = "51e502c3-a670-40ad-b14a-c9eb569d4363"
BASE = "openai/gpt-oss-120b"
PRICE = {"prefill": 0.33, "sample": 0.84}


def sampler_for(service, model):
    return service.create_sampling_client(base_model=BASE) if model == "base" else service.create_sampling_client(model_path=model)


async def decide(sc, text, size, iso, temperature, seed, max_tokens, effort):
    prompt = trader.render(text, size, iso, effort)
    params = tinker.SamplingParams(temperature=temperature, max_tokens=max_tokens, stop=[trader.RET, trader.CALL],
                                   **({"seed": seed} if seed is not None else {}))
    res = await sc.sample_async(prompt, 1, params)
    seq = res.sequences[0]
    toks = list(seq.tokens)
    call, err = trader.parse(toks, seq.stop_reason)
    return {"prompt_tokens": prompt.length, "sample_tokens": len(toks), "call": call, "error": err,
            "analysis": trader.analysis_text(toks)[:4000], "tokens": toks, "logprobs": list(seq.logprobs or [])}


async def window(sc, data, d0, days, size, temperature, seed, max_tokens, effort):
    w = trader.SimWorld(data, d0, size)
    log = []
    spy0 = data.c[data.k["SPY"], d0 - 1]
    for i in range(days):
        d = d0 + i
        text, st, feats = trader.state_text(w, data, d)
        for k in range(3):                       # as live: a malformed output is re-asked up to twice
            r = await decide(sc, text, size, data.iso(d), temperature, None if seed is None else seed * 1000 + i * 10 + k, max_tokens, effort)
            if r["call"]:
                break
        r["attempts"] = k + 1
        notes = w.act(r["call"]) if r["call"] else []
        log.append({"session": data.iso(d), "value": st["value"], "gross": st["gross"], "net": st["net"],
                    "prompt_tokens": r["prompt_tokens"], "sample_tokens": r["sample_tokens"], "error": r["error"],
                    "call": r["call"], "notes": notes, "attempts": r["attempts"], "analysis": r["analysis"][:1500]})
        w.advance()
    end_value = w.m.nav()
    liq = w.liquidate()
    spy1 = data.c[data.k["SPY"], d0 + days - 1]
    return {"start": data.iso(d0), "days": days, "seed": seed, "end_value": end_value, "liquidated": liq,
            "return_pct": (liq / size - 1) * 100, "spy_pct": (spy1 / spy0 - 1) * 100,
            "fills": len(w.m.fills), "commissions": w.m.commissions, "financing": w.m.financing, "realized": w.m.realized,
            "navs": [x["value"] for x in log],
            "malformed": sum(1 for x in log if x["error"]), "decisions": log}


async def main_async(a):
    data = trader.Data(last=a.last)
    service = tinker.ServiceClient(project_id=PROJECT_ID, user_metadata={"purpose": f"sim:{a.tag}"})
    sc = sampler_for(service, a.model if a.model != "base" else "base")
    out = ROOT / "runs" / "sim" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for s in a.starts:
        d0 = next(i for i, d in enumerate(data.dates) if d >= s.replace("-", ""))
        for k in range(a.seeds):
            jobs.append((d0, 1 + k))
    t0 = time.time()
    res = await asyncio.gather(*(window(sc, data, d0, a.days, a.size, a.temperature, seed, a.max_tokens, a.effort) for d0, seed in jobs))
    for r in res:
        (out / f"{r['start']}_s{r['seed']}.json").write_text(json.dumps(r, default=float), encoding="utf-8")
    pt = sum(x["prompt_tokens"] for r in res for x in r["decisions"])
    st = sum(x["sample_tokens"] for r in res for x in r["decisions"])
    n = sum(len(r["decisions"]) for r in res)
    print(f"{len(res)} windows, {n} decisions in {time.time() - t0:.0f}s; prompt {pt / n:.0f} tok, sample {st / n:.0f} tok avg; "
          f"malformed {sum(r['malformed'] for r in res)}; cost <= ${(pt * PRICE['prefill'] + st * PRICE['sample']) / 1e6:.3f}")
    for r in res:
        print(f"  {r['start']} s{r['seed']}: return {r['return_pct']:+.2f}% (SPY {r['spy_pct']:+.2f}%), fills {r['fills']}, "
              f"commissions ${r['commissions']:.2f}, malformed {r['malformed']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base")
    ap.add_argument("--starts", nargs="+", required=True)
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--size", type=float, default=50000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--last", default="20261012")
    ap.add_argument("--tag", required=True)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
