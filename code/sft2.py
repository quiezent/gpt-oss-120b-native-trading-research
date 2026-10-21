"""Rejection-sampling SFT of rl2-step010 (runs/ml/research/rs_sft_prereg.md).

  select: from train2 --keep-best rollouts (runs/train/<rollouts>/best.jsonl), keep a group's best look only if
      1. its reward is above its group's median, 2. it has no malformed submit, rejected order or trace error, and
      3. its notes are grounded: every symbol (2+ letters) they name appears in what the look received (prompt or tool
      results), and every event term they use (earnings, beat, 8-K, CPI, ...) appears in a tool result it received.
  train: cross-entropy on the kept looks' sampled tokens only (train2's mask), from the rl2-step010 training state
      (weights only, fresh Adam), saving a state and a sampler after each epoch.

    python sft2.py --rollouts rs1 --tag sft-rs1 --select-only
    python sft2.py --rollouts rs1 --tag sft-rs1
"""

import argparse
import asyncio
import json
import random
import re
import statistics
from pathlib import Path

import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData

import trader

ROOT = Path(__file__).resolve().parent
PROJECT_ID = "51e502c3-a670-40ad-b14a-c9eb569d4363"
INIT = "tinker://14128c11-bd40-50d9-9def-3ef44adbfb72:train:0/weights/rl2-step010"
PRICE_TRAIN = 0.737
DAY = 86400
TERMS = {"earnings": r"\bearnings\b", "beat": r"\bbeat(s|ing)?\b", "miss": r"\bmiss(es|ed)?\b", "guidance": r"\bguidance\b",
         "upgrade": r"\bupgrad", "downgrade": r"\bdowngrad", "8-K": r"\b8-K\b", "filing": r"\bfiling", "FOMC": r"\bFOMC\b",
         "CPI": r"\bCPI\b", "payroll": r"\bpayroll", "jobs report": r"\bjobs report", "GDP": r"\bGDP\b", "PCE": r"\bPCE\b"}


def received(row):
    """(prompt text, tool-result text) the look was given: the prompt, then every unsampled token after it."""
    tok, seq, mask, n = trader.tokenizer(), row["seq"], row["mask"], row["prompt_len"]
    res = [t for t, m in zip(seq[n:], mask[n:]) if not m]
    return tok.decode(seq[:n]), tok.decode(res)


def grounded(row, symbols):
    notes = (row["call"] or {}).get("notes") or ""
    prompt, tools_text = received(row)
    seen = prompt + "\n" + tools_text
    bad_syms = sorted({s for s in re.findall(r"\b[A-Z]{2,5}\b", notes) if s in symbols and not re.search(rf"\b{s}\b", seen)})
    bad_terms = sorted(k for k, p in TERMS.items()
                       if re.search(p, notes, re.I) and not re.search(p, tools_text, re.I))
    return not bad_syms and not bad_terms, bad_syms, bad_terms


def select(rollouts):
    rows = {}
    for l in (ROOT / "runs" / "train" / rollouts / "best.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            rows[(r["step"], r["date"], r["burn"], r["ml"])] = r          # a resumed step may repeat a row
    symbols = {s for s in trader.universe() if " " not in s} | trader.etfs()
    kept, why = [], {"not above median": 0, "malformed/rejected/error": 0, "ungrounded": 0}
    examples = []
    for r in rows.values():
        if not r["R"] > statistics.median(r["rs"]):
            why["not above median"] += 1
            continue
        if r["call"] is None or r["malformed"] or r["rejected"] or any("error" in t for t in r["trace"]):
            why["malformed/rejected/error"] += 1
            continue
        ok, bs, bt = grounded(r, symbols)
        if not ok:
            why["ungrounded"] += 1
            if len(examples) < 8:
                examples.append({"date": r["date"], "symbols": bs, "terms": bt, "notes": r["call"]["notes"][:300]})
            continue
        kept.append(r)
    return list(rows.values()), kept, why, examples


async def train(cfg, kept):
    out = ROOT / "runs" / "train" / cfg.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(cfg), indent=1))
    datums = [types.Datum(model_input=types.ModelInput.from_ints(r["seq"][:-1]), loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(r["seq"][1:])),
        "weights": TensorData.from_torch(torch.tensor([float(m) for m in r["mask"][1:]]))}) for r in kept]
    service = tinker.ServiceClient(project_id=PROJECT_ID, user_metadata={"purpose": f"train:{cfg.tag}"})
    tc = await service.create_training_client_from_state_async(cfg.init)
    adam = types.AdamParams(learning_rate=cfg.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    rng, step, tokens = random.Random(cfg.seed), 0, 0
    for ep in range(1, cfg.epochs + 1):
        order = list(range(len(datums)))
        rng.shuffle(order)
        for i in range(0, len(order), cfg.batch):
            batch = [datums[j] for j in order[i:i + cfg.batch]]
            fb = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
            op = await tc.optim_step_async(adam)
            res = await fb.result_async()
            await op.result_async()
            step += 1
            n = sum(d.model_input.length for d in batch)
            w = sum(float(d.loss_fn_inputs["weights"].to_torch().sum()) for d in batch)
            nll = -sum(float((o["logprobs"].to_torch() * d.loss_fn_inputs["weights"].to_torch()).sum())
                       for o, d in zip(res.loss_fn_outputs, batch)) / max(w, 1)
            tokens += n
            row = {"epoch": ep, "step": step, "looks": len(batch), "nll_per_token": nll, "cost_usd": tokens * PRICE_TRAIN / 1e6}
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        name = f"{cfg.tag}-ep{ep}"
        st = await (await tc.save_state_async(name, ttl_seconds=cfg.state_ttl_days * DAY)).result_async()
        sp = await (await tc.save_weights_for_sampler_async(name, ttl_seconds=cfg.sampler_ttl_days * DAY)).result_async()
        with (out / "checkpoints.jsonl").open("a") as f:
            f.write(json.dumps({"epoch": ep, "state": st.path, "sampler": sp.path}) + "\n")
        print(f"saved {name}: {sp.path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--init", default=INIT)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--min-kept", type=int, default=60)
    ap.add_argument("--state-ttl-days", type=int, default=21)
    ap.add_argument("--sampler-ttl-days", type=int, default=30)
    ap.add_argument("--select-only", action="store_true")
    cfg = ap.parse_args()
    rows, kept, why, examples = select(cfg.rollouts)
    rep = {"groups": len(rows), "kept": len(kept), "dropped": why, "kept_ml": sum(r["ml"] for r in kept),
           "kept_mean_R": statistics.mean(r["R"] for r in kept) if kept else None,
           "kept_mean_calls": statistics.mean(r["calls"] for r in kept) if kept else None,
           "all_best_mean_calls": statistics.mean(r["calls"] for r in rows) if rows else None,
           "kept_tokens": sum(len(r["seq"]) for r in kept), "ungrounded_examples": examples}
    out = ROOT / "runs" / "train" / cfg.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "ungrounded_examples"}), flush=True)
    if cfg.select_only:
        return
    if len(kept) < cfg.min_kept:
        raise SystemExit(f"only {len(kept)} looks kept (< {cfg.min_kept}): no SFT, as pre-registered")
    asyncio.run(train(cfg, kept))


if __name__ == "__main__":
    main()
