"""rl2: GRPO on full research looks, starting from rl1's last training state.

What changed from rl1 (train.py), from the first week of live trading:
  - Each sample is a multi-turn look: the policy calls research tools (tools.py, run on simgw.SimGateway, so every
    answer is point-in-time) and then submits orders and notes. Only its own sampled tokens are trained.
  - The prompt is the live one (trader.live_system/live_steps/live_user): this week's goal, the margin-limit
    notice, and the notes its previous decision left (burn-in decisions write them). News isn't available for
    past dates, so the training variant leaves news out.
  - In half the groups the look also has the ML forecast tool (forecast.py), so one checkpoint serves rl2 and rl2-ml.
  - Reward: reward.branch with reject_penalty: each order rejected for bad fields or margin (gross > 2x) costs 0.25.
  - --news (off by default): the look also has news, article and tape, answered point-in-time from cached IBKR
    headlines, EDGAR 8-Ks and the macro calendar (ml_research/simnews.py), with the live prompt's news steps.

    python train2.py --tag rl2 --steps 20
    python train2.py --tag rl2 --steps 20 --resume
    python train2.py --tag simnews-smoke --steps 1 --groups 2 --group 2 --burn 1 --news --no-update --init tinker://...
    python train2.py --tag rs1 --steps 15 --news --no-update --sampler tinker://.../sampler_weights/rl2-step010 --keep-best
        # rollouts for rejection-sampling SFT (sft2.py): a sampling client only, and each group's best look saved
        # with its tokens and mask to best.jsonl
    python train2.py --tag rl3 --steps 20 --news --drill-groups 8 --init tinker://.../weights/rl2-step010
        # calibration pilot: looks give p_fill/p_win per order; p_win's skill is reward.calibration_bonus (weight
        # --lam-cal), p_fill is only reported; plus --drill-groups forecast-drill questions a step (ml_research/drill.py,
        # the cookbook's forecasting recipe: one answer, Brier reward, advantage --kappa x (reward - group mean), answers
        # cut off at the token cap left out) in the same update; drill_grad_share logs the drill's share of the update

Effort: --effort high by default. Each turn may write up to --max-tokens (agent.MAX_OUT), less what the context
(agent.CONTEXT) has no room for; a look stops as "context full" below agent.MIN_ROOM tokens of room.
"""

import argparse
import asyncio
import json
import random
import time
from datetime import date
from pathlib import Path

import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData

import agent
import reward
import simgw
import tools
import trader
from ml_research import drill

ROOT = Path(__file__).resolve().parent
PROJECT_ID = "51e502c3-a670-40ad-b14a-c9eb569d4363"
BASE = "openai/gpt-oss-120b"
TRAIN_FIRST, TRAIN_LAST = "20250103", "20251212"   # forecasts (risk_oos.npz) start with the close of 20250102
VAL_FIRST = "20260102"                              # eval2's validation split (and the drill's) starts here
SIZE = 30000.0
PRICE = {"prefill": 0.33, "cached": 0.066, "sample": 0.84, "train": 0.737}
DAY = 86400
REJECT_PENALTY = 0.25
RL1_STATE = "runs/train/rl1/checkpoints.jsonl"
LAMBDA_CAL, KAPPA = 0.0, 10.0      # pre-registered (rl3 pilot): look calibration bonus weight (0: p_fill/p_win only logged,
                                   # trader_cookbook.md C4); drill advantage scale


def iso(data, d):
    return date.fromisoformat(data.iso(d))


def header(data, d):
    t = iso(data, d)
    return f"Date: {t.isoformat()} ({t.strftime('%A')}), 09:00 ET. Before the open. Orders you submit now work from the 09:30 open."


def sessions_left(data, d):
    g = trader.goal_date(iso(data, d))
    return sum(1 for i in range(d, len(data.dates)) if data.dates[i] <= g.strftime("%Y%m%d")) or 1


def build(world, data, d, memo, ml, news=False):
    """(system, user, specs-free state) for a pre-open look on session d."""
    text, st, feats = trader.state_text(world, data, d, header=header(data, d))
    goal = trader.goal_line(SIZE, st["value"], sessions_left(data, d), iso(data, d))
    user = trader.live_user(text, goal, memo, steps=trader.live_steps(ml, news=news))
    return trader.live_system(SIZE, iso(data, d), ml=ml, news=news), user


def renderer(data, d, effort):
    from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
    return GptOssRenderer(trader.tokenizer(), use_system_prompt=True, reasoning_effort=effort, current_date=data.iso(d))


def prompt_ints(r, specs, system, user):
    msgs = r.create_conversation_prefix_with_tools(agent.harmony_specs(specs), system)
    msgs.append({"role": "user", "content": user})
    return list(r.build_generation_prompt(msgs).to_ints())


def params(max_tokens, used=0):
    """Sampling params for a turn after `used` context tokens: up to max_tokens, less what the context has no room for."""
    return tinker.SamplingParams(temperature=1.0, max_tokens=min(max_tokens, agent.CONTEXT - used), stop=[trader.RET, trader.CALL])


async def burn_decision(sc, world, data, d, memo, cfg):
    """A cheap burn-in decision (submit only, no research) that moves the book forward and writes notes."""
    system, user = build(world, data, d, memo, ml=False)
    p = prompt_ints(renderer(data, d, cfg.effort), [trader.TOOL_LIVE], system, user)
    res = await sc.sample_async(tinker.ModelInput.from_ints(p), 1, params(cfg.max_tokens, len(p)))
    s = res.sequences[0]
    kind, name, val = agent.parse_any(list(s.tokens), s.stop_reason, {"submit"})
    ok = kind == "TOOL" and isinstance(val, dict) and isinstance(val.get("place", []), list) and isinstance(val.get("cancel", []), list)
    call = {"place": val.get("place", []), "cancel": val.get("cancel", []), "notes": str(val.get("notes") or "")} if ok else None
    return call, len(p), len(s.tokens)


async def member_look(sc, gw, names, first, prompt, cfg):
    """Continue one sampled look from its first turn to submit. Returns the full token record."""
    seq, mask, lps = list(prompt), [0] * len(prompt), [0.0] * len(prompt)
    toks, lp, stop = first
    calls, trace, cost, force = 0, [], {"prefill": 0, "cached": 0, "sampled": 0}, []
    for step in range(tools.MAX_CALLS + 2):
        if step:
            if agent.CONTEXT - len(seq) < agent.MIN_ROOM:
                trace.append({"error": "context full"})
                break
            try:
                res = await sc.sample_async(tinker.ModelInput.from_ints(seq), 1, params(cfg.max_tokens, len(seq)))
            except Exception as e:            # keep the tokens already paid for; the look counts as malformed
                trace.append({"step": step, "error": f"sampling error: {e!r}"[:200]})
                break
            s = res.sequences[0]
            toks, lp, stop = list(s.tokens), list(s.logprobs), s.stop_reason
            cost["cached"] += len(seq) - last_new
            cost["prefill"] += last_new
        cost["sampled"] += len(toks)
        seq += toks
        mask += [1] * len(toks)
        lps += lp
        kind, name, val = agent.parse_any(force + toks, stop, names)
        if kind != "TOOL":
            trace.append({"step": step, "error": f"{kind}: {str(val)[:200]}"})
            return {"call": None, "seq": seq, "mask": mask, "lps": lps, "trace": trace, "calls": calls, "cost": cost}
        if name == "submit":
            ok = isinstance(val, dict) and isinstance(val.get("place", []), list) and isinstance(val.get("cancel", []), list)
            trace.append({"step": step, "tool": "submit"})
            call = {"place": val.get("place", []), "cancel": val.get("cancel", []), "notes": str(val.get("notes") or "")} if ok else None
            return {"call": call, "seq": seq, "mask": mask, "lps": lps, "trace": trace, "calls": calls, "cost": cost}
        if calls >= tools.MAX_CALLS:      # as live (agent.gptoss_look): the next turn opens the submit call (masked)
            out = f"research limit reached ({tools.MAX_CALLS} calls): call submit now"
            force = agent.SUBMIT_OPEN()
        else:
            calls += 1
            out = tools.run(gw, name, val)
        trace.append({"step": step, "tool": name, "args": val, "result": out[:600]})
        rt = agent.tool_result_tokens(name, out) + force
        seq += rt
        mask += [0] * len(rt)
        lps += [0.0] * len(rt)
        last_new = len(toks) + len(rt)
    return {"call": None, "seq": seq, "mask": mask, "lps": lps, "trace": trace + [{"error": "no submit"}], "calls": calls, "cost": cost}


async def window(sc, data, pre, d, k, ml, forecast, cfg):
    """Burn in k sessions (cheap decisions with notes), then sample a group of full looks at d and score them."""
    w = trader.SimWorld(data, d - k, SIZE)
    memo, burn_cost = {}, {"prefill": 0, "sampled": 0}
    for i in range(k):
        di = d - k + i
        call, pl, sl = await burn_decision(sc, w, data, di, memo, cfg)
        burn_cost["prefill"] += pl
        burn_cost["sampled"] += sl
        if call:
            w.act(call)
            memo = {"when": f"{iso(data, di):%a %d %b} 09:00 ET", "notes": str(call.get("notes") or "")[:2000]}
        w.advance()
    news = getattr(cfg, "news", False)
    if news:
        from ml_research import simnews
        gw = simnews.NewsSimGateway(data, d, pre, forecast=forecast if ml else None)
    else:
        gw = simgw.SimGateway(data, d, pre, forecast=forecast if ml else None)
    specs = tools.available(gw)
    if ml:                                   # never prompt for a tool the look doesn't have
        import forecast as fc
        assert forecast is not None, "ML episode without a forecast provider"
        specs = specs + [fc.SPEC]
    names = {s["name"] for s in specs} | {"submit"}
    system, user = build(w, data, d, memo, ml, news=news)
    prompt = prompt_ints(renderer(data, d, cfg.effort), specs + [trader.TOOL_LIVE], system, user)
    res = await sc.sample_async(tinker.ModelInput.from_ints(prompt), cfg.group, params(cfg.max_tokens, len(prompt)))
    looks = await asyncio.gather(*(member_look(sc, gw, names, (list(s.tokens), list(s.logprobs), s.stop_reason), prompt, cfg)
                                   for s in res.sequences))
    feats_all, sigma_m = data.features(d)
    members = []
    for lk in looks:
        call = lk["call"]
        try:
            parts = reward.branch(w.clone(), call, data, d, feats_all, sigma_m, reject_penalty=REJECT_PENALTY)
        except Exception as e:               # an order the simulator can't handle: score as malformed
            lk["trace"].append({"error": f"simulation error: {e!r}"[:200]})
            call = None
            parts = reward.branch(w.clone(), None, data, d, feats_all, sigma_m, reject_penalty=REJECT_PENALTY)
        cal = reward.calibration_bonus(parts["orders"])
        members.append({**lk, "call": call, **{k: float(v) if not isinstance(v, int) else v for k, v in parts.items() if k != "orders"},
                        "orders": parts["orders"], "cal": cal, "R_trade": float(parts["R"]),
                        "R": float(parts["R"]) + cfg.lam_cal * cal["bonus"]})
    return {"date": data.dates[d], "burn": k, "ml": ml, "prompt_len": len(prompt), "members": members, "burn_cost": burn_cost,
            "names": sorted(names)}


def datum(seq, mask, lps, a):
    return types.Datum(model_input=types.ModelInput.from_ints(seq[:-1]), loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(seq[1:])),
        "logprobs": TensorData.from_torch(torch.tensor(lps[1:])),
        "advantages": TensorData.from_torch(torch.tensor([a * x for x in mask[1:]])),
    })


def datums_for(win):
    rs = [m["R"] for m in win["members"]]
    mean = sum(rs) / len(rs)
    if max(rs) - min(rs) < 1e-9:
        return []
    return [datum(m["seq"], m["mask"], m["lps"], float(r - mean)) for m, r in zip(win["members"], rs)]


# ---------------------------------------------------------------- forecast drill (cookbook forecasting recipe)
def drill_prompt(q, effort):
    from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
    x = q["date"]
    r = GptOssRenderer(trader.tokenizer(), use_system_prompt=True, reasoning_effort=effort, current_date=f"{x[:4]}-{x[4:6]}-{x[6:]}")
    return prompt_ints(r, [], drill.SYSTEM, drill.prompt(q))


def drill_answer(tokens, stop_reason):
    """The final-channel text of a one-turn Harmony answer; None if there is none (cut off, a tool call, no final)."""
    if str(stop_reason).split(".")[-1].lower() != "stop" or not tokens or tokens[-1] != trader.RET:
        return None
    m = trader.SEG.match(trader.tokenizer().decode(tokens).split("<|start|>assistant")[-1])
    return m["body"] if m and m["ch"] == "final" and "<|" not in m["body"] else None


async def drill_group(sc, q, n, cfg):
    """n answers to one drill question (no tools), each scored by the Brier reward (malformed 0). cut: the answer hit
    the token cap."""
    p = drill_prompt(q, cfg.effort)
    res = await sc.sample_async(tinker.ModelInput.from_ints(p), n, params(cfg.max_tokens, len(p)))
    out = []
    for s in res.sequences:
        text = drill_answer(list(s.tokens), s.stop_reason)
        pf = drill.parse_forecast(text) if text is not None else None
        out.append({"p": pf, "r": drill.score(pf, q["outcome"]), "tokens": list(s.tokens), "lps": list(s.logprobs),
                    "cut": str(s.stop_reason).split(".")[-1].lower() != "stop"})
    return {"q": q, "prompt": p, "samples": out}


def drill_datums(g, kappa):
    """Answers cut off at the token cap are left out of the update (counted in drill_cut): scored 0 at kappa 10, one
    such answer outweighed every look of its step (review, 18 Oct), so the update taught short reasoning, not calibration."""
    ss = [s for s in g["samples"] if not s["cut"]]
    rs = [s["r"] for s in ss]
    if len(rs) < 2 or max(rs) - min(rs) < 1e-9:
        return []
    mean = sum(rs) / len(rs)
    n = len(g["prompt"])
    return [datum(g["prompt"] + s["tokens"], [0] * n + [1] * len(s["tokens"]), [0.0] * n + s["lps"], kappa * (s["r"] - mean))
            for s in ss]


def drill_stats(groups):
    """drill_reward over every answer (malformed 0, as the recipe); drill_reward_parsed over the parsed answers only."""
    xs = [(g["q"]["type"], s) for g in groups for s in g["samples"]]
    if not xs:
        return {}
    ps = [s for _, s in xs if s["p"] is not None]
    by = {t: [s["r"] for u, s in xs if u == t and s["p"] is not None] for t in drill.TYPES}
    return {"drill_reward": sum(s["r"] for _, s in xs) / len(xs), "drill_reward_parsed": sum(s["r"] for s in ps) / len(ps) if ps else None,
            "drill_malformed": 1 - len(ps) / len(xs), "drill_cut": sum(s["cut"] for _, s in xs) / len(xs),
            "drill_mean_p": sum(s["p"] for s in ps) / len(ps) if ps else None, "drill_tokens": sum(len(s["tokens"]) for _, s in xs) / len(xs),
            "drill_by_type_parsed": {t: round(sum(v) / len(v), 4) for t, v in by.items() if v}, "drill_samples": len(xs)}


def cal_stats(looks):
    """Over looks (each with reward.branch "orders" and its "cal"): the mean bonus (p_win skill vs the side's base rate),
    and p_fill and p_win scored apart next to their trivial baselines (reward.prob_scores)."""
    ps = reward.prob_scores([o for lk in looks for o in lk["orders"]])
    return {"cal_bonus": sum(lk["cal"]["bonus"] for lk in looks) / len(looks) if looks else None,
            "cal_scored": sum(lk["cal"]["n_scored"] for lk in looks),
            **{f"{k}_{m}": v for k, d in ps.items() for m, v in d.items()}}


def adv_mass(datums):
    """sum |advantage| over sampled tokens: each source's weight in the update."""
    return sum(sum(abs(x) for x in d.loss_fn_inputs["advantages"].data) for d in datums)


def summarize(step, wins, n_datums, secs, cost):
    mem = [m for w in wins for m in w["members"]]
    if not mem:
        return {"step": step, "samples": 0, "datums": n_datums, "seconds": secs, "cost_usd": cost}
    n = len(mem) or 1
    avg = lambda key: sum(float(m.get(key, 0)) for m in mem) / n
    used = {}
    for m in mem:
        for t in m["trace"]:
            if t.get("tool") and t["tool"] != "submit":
                used[t["tool"]] = used.get(t["tool"], 0) + 1
    return {"step": step, "samples": len(mem), "datums": n_datums, "R": avg("R"), "R_trade": avg("R_trade"), "book_pp": avg("book_pp"),
            "hedge_pp": avg("hedge_pp"), "risk_pp": avg("risk_pp"), "malformed": avg("malformed"), "rejected": avg("rejected"),
            "avg_gross": avg("avg_gross"), "research_calls": avg("calls"), "tools": used,
            "notes_written": sum(1 for m in mem if m["call"] and m["call"].get("notes")) / n,
            "orders": sum(len(m["call"]["place"]) for m in mem if m["call"]) / n,
            "ml_groups": sum(1 for w in wins if w["ml"]), **cal_stats(mem),
            "look_tokens": sum(m["cost"]["sampled"] for m in mem) / n, "look_len_max": max(len(m["seq"]) for m in mem),
            "context_full": sum(any(t.get("error") == "context full" for t in m["trace"]) for m in mem) / n,
            "seconds": secs, "cost_usd": cost}


async def main_async(cfg):
    out = ROOT / "runs" / "train" / cfg.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(cfg), indent=1))
    data = trader.Data(last="20261015")
    pre = simgw.Precomputed(data)
    forecast = None
    if cfg.ml_frac > 0:
        import forecast as fc
        forecast = fc.sim_provider(data)
    first = next(i for i, x in enumerate(data.dates) if x >= TRAIN_FIRST)
    last = max(i for i, x in enumerate(data.dates) if x <= TRAIN_LAST)
    # a look's p_win resolves at the close 5 sessions after a fill on session d+9 at the latest: before validation starts
    pool = [d for d in range(first, last + 1) if d - cfg.burn >= 260 and d + reward.HORIZON + 4 < len(data.dates)
            and data.dates[d + reward.HORIZON + 4] < VAL_FIRST]
    service = tinker.ServiceClient(project_id=PROJECT_ID, user_metadata={"purpose": f"train:{cfg.tag}"})
    ckpts = out / "checkpoints.jsonl"
    start, tc = 0, None
    if cfg.sampler:
        if not cfg.no_update:
            raise SystemExit("--sampler only samples: use it with --no-update")
        if cfg.resume:
            start = max([json.loads(l)["step"] for l in (out / "metrics.jsonl").read_text().splitlines() if l.strip()]
                        if (out / "metrics.jsonl").exists() else [0])
        print(f"sampling from {cfg.sampler}" + (f", resuming after step {start}" if start else ""), flush=True)
    elif cfg.resume:
        if not ckpts.exists():
            raise SystemExit(f"--resume: no checkpoint in {ckpts}")
        rows = [json.loads(l) for l in ckpts.read_text().splitlines() if l.strip()]
        tc = await service.create_training_client_from_state_with_optimizer_async(rows[-1]["state"])
        start = rows[-1]["step"]
        for f in (out / "metrics.jsonl", out / "samples.jsonl", out / "drill.jsonl"):   # drop steps after the checkpoint; they re-run
            if f.exists():
                keep = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l)["step"] <= start]
                f.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
        print(f"resumed from step {start}: {rows[-1]['state']}", flush=True)
    elif cfg.init:
        tc = await service.create_training_client_from_state_with_optimizer_async(cfg.init)
        print(f"starting from {cfg.init}", flush=True)
    else:
        rl1 = [json.loads(l) for l in (ROOT / RL1_STATE).read_text().splitlines() if l.strip()][-1]
        tc = await service.create_training_client_from_state_with_optimizer_async(rl1["state"])
        print(f"starting from rl1 step {rl1['step']}: {rl1['state']}", flush=True)
    adam = types.AdamParams(learning_rate=cfg.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    rng = random.Random(cfg.seed)
    draw = lambda: (rng.sample(pool, cfg.groups), [rng.randint(0, cfg.burn) for _ in range(cfg.groups)],
                    [rng.random() < cfg.ml_frac for _ in range(cfg.groups)])
    for _ in range(start):                 # keep the date stream identical across a resume
        draw()
    dq = drill.load("train") if cfg.drill_groups else []
    order = random.Random(cfg.seed + 1).sample(range(len(dq)), len(dq))     # each question once before any repeats
    for step in range(start, cfg.steps):
        t0 = time.time()
        dates, burns, mls = draw()
        qs = [dq[order[(step * cfg.drill_groups + j) % len(dq)]] for j in range(cfg.drill_groups)]
        sc = service.create_sampling_client(model_path=cfg.sampler) if cfg.sampler else await tc.save_weights_and_get_sampling_client_async()
        res = await asyncio.gather(*(window(sc, data, pre, d, k, ml, forecast, cfg) for d, k, ml in zip(dates, burns, mls)),
                                   *(drill_group(sc, q, cfg.drill_group, cfg) for q in qs), return_exceptions=True)
        wins, drills = res[:len(dates)], res[len(dates):]
        errs = [w for w in wins if isinstance(w, Exception)]
        derrs = [g for g in drills if isinstance(g, Exception)]
        wins = [w for w in wins if not isinstance(w, Exception)]
        drills = [g for g in drills if not isinstance(g, Exception)]
        for e in (errs + derrs)[:3]:
            print("window error:" if e in errs else "drill error:", repr(e)[:400], flush=True)
        ldatums = [x for w in wins for x in datums_for(w)]
        ddatums = [x for g in drills for x in drill_datums(g, cfg.kappa)]
        datums = ldatums + ddatums
        if datums and not cfg.no_update:
            fb = await tc.forward_backward_async(datums, loss_fn=cfg.loss)
            op = await tc.optim_step_async(adam)
            await fb.result_async()
            await op.result_async()
        c = {"prefill": 0, "cached": 0, "sampled": 0}
        for w in wins:
            c["prefill"] += w["prompt_len"] + w["burn_cost"]["prefill"]
            c["sampled"] += w["burn_cost"]["sampled"]
            for m in w["members"]:
                for k in c:
                    c[k] += m["cost"][k]
        usd = lambda c, trained: (c["prefill"] * PRICE["prefill"] + c["cached"] * PRICE["cached"] + c["sampled"] * PRICE["sample"]
                                  + (0 if cfg.no_update else trained) * PRICE["train"]) / 1e6
        cd = {"prefill": sum(len(g["prompt"]) for g in drills), "cached": 0,
              "sampled": sum(len(x["tokens"]) for g in drills for x in g["samples"])}
        cost_drill = usd(cd, sum(d.model_input.length for d in ddatums))
        cost = usd(c, sum(d.model_input.length for d in ldatums)) + cost_drill
        row = summarize(step + 1, wins, len(datums), time.time() - t0, cost)
        lm, dm = adv_mass(ldatums), adv_mass(ddatums)
        row.update(drill_stats(drills), drill_datums=len(ddatums), drill_grad_share=dm / (lm + dm) if lm + dm else None,
                   cost_drill_usd=cost_drill, window_errors=len(errs), drill_errors=len(derrs))
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        with (out / "samples.jsonl").open("a") as f:
            for w in wins:
                for m in w["members"]:
                    f.write(json.dumps({"step": step + 1, "date": w["date"], "burn": w["burn"], "ml": w["ml"], "R": m["R"],
                                        "R_trade": m["R_trade"], "rejected": m["rejected"], "calls": m["calls"], "call": m["call"],
                                        "orders": m["orders"], "cal": m["cal"], "trace": m["trace"]}, default=float) + "\n")
        if drills:
            with (out / "drill.jsonl").open("a") as f:
                for g in drills:
                    f.write(json.dumps({"step": step + 1, "id": g["q"]["id"], "type": g["q"]["type"], "outcome": g["q"]["outcome"],
                                        "p": [x["p"] for x in g["samples"]], "r": [x["r"] for x in g["samples"]],
                                        "tokens": [len(x["tokens"]) for x in g["samples"]], "cut": [x["cut"] for x in g["samples"]],
                                        "answer": trader.tokenizer().decode(g["samples"][0]["tokens"])[-300:]}) + "\n")
        if cfg.keep_best:
            with (out / "best.jsonl").open("a") as f:
                for w in wins:
                    rs = [m["R"] for m in w["members"]]
                    b = max(w["members"], key=lambda m: m["R"])
                    f.write(json.dumps({"step": step + 1, "date": w["date"], "burn": w["burn"], "ml": w["ml"], "rs": rs,
                                        "R": b["R"], "malformed": b["malformed"], "rejected": b["rejected"], "calls": b["calls"],
                                        "call": b["call"], "trace": b["trace"], "names": w["names"],
                                        "prompt_len": w["prompt_len"], "seq": b["seq"], "mask": b["mask"]}, default=float) + "\n")
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        if not cfg.no_update and not cfg.no_save and ((step + 1) % cfg.save_every == 0 or step + 1 == cfg.steps):
            name = f"{cfg.tag}-step{step + 1:03d}"
            st = await (await tc.save_state_async(name, ttl_seconds=cfg.state_ttl_days * DAY)).result_async()
            sp = await (await tc.save_weights_for_sampler_async(name, ttl_seconds=cfg.sampler_ttl_days * DAY)).result_async()
            with ckpts.open("a") as f:
                f.write(json.dumps({"step": step + 1, "state": st.path, "sampler": sp.path}) + "\n")
            print(f"saved {name}: {sp.path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--burn", type=int, default=5)
    ap.add_argument("--ml-frac", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--loss", default="ppo")
    ap.add_argument("--max-tokens", type=int, default=agent.MAX_OUT, help="per turn, less what the context has no room for")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--lam-cal", type=float, default=LAMBDA_CAL, help="weight of reward.calibration_bonus in a look's R")
    ap.add_argument("--drill-groups", type=int, default=0, help="forecast-drill questions per step (data/drill/train.jsonl)")
    ap.add_argument("--drill-group", type=int, default=16, help="answers sampled per drill question")
    ap.add_argument("--kappa", type=float, default=KAPPA, help="drill advantage = kappa x (Brier reward - group mean)")
    ap.add_argument("--no-save", action="store_true", help="train but save no checkpoints (smoke test)")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--state-ttl-days", type=int, default=21)
    ap.add_argument("--sampler-ttl-days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--no-update", action="store_true", help="roll out and score only (smoke test); nothing is trained")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--news", action="store_true",
                    help="offer point-in-time news, article and tape (ml_research/simnews.py) with the live news prompt")
    ap.add_argument("--init", default="", help="training state (tinker://.../weights/...) to start from instead of rl1's last")
    ap.add_argument("--sampler", default="", help="with --no-update: sample from this sampler path (no training client)")
    ap.add_argument("--keep-best", action="store_true", help="save each group's best look with tokens and mask (best.jsonl)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
