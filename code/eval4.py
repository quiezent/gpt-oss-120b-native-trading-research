"""Paired evaluation of multi-look policies (rl3 vs rl4) in rl4's intraday simulator, on held-out sessions (Jul-Oct 2026).

For each validation date: one burn-in book (the first model's cheap decisions, under the date's goal gap), then N
episodes per model from that same book (train4.episodes: intrasim.LOOKS looks, the forecast tool on every other date),
each scored by intrasim.Ledger plus train4's penalties, and a scripted hold episode (empty submits at every look). Reports
per-model means, the paired per-date difference against the first model (raw and winsorized at +/-10) with its t-stat, and
the process and concentration metrics of the acceptance criteria (analysis session, 24 Oct; rl4_prereg.md).

    python eval4.py --models base=base rl4=<sampler> --dates 48 --n 8 --tag val-rl4     # base: gpt-oss-120b itself
"""

import argparse
import asyncio
import json
import random
import statistics as stats
from pathlib import Path

import tinker

import agent
import eval2
import intrasim
import simgw
import train2
import train4
import trader

ROOT = Path(__file__).resolve().parent
P = train4.PRICE


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))] if xs else None


def date_bootstrap(rows, a, b, n=2000, seed=0):
    """R3's paired bootstraps over dates (one-sided, 95%): is b's share of looks with a 75%+ position unprotected below
    half of a's, and is b's p95 worst look-to-look loss not worse than a's (5th percentile of worst_step_pp, higher = better)?"""
    rng = random.Random(seed)
    k = len(rows)
    u = lambda nm, r: [lk["book"]["unprotected_75"] for m in r["results"][nm] for lk in m["looks"]]
    w = lambda nm, r: [m["worst_step_pp"] for m in r["results"][nm]]
    half, worse = 0, 0
    for _ in range(n):
        pick = [rows[rng.randrange(k)] for _ in range(k)]
        ua, ub = sum((x for r in pick for x in u(a, r)), 0.0), sum((x for r in pick for x in u(b, r)), 0.0)
        na, nb = sum(len(u(a, r)) for r in pick) or 1, sum(len(u(b, r)) for r in pick) or 1
        half += ub / nb < 0.5 * ua / na
        worse += q([x for r in pick for x in w(b, r)], 0.05) >= q([x for r in pick for x in w(a, r)], 0.05)
    return {"unprotected_75_below_half": half / n >= 0.95, "share_below_half": half / n,
            "worst_step_not_worse": worse / n >= 0.95, "share_not_worse": worse / n}


def metrics(members):
    """The acceptance criteria's process and concentration metrics over episodes (rl4_prereg.md)."""
    lks = [lk for m in members for lk in m["looks"]]
    n, nl = len(members), max(1, len(lks))
    orders = sum(lk["orders"] for lk in lks) or 1
    used = [t for lk in lks for t in lk["trace"] if t.get("tool") and t["tool"] != "submit"]
    return {
        "episodes": n, "looks": len(lks),
        "R": stats.mean(m["R"] for m in members), "R_winsorized10": stats.mean(max(-10, min(10, m["R"])) for m in members),
        "R_trade": stats.mean(m["R_trade"] for m in members), "penalty": stats.mean(m["penalty"] for m in members),
        "book_pp": stats.mean(m["book_pp"] for m in members), "hedge_pp": stats.mean(m["hedge_pp"] for m in members),
        "risk_intra_pp": stats.mean(m["risk_intra_pp"] for m in members), "risk_close_pp": stats.mean(m["risk_close_pp"] for m in members),
        "research_calls": len(used) / nl, "forecast_calls": sum(t["tool"] == "forecast" for t in used) / nl,
        "stop_type_share": sum(lk["stop_type"] for lk in lks) / orders,
        "gross_close": stats.mean(m["gross_close"] for m in members), "orders_per_look": sum(lk["orders"] for lk in lks) / nl,
        "entries_per_look": sum(lk["entries"] for lk in lks) / nl, "overnight_pp": stats.mean(m["overnight_pp"] for m in members),
        "look_gross": stats.mean(lk["gross"] for lk in lks),
        "orders_ex_protective_per_look": sum(lk["orders_ex_protective"] for lk in lks) / nl,
        "protective_per_held_position": sum(lk["protective"] for lk in lks) / max(1, sum(lk["held"] for lk in lks)),
        "no_decision_share": sum(lk["call"] is None or lk["empty"] for lk in lks) / nl,
        "forced_empty_share": sum(lk["forced"] and lk["empty"] for lk in lks) / nl,
        "notes_share": sum(lk["notes"] for lk in lks) / nl, "no_probs_share": sum(lk["no_probs"] for lk in lks) / orders,
        "unresearched_per_look": sum(lk["unresearched"] for lk in lks) / nl,
        "carried_unprotected_share": sum(m["carried_unprotected"] for m in members) / max(1, sum(m["carried"] for m in members)),
        "R_pnl": stats.mean(m["book_pp"] - m["hedge_pp"] for m in members),
        "malformed_looks": sum(lk["call"] is None for lk in lks) / nl,
        "context_full": sum(any(t.get("error") == "context full" for t in lk["trace"]) for lk in lks) / nl,
        "bad_turns_per_look": sum(lk["bad"] for lk in lks) / nl,
        "forced_submit": sum(any("research limit" in str(t.get("result", "")) for t in lk["trace"]) for lk in lks) / nl,
        "rejected_per_look": sum(m["rejected"] for m in members) / nl,
        "m1_big_per_look": sum(lk["m1_big"] for lk in lks) / nl,
        "under_1k_share": sum(lk["under_1k"] for lk in lks) / orders,
        "entries_no_stop_share": sum(lk["entries_no_stop"] for lk in lks) / max(1, sum(lk["entries"] for lk in lks)),
        "protective_per_look": sum(lk["protective"] for lk in lks) / nl,
        "look_unprotected_75": sum(lk["book"]["unprotected_75"] for lk in lks) / nl,
        "look_top_weight_median": q([lk["book"]["top_weight"] for lk in lks], 0.5),
        "close_unprotected_75": sum(m["at_close"]["unprotected_75"] for m in members) / n,
        "close_top_weight_median": q([m["at_close"]["top_weight"] for m in members], 0.5),
        "worst_step_pp_p05": q([m["worst_step_pp"] for m in members], 0.05),
        "look_tokens": sum(lk["cost"]["sampled"] for lk in lks) / nl,
    }


async def main_async(cfg):
    data = trader.Data()
    pre = simgw.Precomputed(data)
    intra = intrasim.Intra(data)
    import forecast as fc
    prov = fc.sim_provider(data)
    pool = [d for d in train4.pool(data, train4.VAL_FIRST, train4.VAL_LAST) if intra.has(d)]
    rng = random.Random(cfg.seed)
    dates = sorted(rng.sample(pool, cfg.dates))
    burns = [rng.randint(1, 5) for _ in dates]
    gaps = [round(rng.uniform(train4.LOSS_LO, train4.LOSS_HI), 4) for _ in dates]
    service = tinker.ServiceClient(project_id=train4.PROJECT_ID, user_metadata={"purpose": f"eval4:{cfg.tag}"})
    models = dict(m.split("=", 1) for m in cfg.models)
    clients = {name: service.create_sampling_client(base_model=train2.BASE) if path == "base"   # gpt-oss-120b, as live.py
               else service.create_sampling_client(model_path=path) for name, path in models.items()}
    ref = clients[next(iter(clients))]
    out = ROOT / "runs" / "eval4" / cfg.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(cfg), "dates_used": [data.dates[d] for d in dates]}, indent=1))

    async def one(i, d, k, gap):
        w0, memo0, used = await train4.burn(ref, data, d, k, round(train4.SIZE * (1 - gap), 2), cfg)
        base = intrasim.IntraWorld(w0, intra)
        ml = i % 2 == 1
        got = await asyncio.gather(*(train4.episodes(sc, data, pre, intra, prov, d, ml, base, memo0, cfg.n, cfg)
                                     for sc in clients.values()))
        res = dict(zip(clients, got))
        hold = await train4.episodes(None, data, pre, intra, prov, d, ml, base, memo0, 1, cfg,
                                     script=lambda hhmm: {"place": [], "cancel": [], "notes": ""})
        return {"date": data.dates[d], "burn": k, "ml": ml, "loss": gap, "burn_usd": train4.usd(used, update=False),
                "hold_R": hold[0]["R"], "results": res}

    rows = await asyncio.gather(*(one(i, d, k, gp) for i, (d, k, gp) in enumerate(zip(dates, burns, gaps))))
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**r, "results": {n: [{k: v for k, v in m.items() if k != "looks"} | {"looks": [
                {k: v for k, v in lk.items() if k not in ("seq", "mask", "lps")} for lk in m["looks"]]} for m in ms]
                for n, ms in r["results"].items()}}, default=float) + "\n")
    names = list(models)
    summ = {"dates": len(rows), "n": cfg.n, "hold_R": stats.mean(r["hold_R"] for r in rows)}
    for name in names:
        mem = [m for r in rows for m in r["results"][name]]
        summ[name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics(mem).items()}
        summ[name]["R_minus_hold"] = round(stats.mean(stats.mean(m["R"] for m in r["results"][name]) - r["hold_R"] for r in rows), 4)
        summ[name]["usd"] = round(sum(train4.usd(lk["cost"], update=False) for m in mem for lk in m["looks"]), 4)
    a = names[0]                    # base, the comparison model (rl4_acceptance_criteria.md v1.1, R2b-R4)
    pnl = lambda m: m["book_pp"] - m["hedge_pp"]           # R2 scores the reward's P&L part only
    sd_base = stats.pstdev([pnl(m) for r in rows for m in r["results"][a]])
    summ["R2b bar (-0.3 x base's per-episode P&L sd)"] = round(-0.3 * sd_base, 4)
    for b in names[1:]:
        per = lambda nm, f: [stats.mean(f(pnl(m)) for m in r["results"][nm]) for r in rows]
        summ[f"R2b paired P&L {b} - {a}"] = eval2.paired(per(a, lambda x: x), per(b, lambda x: x))
        summ[f"R2b paired P&L winsorized10 {b} - {a}"] = eval2.paired(per(a, lambda x: max(-10, min(10, x))),
                                                                      per(b, lambda x: max(-10, min(10, x))))
        summ[f"paired full R {b} - {a}"] = eval2.paired([stats.mean(m["R"] for m in r["results"][a]) for r in rows],
                                                        [stats.mean(m["R"] for m in r["results"][b]) for r in rows])
        A, B = summ[a], summ[b]
        within = lambda k: bool(A[k]) and abs(B[k] / A[k] - 1) <= 0.30
        boot = date_bootstrap(rows, a, b)
        summ[f"R3 bootstrap {b} vs {a}"] = boot
        summ[f"criteria {b} vs {a}"] = {
            "R2b P&L not below -0.3 sd (reported)": summ[f"R2b paired P&L {b} - {a}"].get("mean", -9) >= -0.3 * sd_base,
            "R2b P&L winsorized not below -0.3 sd (reported)": summ[f"R2b paired P&L winsorized10 {b} - {a}"].get("mean", -9) >= -0.3 * sd_base,
            "R2b research calls within 30%": within("research_calls"),
            "R2b orders per look (excluding protective stops) within 30%": within("orders_ex_protective_per_look"),
            "R2b no-decision + context full not 3 pts above": B["no_decision_share"] + B["context_full"]
                                                            <= A["no_decision_share"] + A["context_full"] + 0.03,
            "R2b rejected per look not 2 pts above": B["rejected_per_look"] <= A["rejected_per_look"] + 0.02,
            "R2b malformed <= 3%": B["malformed_looks"] <= 0.03,
            "R3 75%+ unprotected below half of base's (bootstrap) and <= 10%": boot["unprotected_75_below_half"] and B["look_unprotected_75"] <= 0.10,
            "R3 median largest position <= base's and <= 35%": B["look_top_weight_median"] <= min(A["look_top_weight_median"], 0.35),
            "R3 p95 worst look-to-look loss not worse than base's (bootstrap)": boot["worst_step_not_worse"],
            "R3 carried positions unprotected <= 50%": B["carried_unprotected_share"] <= 0.50,
            "R4 malformed <= base + 3 pts and <= 5%": B["malformed_looks"] <= min(A["malformed_looks"] + 0.03, 0.05),
            "R4 forced-empty <= base + 3 pts": B["forced_empty_share"] <= A["forced_empty_share"] + 0.03,
            "R4 notes in >= 90% of looks": B["notes_share"] >= 0.90,
        }
    summ["cost_usd"] = round(sum(r["burn_usd"] for r in rows) + sum(summ[n]["usd"] for n in names), 2)
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=sampler_path; the first also builds the burn-in books")
    ap.add_argument("--dates", type=int, default=24)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--looks", nargs="+", default=intrasim.LOOKS)
    ap.add_argument("--forward", type=float, default=intrasim.FORWARD)
    ap.add_argument("--max-tokens", type=int, default=agent.MAX_OUT)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--tag", required=True)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
