"""Summarize a NAV re-test run: each model against cash, a passive basket and base, with paired tests.

    python analyze_nav.py oos1
"""

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import p1
from nav_runner import passive

ROOT = Path(__file__).resolve().parent


def sign_p(w, l):
    """Two-sided exact sign test."""
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def perm_p(diffs, reps=20000, seed=0):
    """Two-sided sign-flip permutation test on the mean of paired differences."""
    rng = random.Random(seed)
    obs = abs(sum(diffs))
    hits = sum(abs(sum(d if rng.random() < .5 else -d for d in diffs)) >= obs - 1e-12 for _ in range(reps))
    return (hits + 1) / (reps + 1)


def boot_ci(diffs, reps=20000, seed=1):
    """95% CI of the mean by cluster bootstrap over start dates (cases sharing a start are resampled together)."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for k, d in diffs:
        by[k].append(d)
    keys = list(by)
    means = []
    for _ in range(reps):
        xs = [x for k in (rng.choice(keys) for _ in keys) for x in by[k]]
        means.append(sum(xs) / len(xs))
    means.sort()
    return means[int(.025 * reps)], means[int(.975 * reps)]


def main(tag):
    rows = [json.loads(f.read_text()) for f in (ROOT / "runs/navtest" / tag).glob("*.json")]
    last = max(r["start"] for r in rows)
    bars = {k: [b for b in v if b["session"] <= "2026-10-12"] for k, v in p1.load_bars("ibkr").items()}
    cases = sorted({(r["start"], r["horizon"]) for r in rows})
    bench = {c: passive(bars, *c)["marked_return_pct"] for c in cases}
    spy = {c: passive(bars, *c, symbols=("SPY",))["marked_return_pct"] for c in cases}
    arms = sorted({r["arm"] for r in rows})
    # average over seeds within a case so each case counts once
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[r["arm"]][(r["start"], r["horizon"])].append(r)
    out = {"tag": tag, "cases": len(cases), "episodes": len(rows), "arms": {}}
    lines = [f"# NAV re-test {tag}: {len(rows)} episodes, {len(cases)} cases "
             f"(starts {cases[0][0]}..{last}, horizons {sorted({h for _, h in cases})})", ""]
    lines.append("| | mean marked % | mean liquidated % | realized $/ep | closed flat | still holding | FINAL / STEP / MALF / CTX | cost $ |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|")
    for name, vals in [("cash", {c: 0.0 for c in cases}), ("passive basket", bench), ("$2,500 of SPY, rest in cash", spy)]:
        lines.append(f"| {name} | {sum(vals.values()) / len(vals):+.2f} | | | | | | |")
    case_mean = {}
    for a in arms:
        rs = [r for c in cases for r in per[a][c]]
        cm = {c: sum(r["marked_return_pct"] for r in per[a][c]) / len(per[a][c]) for c in cases if per[a][c]}
        case_mean[a] = cm
        st = defaultdict(int)
        for r in rs:
            st[r["status"]] += 1
        info = {
            "episodes": len(rs),
            "mean_marked_pct": sum(r["marked_return_pct"] for r in rs) / len(rs),
            "mean_liquidated_pct": sum(r["liquidated_return_pct"] for r in rs) / len(rs),
            "mean_realized_usd": sum(r["realized_usd"] for r in rs) / len(rs),
            "closed_flat": sum(r["closed_flat"] for r in rs),
            "holding_at_end": sum(bool(r["residual_positions"]) for r in rs),
            "no_trade": sum(r["fills_buy"] + r["fills_sell"] == 0 for r in rs),
            "status": dict(st), "cost_usd_upper": sum(r["cost_usd_upper"] for r in rs),
            "evaluator_sessions": sum(r["evaluator_sessions"] for r in rs),
        }
        out["arms"][a] = info
        lines.append(f"| {a} | {info['mean_marked_pct']:+.2f} | {info['mean_liquidated_pct']:+.2f} | {info['mean_realized_usd']:+.1f} | "
                     f"{info['closed_flat']}/{len(rs)} | {info['holding_at_end']}/{len(rs)} | "
                     f"{st['FINAL']} / {st['STEP_LIMIT']} / {st['MALFORMED']} / {st['CONTEXT_BUDGET']} | {info['cost_usd_upper']:.2f} |")
    lines += ["", "Paired by case (seed-averaged), cluster bootstrap over start dates:", "",
              "| comparison | mean diff pp | wins / losses | sign p | permutation p | 95% CI |", "|---|---:|---|---:|---:|---|"]
    pairs = [(a, "passive basket") for a in arms] + [(a, "base") for a in arms if a != "base"] + [("nav9", "nav4")] * ("nav9" in arms and "nav4" in arms)
    out["pairs"] = {}
    for a, b in pairs:
        other = bench if b == "passive basket" else case_mean[b]
        ds = [(c, case_mean[a][c] - other[c]) for c in cases if c in case_mean[a] and c in other]
        d = [x for _, x in ds]
        w, l = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
        lo, hi = boot_ci([(c[0], x) for c, x in ds])
        res = {"mean_pp": sum(d) / len(d), "wins": w, "losses": l, "sign_p": sign_p(w, l), "perm_p": perm_p(d), "ci95": [lo, hi]}
        out["pairs"][f"{a} - {b}"] = res
        lines.append(f"| {a} - {b} | {res['mean_pp']:+.2f} | {w} / {l} | {res['sign_p']:.3f} | {res['perm_p']:.3f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "By horizon (mean marked %):", "", "| horizon | " + " | ".join(arms) + " | passive basket | $2,500 SPY |", "|---" * (len(arms) + 3) + "|"]
    for h in sorted({h for _, h in cases}):
        cs = [c for c in cases if c[1] == h]
        vals = [sum(case_mean[a][c] for c in cs if c in case_mean[a]) / len(cs) for a in arms]
        lines.append(f"| {h} | " + " | ".join(f"{v:+.2f}" for v in vals) + f" | {sum(bench[c] for c in cs) / len(cs):+.2f} | {sum(spy[c] for c in cs) / len(cs):+.2f} |")
    (ROOT / "runs/navtest" / f"{tag}_summary.json").write_text(json.dumps(out, indent=1))
    text = "\n".join(lines)
    (ROOT / "runs/navtest" / f"{tag}_summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
