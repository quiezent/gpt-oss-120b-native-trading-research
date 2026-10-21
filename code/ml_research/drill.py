"""Forecast drill: binary questions on US stocks and ETFs from the training simulator's point-in-time data, for
calibration training (the Tinker cookbook's forecasting recipe: one answer, Brier reward 1-(p-y)^2, malformed = 0).

Each question is asked at 09:00 ET on session d (before the open) and resolves from daily bars of sessions d..d+4:
  R range:      session d's high-low range exceeds K% of the previous close (K near the stock's 20-session median range)
  S stop touch: the price trades at or below x% under the previous close (or at or above x% over it) in d..d+4
  L limit fill: a DAY buy limit y% under the previous close (or a sell limit y% over) fills in session d (gaps count)
  D direction:  the close of d+4 is more than 0.2% above (or below) the previous close
Levels are multiples of the stock's own recent range or volatility (MULT), so base rates spread out.
Reference material: the last 10 daily bars, forecast.sim_provider's row (as of the close of d-1), its SPY and macro
lines, and up to 5 headlines visible at 09:00 ET (ml_research/simnews.py). The outcome is stored in its own field;
prompt() never reads it.

    python ml_research/drill.py build        # data/drill/train.jsonl, val.jsonl
    python ml_research/drill.py baselines    # runs/ml/research/drill_baselines.json, .md
    python ml_research/drill.py leak         # perturb bars from session d on for 50 questions; prompts must not change

Trainer API: SYSTEM, TYPES, load(split), prompt(q), parse_forecast(text), score(p, outcome).
"""

import html
import json
import math
import random
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ml_research import simnews  # noqa: E402

OUT = ROOT / "data" / "drill"
RES = ROOT / "runs" / "ml" / "research"
TYPES = ("R", "S", "L", "D")
# train2.TRAIN_FIRST..TRAIN_LAST and eval2.VAL_FIRST..VAL_LAST (not imported: train2 pulls in torch and may import us)
SPLITS = {"train": ("20250103", "20251212", 4000), "val": ("20260102", "20260514", 800)}
TRAIN_RESOLVE_BY = "20251219"
SEED = 20261018
HORIZON = {"R": 0, "S": 4, "L": 0, "D": 4}                # sessions after d that a question covers
MULT = {"R": (0.7, 0.85, 1.0, 1.2, 1.5),                  # x the 20-session median range
        "S": (0.3, 0.5, 0.75, 1.0, 1.4),                  # x the 5-session volatility (20-session daily vol x sqrt 5)
        "L": (0.1, 0.2, 0.35, 0.5, 0.75),                 # x the 20-session median range
        "D": (0.2,)}                                      # fixed 0.2% move
NEWS_DAYS = 30

SYSTEM = ("You forecast price events for US stocks and ETFs. Prices are daily bars of regular trading hours, adjusted "
          "for splits and dividends. Forecast lines come from our models, as of the previous close: range1 = expected "
          "high-low range of the next session, % of the last close, with its 10-90% range; absr1 = expected size of the "
          "next close-to-close move, %; vol20 = expected annualized volatility over the next 20 sessions, %; SPY "
          "vol20/vol5 = expected annualized SPY volatility over the next 20/5 sessions. No model of ours predicts "
          "direction. Give calibrated probabilities: you are scored by 1 - (p - outcome)^2.")

PROMPT = ("Forecast whether this will happen using information available through {asof}.\n\nReference material:\n{ref}\n\n"
          "Question:\n{question}\n\nOutput only the probability of YES as a number between 0 and 1.")


def prompt(q):
    return PROMPT.format(asof=q["asof"], ref=q["reference"], question=q["question"])


def load(split):
    with open(OUT / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- parsing and scoring (tinker_cookbook forecasting/env.py)
_FINAL_FORECAST_RE = re.compile(
    r"""
    \A\s*
    (?:\*{1,2}|_{1,2})?
    (?:(?:(?:final\s+)?(?:answer|forecast)|probability(?:\s+of\s+yes)?)\s*[:=]\s*)?
    (?:\*{1,2}|_{1,2})?
    (?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
    \s*(?P<percent>%?)
    (?:\*{1,2}|_{1,2})?
    [.!]?
    (?:\*{1,2}|_{1,2})?
    \s*\Z
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_forecast(response):
    """A probability from the response's final non-empty line (0.37 or 37%), else None."""
    lines = [line for line in str(response).splitlines() if line.strip()]
    if not lines:
        return None
    m = _FINAL_FORECAST_RE.fullmatch(lines[-1])
    if m is None:
        return None
    p = float(m.group("value")) / (100.0 if m.group("percent") else 1.0)
    return p if 0.0 <= p <= 1.0 else None


def brier_reward(p, outcome):
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must be finite and in [0, 1], got {p}")
    if isinstance(outcome, bool) or outcome not in (0, 1):
        raise ValueError(f"outcome must be 0 or 1, got {outcome}")
    return 1.0 - (p - outcome) ** 2


def score(p, outcome):
    """Brier reward; a malformed answer (p None) scores 0."""
    return 0.0 if p is None else brier_reward(p, outcome)


# ---------------------------------------------------------------- questions
_NAMES = None


def _name(s):
    global _NAMES
    if _NAMES is None:
        _NAMES = simnews.company_names()
    n = _NAMES.get(s)
    return f"{s} ({n})" if n and n != s else s


def _day(data, i):
    x = date.fromisoformat(data.iso(i))
    return f"{x:%a} {x.isoformat()}"


def _nice(x):
    return max(0.1, round(x, 1) if x < 5 else round(x * 2) / 2)


def _f(x):
    return "na" if x is None else f"{x:g}"


def reference(data, fc, d, s, archive=None):
    """The reference text for symbol s before the open of session d, from bars through d-1, the forecast row
    fc = forecast.sim_provider(data)(d), and headlines strictly before 09:00 ET of d. None if s has no forecast row."""
    if not fc or s not in fc[1]["rows"]:
        return None
    as_of, F = fc
    j, t = data.k[s], d - 1
    o, h, l, c, v = (a[j] for a in (data.o, data.h, data.l, data.c, data.v))
    L = [f"Information available through 09:00 ET on {_day(data, d)} (before the open).",
         f"{_name(s)}, last 10 daily bars (chg% = close vs previous close; range% = high-low as % of previous close; "
         "vol = volume in millions):", "date open high low close chg% range% vol"]
    for i in range(t - 9, t + 1):
        L.append(f"{data.iso(i)} {o[i]:.2f} {h[i]:.2f} {l[i]:.2f} {c[i]:.2f} {100 * (c[i] / c[i - 1] - 1):+.2f} "
                 f"{100 * (h[i] - l[i]) / c[i - 1]:.2f} {v[i] / 1e6:.2f}")
    r = F["rows"][s]
    note = "; ".join(x for x in (r.get("note") or "").split("; ") if x and not x.startswith("no implied vol"))
    band = f" (10-90 {r['range1_lo']:g}-{r['range1_hi']:g})" if r.get("range1_lo") is not None and r.get("range1_hi") is not None else ""
    L.append(f"Forecast as of the close of {as_of[:4]}-{as_of[4:6]}-{as_of[6:]}: range1 {_f(r.get('range1_pct'))}%{band}, "
             f"absr1 {_f(r.get('absr1_pct'))}%, vol20 {_f(r.get('vol20_pct'))}%" + (f". Note: {note}" if note else ""))
    m, k = F["market"], data.k["SPY"]
    sc = data.c[k]
    spy = f"SPY close {sc[t]:.2f} ({100 * (sc[t] / sc[t - 1] - 1):+.1f}% last session, {100 * (sc[t] / sc[t - 5] - 1):+.1f}% over 5)"
    if m.get("spy_vol20_pct"):
        rng = lambda key: "-".join(f"{x:g}" for x in (m.get(key) or []) if x is not None) or "na"
        L.append(f"Market: SPY vol20 {m['spy_vol20_pct']:g}% (10-90 {rng('spy_vol20_10_90_pct')}), vol5 "
                 f"{_f(m.get('spy_vol5_pct'))}% (10-90 {rng('spy_vol5_10_90_pct')}), typical {_f(m.get('typical_spy_vol_pct'))}%; {spy}.")
    else:
        L.append(f"Market: no SPY volatility forecast; {spy}.")
    if m.get("macro"):
        L.append(m["macro"][0].upper() + m["macro"][1:])
    gw = simnews.NewsSimGateway(data, d, None, news_archive=archive)
    since = gw.cutoff - NEWS_DAYS * 86400
    news = [x for x in gw.news(s, limit=5) if simnews.ts(x["time"]) >= since]
    L.append(f"Headlines for {s}, newest first (last {NEWS_DAYS} days):" if news else f"Headlines for {s}: none in the last {NEWS_DAYS} days.")
    for x in news:
        L.append(f"{simnews._et(simnews.ts(x['time']))} {x['provider']}: {html.unescape(x['headline']).lstrip('!')}")
    return "\n".join(L)


def make(data, fc, d, s, typ, side, m, archive=None):
    """One question dict (with its outcome) for symbol s, session d, type typ, side 'down'/'up' ('' for R), level
    multiplier m; None if the data are missing."""
    H = HORIZON[typ]
    j, t = data.k[s], d - 1
    if d + H >= len(data.dates) or t < 21:
        return None
    o, h, l, c = (a[j] for a in (data.o, data.h, data.l, data.c))
    past, fut = slice(t - 20, t + 1), slice(d, d + H + 1)
    if not (np.isfinite(h[past]).all() and np.isfinite(l[past]).all() and np.isfinite(c[past]).all()
            and np.isfinite(h[fut]).all() and np.isfinite(l[fut]).all() and np.isfinite(c[fut]).all()):
        return None
    ref = reference(data, fc, d, s, archive)
    if ref is None:
        return None
    pc = float(c[t])
    med = float(np.median((h[t - 19:t + 1] - l[t - 19:t + 1]) / c[t - 20:t])) * 100
    sig5 = float(np.std(data.lr[j, t - 19:t + 1], ddof=1)) * 100 * math.sqrt(5)
    D0, DH, sym = _day(data, d), _day(data, d + H), _name(s)
    sgn = -1 if side == "down" else 1
    if typ == "R":
        lev = _nice(m * med)
        val = 100 * (h[d] - l[d]) / pc
        y = val > lev
        q = (f"Will the high-low range of {sym} on {D0} exceed {lev:g}% of the previous close of {pc:.2f}? Resolves YES "
             f"if (session high - session low) / {pc:.2f} is more than {lev:g}%.")
    elif typ in ("S", "L"):
        lev = _nice(m * (sig5 if typ == "S" else med))
        px = round(pc * (1 + sgn * lev / 100), 2)
        word, rel = ("below", "at or below") if side == "down" else ("above", "at or above")
        if typ == "S":
            ext = float(np.min(l[fut])) if side == "down" else float(np.max(h[fut]))
            y = ext <= px if side == "down" else ext >= px
            q = (f"Will {sym} trade {rel} {px:.2f} ({lev:g}% {word} the previous close of {pc:.2f}) at any time during "
                 f"regular hours from the open on {D0} through the close on {DH} (5 sessions)?")
        else:
            ext = float(l[d]) if side == "down" else float(h[d])
            y = ext <= px if side == "down" else ext >= px
            q = (f"A DAY {'buy' if side == 'down' else 'sell'} limit order for {s} at {px:.2f} ({lev:g}% {word} the "
                 f"previous close of {pc:.2f}) is placed before the open on {D0}. Will it fill? Resolves YES if the "
                 f"session's {'low' if side == 'down' else 'high'} is {rel} {px:.2f} (an open through the limit counts).")
        val = 100 * (ext / pc - 1)
    else:
        lev = m
        val = 100 * (c[d + H] / pc - 1)
        y = val > lev if side == "up" else val < -lev
        q = (f"Will {sym} close on {DH} more than {lev:g}% {'above' if side == 'up' else 'below'} the previous close of "
             f"{pc:.2f} (the close of {_day(data, t)}), that is, {'above' if side == 'up' else 'below'} "
             f"{pc * (1 + sgn * lev / 100):.2f}? That is 5 sessions: {D0} through {DH}.")
    r = fc[1]["rows"][s]
    return {"date": data.dates[d], "symbol": s, "type": typ, "side": side, "m": m, "level_pct": lev, "prev_close": round(pc, 4),
            "resolves": data.dates[d + H], "asof": f"09:00 ET on {D0}, before the open", "reference": ref, "question": q,
            "ml": {"range1": r.get("range1_pct"), "lo": r.get("range1_lo"), "hi": r.get("range1_hi")},
            "outcome": int(bool(y)), "realized_pct": round(float(val), 4)}


def _setup():
    import forecast
    import trader
    data = trader.Data(last="20261015")
    return data, forecast.sim_provider(data), simnews.archive()


def build(seed=SEED):
    data, fc_at, A = _setup()
    rng = random.Random(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    for split, (first, last, n) in SPLITS.items():
        days = [i for i, x in enumerate(data.dates) if first <= x <= last]
        cache, out, seen = {}, [], set()
        while len(out) < n:
            typ = TYPES[len(out) % len(TYPES)]
            d = rng.choice(days)
            if d not in cache:
                cache[d] = (fc_at(d), data.table_rows(d)[0])
            fc, table = cache[d]
            s = rng.choice(table) if rng.random() < 0.5 else rng.choice(data.sym)
            side = "" if typ == "R" else rng.choice(("down", "up"))
            m = rng.choice(MULT[typ])
            if (d, s, typ) in seen:
                continue
            q = make(data, fc, d, s, typ, side, m, A)
            if q is None:
                continue
            if split == "train":
                assert q["resolves"] <= TRAIN_RESOLVE_BY, q
            seen.add((d, s, typ))
            out.append({"id": f"{split}-{len(out):05d}", "split": split, "in_table": s in table, **q})
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for q in out:
                f.write(json.dumps(q) + "\n")
        print(split, len(out), {t: round(np.mean([q["outcome"] for q in out if q["type"] == t]), 3) for t in TYPES})


# ---------------------------------------------------------------- baselines
def bucket(q):
    return f"{q['type']} {q['side'] or '-'} {q['m']:g}"


def _x(q, own=False):
    """Features of the ML mapping; own=True adds ln(m), the level as a multiple of the stock's own recent range or vol."""
    u = math.log(q["level_pct"] / q["ml"]["range1"])
    w = math.log(q["ml"]["hi"] / q["ml"]["lo"]) / 2.563          # 10-90 spread -> a log-scale sigma
    return [u, u / w, w, 1.0 if q["side"] == "up" else 0.0] + ([math.log(q["m"])] if own else [])


def baselines():
    from sklearn.linear_model import LogisticRegression
    import trader
    tr, va = load("train"), load("val")
    rate = lambda qs: sum(q["outcome"] for q in qs) / len(qs)
    by_t = {t: rate([q for q in tr if q["type"] == t]) for t in TYPES}
    groups = {}
    for q in tr:
        groups.setdefault(bucket(q), []).append(q)
    by_b = {k: rate(v) for k, v in sorted(groups.items())}
    ok = lambda q: all((q["ml"].get(k) or 0) > 0 for k in ("range1", "lo", "hi"))     # Jan 2025 rows have no 10-90 range
    fits, fits2 = {}, {}
    for t in "RSL":
        qs = [q for q in tr if q["type"] == t and ok(q)]
        y = [q["outcome"] for q in qs]
        fits[t] = LogisticRegression(C=10.0, max_iter=1000).fit(np.array([_x(q) for q in qs]), y)
        fits2[t] = LogisticRegression(C=10.0, max_iter=1000).fit(np.array([_x(q, True) for q in qs]), y)
    ml = lambda F, own: lambda q: (float(F[q["type"]].predict_proba([_x(q, own)])[0, 1]) if q["type"] in F and ok(q)
                                   else by_b.get(bucket(q), by_t[q["type"]]))
    preds = {"always_0.5": lambda q: 0.5, "type_base_rate": lambda q: by_t[q["type"]],
             "type_level_base_rate": lambda q: by_b.get(bucket(q), by_t[q["type"]]), "our_ml": ml(fits, False),
             "our_ml_plus_level": ml(fits2, True)}
    table = {}
    for split, qs in (("val", va), ("train", tr)):
        table[split] = {}
        for name, f in preds.items():
            rw = {t: float(np.mean([score(f(q), q["outcome"]) for q in qs if q["type"] == t])) for t in TYPES}
            rw["all"] = float(np.mean([score(f(q), q["outcome"]) for q in qs]))
            table[split][name] = {k: round(v, 4) for k, v in rw.items()}
    counts = {s: {t: sum(q["type"] == t for q in qs) for t in TYPES} | {"all": len(qs)} for s, qs in (("train", tr), ("val", va))}
    rates = {s: {t: round(rate([q for q in qs if q["type"] == t]), 4) for t in TYPES} for s, qs in (("train", tr), ("val", va))}
    vb = {}
    for q in va:
        vb.setdefault(bucket(q), []).append(q)
    Rq = [q for q in va if q["type"] == "R" and ok(q)]
    yv, fv = np.array([q["realized_pct"] for q in Rq]), np.array([q["ml"]["range1"] for q in Rq])
    check = {"n": len(Rq), "r2": round(float(1 - np.sum((yv - fv) ** 2) / np.sum((yv - yv.mean()) ** 2)), 3),
             "cover_10_90": round(float(np.mean([q["ml"]["lo"] <= q["realized_pct"] <= q["ml"]["hi"] for q in Rq])), 3)}
    tok = trader.tokenizer()
    ntok = [len(tok.encode(SYSTEM + "\n" + prompt(q))) for q in va]
    res = {"counts": counts, "base_rates": rates, "range1_check_val_R": check,
           "bucket_base_rates": {k: {"train": round(v, 4), "n_train": len(groups[k]),
                                     "val": round(rate(vb[k]), 4) if k in vb else None, "n_val": len(vb.get(k, []))}
                                 for k, v in by_b.items()},
           "brier_reward": table, "in_table_share": {s: round(np.mean([q["in_table"] for q in qs]), 3) for s, qs in (("train", tr), ("val", va))},
           "prompt_tokens_val": {"mean": round(float(np.mean(ntok))), "max": int(max(ntok)), "p95": int(np.quantile(ntok, .95))},
           "ml_mapping": {t: {"features": ["ln(level/range1)", "ln(level/range1)/s", "s = ln(range1_hi/range1_lo)/2.563", "side up"],
                              "coef": [round(float(x), 4) for x in fits[t].coef_[0]], "intercept": round(float(fits[t].intercept_[0]), 4)}
                          for t in fits}}
    (RES / "drill_baselines.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    _md(res, va)
    print(json.dumps(res["brier_reward"], indent=1))
    return res


def _md(res, va):
    T = res["brier_reward"]
    names = {"always_0.5": "always 0.5", "type_base_rate": "training base rate, per type",
             "type_level_base_rate": "training base rate, per type and level bucket",
             "our_ml": "our ML (R, S, L: logistic on range1 10/50/90; D: bucket base rate)",
             "our_ml_plus_level": "our ML plus the level bucket (adds ln of the level's multiple of the stock's own range)"}
    L = ["# Forecast drill: question set and trivial baselines", "",
         "Built by `ml_research/drill.py` (no paid calls). Questions: `data/drill/train.jsonl`, `data/drill/val.jsonl`. "
         "Score: Brier reward 1-(p-y)^2 (higher is better; always 0.5 scores 0.75; a malformed answer scores 0).", "",
         "Types: **R** session d's high-low range > K% of the previous close; **S** stop touch x% below (or above) the "
         "previous close within sessions d..d+4; **L** a DAY limit y% below (buy) or above (sell) the previous close fills "
         "in session d; **D** the close of d+4 is more than 0.2% above (or below) the previous close. Questions are asked "
         "at 09:00 ET on session d; reference material uses bars through d-1, the forecast row as of the close of d-1, and "
         "headlines before 09:00 ET.", "",
         "Splits: train sessions 2025-01-03..2025-12-12 (all resolve by 2025-12-19), validation 2026-01-02..2026-05-14. "
         f"Symbols: half from the look's market table (share in table: train {res['in_table_share']['train']}, "
         f"val {res['in_table_share']['val']}), half drawn from the 523-symbol universe; ETFs included.", "",
         "## Counts and base rates", "", "| split | R | S | L | D | all |", "|---|---|---|---|---|---|"]
    for s in ("train", "val"):
        c, r = res["counts"][s], res["base_rates"][s]
        L.append(f"| {s} n | {c['R']} | {c['S']} | {c['L']} | {c['D']} | {c['all']} |")
        L.append(f"| {s} YES rate | {r['R']:.3f} | {r['S']:.3f} | {r['L']:.3f} | {r['D']:.3f} | |")
    L += ["", "## Brier reward", ""]
    for s in ("val", "train"):
        L += [f"**{'Validation' if s == 'val' else 'Train (in-sample, for reference)'}**", "",
              "| baseline | R | S | L | D | all |", "|---|---|---|---|---|---|"]
        for k, n in names.items():
            x = T[s][k]
            L.append(f"| {n} | {x['R']:.4f} | {x['S']:.4f} | {x['L']:.4f} | {x['D']:.4f} | {x['all']:.4f} |")
        L.append("")
    L += ["## Base rate per level bucket (type, side, multiplier)", "",
          "R and L levels are multiples of the 20-session median range; S levels of the 5-session volatility "
          "(20-session daily vol x sqrt 5); D is fixed at 0.2%.", "", "| bucket | train | n | val | n |", "|---|---|---|---|---|"]
    for k, v in res["bucket_base_rates"].items():
        L.append(f"| {k} | {v['train']:.3f} | {v['n_train']} | {'na' if v['val'] is None else format(v['val'], '.3f')} | {v['n_val']} |")
    c = res["range1_check_val_R"]
    L += ["", "## Checks", "",
          f"- range1 vs the realized session range on the {c['n']} validation R questions: R2 {c['r2']}, 10-90% range held "
          f"{c['cover_10_90']}, close to forecast.py's track record (R2 0.37, 81%; 200 questions are noisy), not the jump a "
          "row that saw session d would show.",
          "- forecast.sim_provider serves column d-1: vol2, regime and cost read the column of the close of d-1; events "
          "count filings before 09:00 ET of d and read its range1 grid at d-1; the macro line is as of 09:00 ET with a "
          "60-minute value delay."]
    lk = RES / "drill_leak.json"
    if lk.exists():
        x = json.loads(lk.read_text(encoding="utf-8"))
        L.append(f"- Leak test (`drill.py leak`): on {x['questions']} questions, every bar from session d on was multiplied by "
                 f"random factors in [0.7, 1.3] and the question rebuilt: the prompt was unchanged in {x['prompt_unchanged']} "
                 f"of {x['questions']} while the perturbation changed {x['outcome_changed_by_perturbation']} outcomes; flipping "
                 f"the stored outcome never changed prompt(); the forecast row was as of the close of d-1 in "
                 f"{x['forecast_row_is_close_of_d_minus_1']} of {x['questions']}.")
    p = res["prompt_tokens_val"]
    L += ["", f"Prompt length (system + user, gpt-oss tokenizer, validation): mean {p['mean']}, p95 {p['p95']}, max {p['max']} tokens.", "",
          "## Two rendered examples (validation)", ""]
    ex = [next(q for q in va if q["type"] == "S"), next(q for q in va if q["type"] == "L")]
    for q in ex:
        L += [f"`{q['id']}` type {q['type']}, outcome {q['outcome']} (stored apart from the prompt):", "", "```text", prompt(q), "```", ""]
    L += ["System prompt:", "", "```text", SYSTEM, "```", ""]
    (RES / "drill_baselines.md").write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------- leak test
def leak(n=50, seed=1):
    """For n questions: perturb every bar from session d on (o, h, l, c, v x independent factors in [0.7, 1.3]), rebuild
    the question on the perturbed data, and require the asof, reference and question text to be unchanged; also flip the
    stored outcome and require prompt() to be unchanged."""
    import copy
    import forecast
    data, _, A = _setup()
    qs = random.Random(seed).sample(load("train") + load("val"), n)
    g = np.random.default_rng(seed)
    same = flipped = asof_ok = 0
    bad = []
    for q in qs:
        d = data.ix[q["date"]]
        D2 = copy.copy(data)
        for k in ("o", "h", "l", "c", "v"):
            a = getattr(data, k).copy()
            a[:, d:] *= g.uniform(0.7, 1.3, a[:, d:].shape)
            setattr(D2, k, a)
        with np.errstate(divide="ignore", invalid="ignore"):
            D2.lr = np.diff(np.log(D2.c), axis=1, prepend=np.nan)
        D2._simbars = None
        fc2 = forecast.sim_provider(D2)(d)
        asof_ok += fc2[0] == data.dates[d - 1] and fc2[1]["rows"][q["symbol"]]["close"] == round(float(data.c[data.k[q["symbol"]], d - 1]), 2)
        q2 = make(D2, fc2, d, q["symbol"], q["type"], q["side"], q["m"], A)
        ok = q2 is not None and all(q2[k] == q[k] for k in ("asof", "reference", "question"))
        ok &= prompt(dict(q, outcome=1 - q["outcome"], realized_pct=None)) == prompt(q)
        same += ok
        flipped += bool(q2) and q2["outcome"] != q["outcome"]
        if not ok:
            bad.append(q["id"])
    res = {"questions": n, "prompt_unchanged": same, "outcome_changed_by_perturbation": flipped,
           "forecast_row_is_close_of_d_minus_1": asof_ok, "failed": bad}
    (RES / "drill_leak.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"build": build, "baselines": baselines, "leak": leak}.get(cmd, lambda: print(__doc__))()
