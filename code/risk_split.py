"""Split each sleeve's week into market, chip and selection parts, for the decks' Risk assessment slide.

    python risk_split.py runs/live/daily.jsonl 760.05 2026-10-23

Arguments: the week's daily records, the SPY baseline (the previous Friday's close) and its date.
For each sleeve and day, the book's net exposure, holdings beta (sum of weight x each name's beta to SPY) and chip
tilt (loading on SMH beyond its own SPY beta) are averaged over the previous close and this close (day 1: this close),
with every loading estimated on the WINDOW sessions up to the previous close from data/bars, so it is point in time.
Printed per sleeve: return since start, average net exposure, beta and chip tilt, and the return minus each model's
market part (net x SPY, beta x SPY, SPY + chips), in points. Names without bars count at beta 1, chip tilt 0.
"""
import collections
import json
import os
import sys

import numpy as np

WINDOW = 120
_bars = {}


def closes(sym):
    if sym not in _bars:
        p = f"data/bars/{sym}.json"
        _bars[sym] = {b["date"]: b["close"] for b in json.load(open(p))["bars"]} if os.path.exists(p) else None
    return _bars[sym]


DAYS = sorted(closes("SPY"))


def rets(sym, end):
    c = closes(sym)
    if c is None:
        return None
    i = DAYS.index(end)
    x = [c.get(d) for d in DAYS[i - WINDOW:i + 1]]
    return np.array([x[k] / x[k - 1] - 1 if x[k] and x[k - 1] else np.nan for k in range(1, len(x))])


_loadings = {}


def loadings(sym, end):
    """(beta to SPY, SPY loading and chip loading in the two-factor fit, SMH's own beta) through `end`."""
    if (sym, end) not in _loadings:
        out = None
        r, sp, sm = rets(sym, end), rets("SPY", end), rets("SMH", end)
        if r is not None:
            ok = ~np.isnan(sp) & ~np.isnan(sm)
            b_smh = np.cov(sp[ok], sm[ok])[0, 1] / np.var(sp[ok], ddof=1)
            chips = sm - b_smh * sp
            ok &= ~np.isnan(r)
            if ok.sum() >= 40:
                beta = np.cov(sp[ok], r[ok])[0, 1] / np.var(sp[ok], ddof=1)
                X = np.column_stack([np.ones(ok.sum()), sp[ok], chips[ok]])
                co = np.linalg.lstsq(X, r[ok], rcond=None)[0]
                out = (beta, co[1], co[2], b_smh)
        _loadings[(sym, end)] = out
    return _loadings[(sym, end)]


def book(row, day, prev):
    nav = row["nav"]
    known = beta = mkt = chip = 0.0
    for sym, qty in (row["positions"] or {}).items():
        c = closes(sym)
        px = None
        if c:
            prior = [k for k in c if k <= day]
            px = c[max(prior)] if prior else None
        f = loadings(sym, prev) if px else None
        if f is None:
            continue
        w = qty * px / nav
        known += w
        beta += w * f[0]
        mkt += w * f[1]
        chip += w * f[2]
    net = (nav - row["cash"]) / nav
    rest = net - known                      # names without bars: beta 1, no chip tilt
    return {"net": net, "beta": beta + rest, "mkt": mkt + rest, "chip": chip}


def main(path, baseline, base_date):
    rows, spys = {}, collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        rows[(r["date"], r["sleeve"])] = r
    for (d, _), r in rows.items():
        spys[d].append(r["spy"])
    dates = sorted(spys)
    spy = {d: collections.Counter(spys[d]).most_common(1)[0][0] for d in dates}   # the value most rows give
    sleeves = sorted({s for _, s in rows})
    print("sleeve return net beta chip vs_net vs_beta vs_spy_chips")
    for s in sleeves:
        prev_nav, prev_spy, prev_day, prev_book = 30000.0, baseline, base_date.replace("-", ""), None
        acc = collections.defaultdict(float)
        for d in dates:
            row = rows[(d, s)]
            day = d.replace("-", "")
            bk = book(row, day, prev_day)
            s_ret = spy[d] / prev_spy - 1                                        # at the 4:15 pm marks, as the NAVs
            spy_bar = closes("SPY")[day] / closes("SPY")[prev_day] - 1
            chips = closes("SMH")[day] / closes("SMH")[prev_day] - 1 - loadings("SMH", prev_day)[3] * spy_bar
            avg = bk if prev_book is None else {k: (bk[k] + prev_book[k]) / 2 for k in bk}
            for k in ("net", "beta", "chip"):
                acc[k] += avg[k]
            acc["net_part"] += avg["net"] * s_ret
            acc["beta_part"] += avg["beta"] * s_ret
            acc["two_part"] += avg["mkt"] * s_ret + avg["chip"] * chips
            prev_nav, prev_spy, prev_day, prev_book = row["nav"], spy[d], day, bk
        ret = row["nav"] / row["start_cash"] - 1
        n = len(dates)
        print(f"{s} {ret * 100:+.2f}% {acc['net'] / n:.2f} {acc['beta'] / n:.2f} {acc['chip'] / n:.2f} "
              f"{(ret - acc['net_part']) * 100:+.2f} {(ret - acc['beta_part']) * 100:+.2f} {(ret - acc['two_part']) * 100:+.2f}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3])
