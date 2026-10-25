"""R6 parity between rl4's simulator and live (rl4_acceptance_criteria.md), before the first paid step.

    python parity4.py --states runs/live/2026-10-21/rl3-ml-1130.json ...   # state text: live renderer vs the simulator
    python parity4.py --forecast                                         # forecast tool: sim provider vs the live cache

States: each live look file's stored state (positions, open orders, cash, marks, events) is rendered twice with
trader.account_text: (a) as live renders it (a PaperWorld stand-in returning the stored state) and (b) by rebuilding the
book in an intrasim.IntraWorld (positions, average costs, cash, open orders, the same marks) and letting the simulator
produce the state. The two texts are compared line by line with every number masked, so only wording, structure and
field order count; then the numbers are compared with the marks equal. Forecast: forecast.run's output from the
simulator's provider (sim_provider at the close of 15 Oct 2026, the last close both have) against the live cache
runs/ml/forecast/fc_20261015.json served through the same formatter, for the same symbols.
Writes runs/eval4/parity/parity.md.
"""

import argparse
import json
import re
from pathlib import Path

import intrasim
import sim
import trader

ROOT = Path(__file__).resolve().parent
LIVE_ROOT = Path(r"C:\Claude\Trader")
OUT = ROOT / "runs" / "eval4" / "parity"
NUM = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")


class Stub:
    """What live.look hands trader.account_text: a world whose state() is PaperWorld.state's dict."""
    MAX_GROSS = 2.0

    def __init__(self, st):
        self.st = st

    def state(self, d, feats, live=None):
        return dict(self.st)


def rebuild(look, data, intra, d):
    """An IntraWorld holding the stored book, marked at the stored marks."""
    st = look["state"]
    w = trader.SimWorld(data, d, st["cash"])
    for p in st["positions"]:
        w.m.positions[p["symbol"]] = [p["quantity"], p["avg_cost"]]
        w.opened[p["symbol"]] = d - p["days"]
    for i, o in enumerate(st["orders"]):
        typ = o["type"]
        w.m.orders.append(sim.Order(o["ref"], o["symbol"], o["action"], o["quantity"], typ, o.get("limit"),
                                    o.get("stop"), o.get("tif", "DAY"), o.get("trail_percent"), d))
    iw = intrasim.IntraWorld(w, intra)
    iw.events = list(look["events"])
    marks = {p["symbol"]: p["close"] for p in st["positions"]}
    iw.m.last = lambda s, _m=marks: _m.get(s)          # the stored marks (live: delayed last prices)
    return iw


def mask(text):
    return [NUM.sub("#", line) for line in text.splitlines()]


def states(paths):
    data = trader.Data()
    intra = intrasim.Intra(data)
    rows = []
    for p in paths:
        look = json.loads(Path(p).read_text(encoding="utf-8"))
        day = look["time"][:10].replace("-", "")
        d = data.ix.get(day, len(data.dates) - 1)
        feats, _ = data.features(d)
        header = look["state_text"].split("\n\n", 1)[1].split("\n", 1)[0] if "\n\n" in look["state_text"] else None
        live_st = {**look["state"], "events": look["events"]}
        a, _ = trader.account_text(Stub(live_st), data, d, feats, header=header, live=None)
        iw = rebuild(look, data, intra, d)
        st_b = iw.state(d, feats)
        st_b["events"] = look["events"]
        b, _ = trader.account_text(Stub(st_b), data, d, feats, header=header, live=None)
        ma, mb = mask(a), mask(b)
        diff = [(i, x, y) for i, (x, y) in enumerate(zip(ma, mb)) if x != y] + (
            [("length", len(ma), len(mb))] if len(ma) != len(mb) else [])
        num = [(i, x, y) for i, (x, y) in enumerate(zip(a.splitlines(), b.splitlines())) if x != y]
        rows.append({"file": str(p), "lines": len(ma), "structural_diffs": diff, "numeric_diffs": num[:6], "live": a, "sim": b})
    return rows


def forecast_parity(symbols):
    import forecast as fc
    data = trader.Data()
    prov = fc.sim_provider(data)
    d = data.ix["20261016"]                              # decision before 16 Oct: as of the close of 15 Oct

    class SimGw:
        def forecast_provider(self):
            return prov(d, "09:00")

    class CacheGw:                                     # the live cache through the simulator's code path
        def forecast_provider(self):
            doc = json.loads((LIVE_ROOT / "runs" / "ml" / "forecast" / "fc_20261015.json").read_text(encoding="utf-8"))
            return doc["as_of"], {"market": {**doc["market"], "macro": prov(d, "09:00")[1]["market"]["macro"]},
                                  "rows": doc["forecasts"]}

    args = {"symbols": symbols, "notional_usd": 5000}
    sim_txt, live_txt = fc.run(SimGw(), args), fc.run(CacheGw(), args)
    return sim_txt, live_txt


def fills(n=20, seed=4):
    """R6 fill fidelity (criteria v1.1): on n random symbol-days, the same pre-open DAY orders (MKT, MOO, MOC, a LMT and a
    STP 1% through the previous close, each side) against the daily bar (sim.Market) and against its 13 scaled 30-minute
    bars (intrasim.IntraMarket): does each fill in both or neither, and at what price; plus the price-scale rule (the scaled
    16:00 close equals the daily close to the cent)."""
    import random
    data = trader.Data()
    intra = intrasim.Intra(data)
    rng = random.Random(seed)
    days = [d for d in intra.day if 260 < d < len(data.dates) - 1]
    rows, scale_bad = [], 0
    for _ in range(n):
        d = rng.choice(days)
        j = rng.choice([j for j in range(len(data.sym)) if intra.bar(j, d, 0) and intra.bar(j, d, intrasim.NS - 1)])
        s = data.sym[j]
        scale_bad += abs(intra.bar(j, d, intrasim.NS - 1)["close"] - data.c[j, d]) >= 0.005
        prev = data.c[j, d - 1]
        orders = [("MKT", "BUY", None, None), ("MOO", "SELL", None, None), ("MOC", "BUY", None, None),
                  ("LMT", "BUY", round(prev * 0.99, 2), None), ("LMT", "SELL", round(prev * 1.01, 2), None),
                  ("STP", "SELL", None, round(prev * 0.99, 2)), ("STP", "BUY", None, round(prev * 1.01, 2))]
        for typ, side, lim, aux in orders:
            out = {}
            for kind in ("daily", "bars30"):
                m = sim.Market(data.simbars(), data.dates, d, 1e7, max_gross=1e9)
                if kind == "bars30":
                    m = intrasim.IntraMarket.from_market(m, intra)
                o = m.place(s, side, 10, typ, lim, aux, "DAY")
                m.end_session() if kind == "bars30" else m.advance(1)
                out[kind] = (o.status, round(o.fill_price, 2) if o.fill_price else None)
            rows.append({"date": data.dates[d], "symbol": s, "order": f"{side} {typ} {lim or aux or ''}".strip(),
                         "daily": out["daily"], "bars30": out["bars30"],
                         "same_fill": (out["daily"][0] == "Filled") == (out["bars30"][0] == "Filled")})
    return rows, scale_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="*", default=[])
    ap.add_argument("--forecast", action="store_true")
    ap.add_argument("--fills", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# R6 parity: rl4 simulator vs live", ""]
    if a.fills:
        rows, scale_bad = fills()
        same = sum(r["same_fill"] for r in rows)
        md += [f"## Fill fidelity ({len(rows)} orders on 20 symbol-days): {same} fill or not in both; scaled 16:00 close "
               f"off the daily close by a cent or more on {scale_bad} of 20 days", "",
               "| date | symbol | order | daily bar | 30-minute bars | same |", "|---|---|---|---|---|---|"]
        md += [f"| {r['date']} | {r['symbol']} | {r['order']} | {r['daily'][0]} {r['daily'][1] or ''} | "
               f"{r['bars30'][0]} {r['bars30'][1] or ''} | {'yes' if r['same_fill'] else 'NO'} |" for r in rows]
        md += [""]
    if a.states:
        rows = states(a.states)
        bad = [r for r in rows if r["structural_diffs"]]
        md += [f"## State text ({len(rows)} states; {len(bad)} with structural differences)", ""]
        for r in rows:
            md += [f"- `{r['file']}`: {r['lines']} lines, structural differences {len(r['structural_diffs'])}, "
                   f"numeric differences {len(r['numeric_diffs'])}"]
            for x in r["structural_diffs"][:5]:
                md += [f"  - line {x[0]}: live `{x[1]}` / sim `{x[2]}`"]
            for x in r["numeric_diffs"][:3]:
                md += [f"  - numbers, line {x[0]}: live `{x[1]}` / sim `{x[2]}`"]
        (OUT / "states.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    if a.forecast:
        syms = ["NVDA", "AAPL", "MRNA", "AMD", "SMCI", "TSLA", "MU", "LRCX", "XOM", "JPM", "UNH", "SPY", "QQQ", "TLT", "HYG",
                "LQD", "GLD", "XLE", "INTC", "PLTR"]
        s, l = forecast_parity(syms)
        sm, lm = mask(s), mask(l)
        diff = [(i, x, y) for i, (x, y) in enumerate(zip(sm, lm)) if x != y]
        md += ["", f"## Forecast tool (20 symbols, as of the close of 15 Oct 2026): {len(diff)} structural differences", "",
               "Simulator:", "```", s, "```", "Live cache through the same formatter:", "```", l, "```"]
        for x in diff:
            md += [f"- line {x[0]}: sim `{x[1]}` / live `{x[2]}`"]
    (OUT / "parity.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:80]))


if __name__ == "__main__":
    main()
