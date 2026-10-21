"""Do the models research before trading? Sample many looks on the same real sleeve state, place
nothing, and count tool use, per model and prompt variant.

    python tooltest.py --models base rl1 --variants v0 v1 v2 v0m --n 10 --tag t1

A variant changes only the instructions (prompt engineering); the tools, state and model are fixed.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import tinker

import agent
import paper
import tools
import trader

ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
PROJECT_ID = "51e502c3-a670-40ad-b14a-c9eb569d4363"
BASE = "openai/gpt-oss-120b"
MODELS = {"base": None, "rl1": "tinker://033b96dd-9d6b-5cd6-8bfc-c2ffee6b144f:train:0/sampler_weights/rl1-step030"}

RESEARCH_RULE = ("\n\nHow to decide: research first, then act. Call tape to read today's market news. For every stock you "
                 "consider buying or selling, call news and quote (and bars if the trend matters). Do not trade a symbol you "
                 "have not checked in this look. Then call submit once.")
USER_STEPS = ("Before you submit: (1) call tape for market news; (2) pick candidates from the table and your positions; "
              "(3) call news and quote for each candidate; (4) call submit.\n\n")
VARIANTS = {
    "v0": {"rule": "", "steps": "", "effort": "low"},          # the live prompt as it is
    "v0m": {"rule": "", "steps": "", "effort": "medium"},       # same prompt, more reasoning
    "v1": {"rule": RESEARCH_RULE, "steps": "", "effort": "low"},     # research rule in the system text
    "v2": {"rule": RESEARCH_RULE, "steps": USER_STEPS, "effort": "low"},   # rule + steps right before the state
    "live": {"rule": "", "steps": None, "effort": "low"},      # what live.py sends now (SYSTEM_LIVE has the rule; LIVE_STEPS)
}


def state_for(gw, sleeve, data, d, now):
    """The sleeve's real state text, as the live loop would build it (read-only)."""
    session = now.time() >= datetime.strptime("09:30", "%H:%M").time() and now.time() < datetime.strptime("16:00", "%H:%M").time()
    day = f"Date: {now.date().isoformat()} ({now.strftime('%A')}), {now:%H:%M} ET."
    header = (f"{day} The market is open until 16:00 ET; prices are delayed about 15 minutes. Orders you submit now work immediately."
              if session else f"{day} Before the open. Orders you submit now work from the 09:30 open.")
    live = None
    if session:
        rows, _ = data.table_rows(d, extra=sorted(paper.book(sleeve)["positions"]))
        live = {q["symbol"]: paper.mark(q) for q in gw.quotes(rows) if not q.get("error")}
    w = trader.PaperWorld(gw, sleeve, data)
    w.dry_run = True
    text, st, feats = trader.state_text(w, data, d, header=header, live=live)
    rows, _ = data.table_rows(d, extra=w.symbols_in_play())
    import live as livemod
    goal = trader.goal_line(paper.sleeves()[sleeve]["start_cash"], st["value"], livemod.sessions_left(now.date()), now.date())
    return text, rows, goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["base", "rl1"])
    ap.add_argument("--variants", nargs="+", default=["v0", "v1", "v2"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--client", default="trader-research")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    from tws import Gateway
    out = ROOT / "runs" / "tooltest" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(ET)
    data = trader.Data(last=(now.date() - timedelta(days=1)).strftime("%Y%m%d"))
    data.next_date = now.date().isoformat()
    d = len(data.dates)
    service = tinker.ServiceClient(project_id=PROJECT_ID, user_metadata={"purpose": f"tooltest:{a.tag}"})
    size = paper.sleeves()["base"]["start_cash"]
    summary = []
    with Gateway(a.client, timeout=30) as gw:
        specs = tools.available(gw)
        for m in a.models:
            sc = service.create_sampling_client(base_model=BASE) if MODELS[m] is None else service.create_sampling_client(model_path=MODELS[m])
            text, rows, goal = state_for(gw, m if m in paper.sleeves() else "base", data, d, now)
            for v in a.variants:
                cfg = VARIANTS[v]
                if cfg["steps"] is None:                     # exactly what live.py sends
                    system, user = trader.live_system(size, now.date()), trader.live_user(text, goal)
                else:
                    system = trader.SYSTEM_LIVE.replace("${size:,}", f"${size:,.0f}") + cfg["rule"]
                    user = cfg["steps"] + text

                def one(i):
                    t0 = time.time()
                    try:
                        call, trace, pf, sp = agent.gptoss_look(sc, gw, system, user, now.date().isoformat(), specs, effort=cfg["effort"])
                    except Exception as e:
                        return {"i": i, "error": repr(e)[:300]}
                    calls = [t for t in trace if t.get("tool") and t["tool"] != "submit"]
                    used = {}
                    researched = set()
                    for t in calls:
                        used[t["tool"]] = used.get(t["tool"], 0) + 1
                        researched |= set(tools.symbols_in(t["tool"], t.get("args")))
                    traded = {str(o.get("symbol", "")).upper() for o in (call or {}).get("place", []) if isinstance(o, dict)}
                    return {"i": i, "valid": call is not None, "research_calls": len(calls), "tools": used,
                            "researched": sorted(researched), "traded": sorted(traded),
                            "traded_researched": sorted(traded & researched), "secs": round(time.time() - t0, 1),
                            "prefill": pf, "sampled": sp, "trace": trace, "call": call}
                with ThreadPoolExecutor(a.threads) as ex:
                    res = list(ex.map(one, range(a.n)))
                (out / f"{m}_{v}.json").write_text(json.dumps(res, default=str, indent=1), encoding="utf-8")
                ok = [r for r in res if "error" not in r]
                n = len(ok) or 1
                tr = sum(len(r["traded"]) for r in ok)
                row = {"model": m, "variant": v, "looks": len(ok), "valid": sum(r["valid"] for r in ok),
                       "researched_any": sum(r["research_calls"] > 0 for r in ok), "mean_calls": sum(r["research_calls"] for r in ok) / n,
                       "tools": {k: sum(r["tools"].get(k, 0) for r in ok) for k in ("quote", "bars", "news", "tape", "article", "scanner", "stats")},
                       "traded_not_in_table": sum(len(set(r["traded"]) - set(rows)) for r in ok),
                       "traded_symbols": tr, "traded_researched": sum(len(r["traded_researched"]) for r in ok),
                       "no_trade": sum(not r["traded"] for r in ok if r["valid"]),
                       "cost_usd": sum(r["prefill"] * 0.33 + r["sampled"] * 0.84 for r in ok) / 1e6, "errors": len(res) - len(ok)}
                summary.append(row)
                print(json.dumps(row), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
