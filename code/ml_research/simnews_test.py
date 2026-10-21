"""Tests for ml_research/simnews.py: no item at or after the decision time is ever returned, and defaults are unchanged.

    .venv\\Scripts\\python.exe ml_research\\simnews_test.py          # writes runs/ml/research/simnews_test.json

1. Sweep: every training-pool session plus 60 random sessions, every symbol, news(limit=1e6), tape(), and article()
   of returned ids: every item's time is strictly before 09:00 ET of the session (a macro release also before that
   day, as the live calendar has it; macro_feats.on_tape).
2. Perturbation: per session, alter or delete every real item at or after the cutoff and inject synthetic ones at
   the cutoff, +1 s, +1 day and far in the future (per-symbol headlines, EDGAR filings, macro and BRFG tape items).
   news, tape, article and the tools.run text must be byte-identical; injected future ids must give the same error
   as an unknown id.
3. Power: an item injected at cutoff - 1 s must change the output.
4. Defaults: simgw.SimGateway offers the same tools as before; train2.build with news off equals HEAD's prompt.
"""

import copy
import json
import random
import subprocess
import sys
import time
import types
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import simgw  # noqa: E402
import tools  # noqa: E402
import trader  # noqa: E402
from ml_research import macro_feats  # noqa: E402
from ml_research import simnews as sn  # noqa: E402

OUT = ROOT / "runs" / "ml" / "research" / "simnews_test.json"
BIG = 10 ** 6


def pool_dates(data):
    return [i for i, x in enumerate(data.dates) if "20250103" <= x <= "20251212"]


def random_dates(data, n=60, seed=1):
    rng = random.Random(seed)
    cand = [i for i, x in enumerate(data.dates) if "20231101" <= x <= "20261015"]
    return sorted(rng.sample(cand, n))


def all_symbols(A, data):
    return sorted(set(A.sym) | set(data.sym))


def snapshot(gw, syms, article_ids):
    """Everything the tools can show at this look, raw rows and rendered text."""
    out = {"news": {s: gw.news(s, limit=BIG) for s in syms},
           "news_text": {s: tools.run(gw, "news", {"symbol": s}) for s in syms},
           "news_brfg": {s: gw.news(s, providers="BRFG", limit=BIG) for s in syms[:40]},
           "news_before": {s: gw.news(s, limit=BIG, before="2099-01-01 00:00:00") for s in syms[:40]},
           "tape": gw.tape(limit=BIG), "tape_text": tools.run(gw, "tape", {}),
           "articles": {f"{p}|{a}": tools.run(gw, "article", {"provider": p, "article_id": a}) for p, a in article_ids}}
    return out


def visible_ids(gw, syms, per=3):
    ids = set()
    for s in syms:
        for r in gw.news(s, limit=per):
            ids.add((r["provider"], r["article_id"]))
    for r in gw.tape(limit=BIG):
        ids.add((r["provider"], r["article_id"]))
    return sorted(ids)


# ---------------------------------------------------------------- 1. sweep
def sweep(data, pre, A, dates):
    syms = all_symbols(A, data)
    n_rows = n_tape = n_art = 0
    bad = []
    for d in dates:
        gw = sn.NewsSimGateway(data, d, pre, news_archive=A)
        C = gw.cutoff
        assert C == sn.decision_ts(data.dates[d])
        for s in syms:
            try:
                rows = gw.news(s, limit=BIG)
            except ValueError:
                continue
            n_rows += len(rows)
            bad += [(data.dates[d], s, r["time"]) for r in rows if sn.ts(r["time"]) >= C]
        tape = gw.tape(limit=BIG)
        n_tape += len(tape)
        bad += [(data.dates[d], "tape", r["time"]) for r in tape if not (C - sn.TAPE_DAYS * 86400 <= sn.ts(r["time"]) < C)]
        # every EDGAR / MACRO id in the archive: visible iff accepted before the cutoff
        if d in dates[::7]:
            for acc, f in A.edgar.items():
                try:
                    gw.article("EDGAR", acc)
                    ok = True
                except ValueError as e:
                    ok = False
                    assert str(e) == sn.NOT_FOUND
                n_art += 1
                if ok != (f["t"] < C):
                    bad.append((data.dates[d], "article EDGAR", acc))
            for aid, (t, r) in A.macro_by_id.items():
                try:
                    gw.article("MACRO", aid)
                    ok = True
                except ValueError:
                    ok = False
                n_art += 1
                if ok != (t < C and macro_feats.on_tape(t, aid, datetime.fromtimestamp(C, sn.ET))):
                    bad.append((data.dates[d], "article MACRO", aid))
            for (p, aid), t in A.ibkr.items():                  # IBKR ids, now with backfilled text for BRFG
                try:
                    gw.article(p, aid)
                    ok = True
                except ValueError:
                    ok = False
                n_art += 1
                if ok != (t < C):
                    bad.append((data.dates[d], f"article {p}", aid))
    return {"dates": len(dates), "symbols": len(syms), "news_rows_checked": n_rows, "tape_rows_checked": n_tape,
            "article_ids_checked": n_art, "violations": len(bad), "examples": bad[:10]}


# ---------------------------------------------------------------- 2-3. perturbation
FUT = "9999999999-99-"


def perturbed(A, C, rng, inject_before=False):
    """A copy of the archive as if the future had been different: every item at or after C altered or deleted,
    plus synthetic items at C, C+1, C+1 day and C+3 years. inject_before also adds one item at C-1 (power check)."""
    offs = [0, 1, 86400, 10 ** 8]
    sym_items, k = {}, 0
    for s, v in A.sym.items():
        rows = []
        for t, r in v:
            if t < C:
                rows.append((t, r))
            elif rng.random() < 0.5:                       # delete half, alter the rest (headline and time)
                rows.append((t + rng.randint(0, 3600), {**r, "headline": "ALTERED " + r["headline"][::-1]}))
        for o in offs:
            k += 1
            rows.append((C + o, {"time": sn.iso_utc(C + o), "provider": "BRFUPDN", "article_id": f"BRFUPDN$fut{k}",
                                 "headline": f"!FUTURE upgrade of {s}", "meta": ""}))
            acc = f"{FUT}{k:06d}"
            rows.append((C + o, {"time": sn.iso_utc(C + o), "provider": "EDGAR", "article_id": acc,
                                 "headline": f"{s} files 8-K: FUTURE (2.02)", "meta": ""}))
        sym_items[s] = rows
    edgar = {a: f for a, f in A.edgar.items() if f["t"] < C}
    for a, f in A.edgar.items():
        if f["t"] >= C and rng.random() < 0.5:
            edgar[a] = {**f, "items": ["1.03"], "t": f["t"] + 60}
    for i, o in enumerate(offs):
        edgar[f"{FUT}x{i}"] = {"t": C + o, "symbols": ["AAPL"], "form": "8-K", "items": ["2.02"], "name": "FUTURE", "has_text": True}
    macro = [(t, r) for t, r in A.macro if t < C]
    for t, r in A.macro:
        if t >= C and rng.random() < 0.5:
            r2 = copy.deepcopy(r)
            r2["headline"] = "ALTERED " + r["headline"]
            if "_detail" in r2 and r2["_detail"].get("first_release"):
                r2["_detail"]["first_release"] = {kk: (vv * 3 if isinstance(vv, (int, float)) and not isinstance(vv, bool) else vv)
                                                  for kk, vv in r2["_detail"]["first_release"].items()}
            macro.append((t, r2))
    brfg = [(t, r) for t, r in A.brfg if t < C]
    for i, o in enumerate(offs):
        macro.append((C + o, {"time": sn.iso_utc(C + o), "provider": "MACRO", "article_id": f"MACRO$fut{i}",
                              "headline": "CPI rose 9.9% (FUTURE)", "meta": ""}))
        brfg.append((C + o, {"time": sn.iso_utc(C + o), "provider": "BRFG", "article_id": f"BRFG$fut{i}",
                             "headline": "FUTURE market column", "meta": ""}))
    if inject_before:
        for s in ("AAPL", "MSFT"):
            sym_items.setdefault(s, []).append((C - 1, {"time": sn.iso_utc(C - 1), "provider": "BRFG", "article_id": f"BRFG$pre{s}",
                                                        "headline": f"PAST item for {s}", "meta": ""}))
        brfg.append((C - 1, {"time": sn.iso_utc(C - 1), "provider": "BRFG", "article_id": "BRFG$pretape",
                             "headline": "PAST market column", "meta": ""}))
    return sn.Archive(sym_items, edgar, macro, brfg, A.names, A.edgar_text_dir)


def perturbation(data, pre, A, dates, seed=7):
    rng = random.Random(seed)
    syms = all_symbols(A, data)
    res = {"dates": len(dates), "identical": 0, "differences": [], "future_ids_refused": 0, "power_detected": 0}
    for d in dates:
        base_gw = sn.NewsSimGateway(data, d, pre, news_archive=A)
        C = base_gw.cutoff
        ids = visible_ids(base_gw, syms)
        base = snapshot(base_gw, syms, ids)
        P = perturbed(A, C, rng)
        pgw = sn.NewsSimGateway(data, d, pre, news_archive=P)
        pert = snapshot(pgw, syms, ids)
        if pert == base:
            res["identical"] += 1
        else:
            diff = [k for k in base if base[k] != pert[k]]
            res["differences"].append((data.dates[d], diff))
        # injected future ids: same error text as an id that never existed
        unknown = tools.run(pgw, "article", {"provider": "EDGAR", "article_id": "0000000000-00-000000"})
        for p, a in [("EDGAR", f"{FUT}x0"), ("EDGAR", f"{FUT}x3"), ("MACRO", "MACRO$fut0"), ("BRFUPDN", "BRFUPDN$fut1"),
                     ("BRFG", "BRFG$fut0")]:
            got = tools.run(pgw, "article", {"provider": p, "article_id": a})
            assert NOT_FOUND_TEXT in got and got.replace(p, "EDGAR") == unknown.replace(p, "EDGAR"), (p, a, got)
            res["future_ids_refused"] += 1
        # power check: an item one second before the cutoff must show up
        Q = perturbed(A, C, random.Random(seed), inject_before=True)
        qgw = sn.NewsSimGateway(data, d, pre, news_archive=Q)
        q = {"aapl": qgw.news("AAPL", limit=BIG), "tape": qgw.tape(limit=BIG)}
        brfg_first = [r for r in q["tape"] if r["provider"] != "MACRO"]          # macro rows come first on the tape
        if (q["aapl"] != base["news"]["AAPL"] and q["aapl"][0]["article_id"] == "BRFG$preAAPL"
                and q["tape"] != base["tape"] and brfg_first and brfg_first[0]["article_id"] == "BRFG$pretape"):
            res["power_detected"] += 1
    return res


NOT_FOUND_TEXT = sn.NOT_FOUND


# ---------------------------------------------------------------- 4. defaults
def defaults(data, pre):
    out = {}
    d = data.ix["20250506"]
    plain = simgw.SimGateway(data, d, pre)
    out["simgw_tools"] = [s["name"] for s in tools.available(plain)]
    out["simgw_has_news"] = any(hasattr(plain, a) for a in ("news", "tape", "article"))
    out["newsgw_tools"] = [s["name"] for s in tools.available(sn.NewsSimGateway(data, d, pre))]
    head_src = subprocess.run(["git", "show", "HEAD:train2.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout
    head = types.ModuleType("train2_head")
    head.__file__ = str(ROOT / "train2.py")
    exec(compile(head_src, "train2_head", "exec"), head.__dict__)
    import train2
    same, n = 0, 0
    for ds in ("20250103", "20250506", "20251015"):
        dd = data.ix[ds]
        for ml in (False, True):
            w = trader.SimWorld(data, dd - 1, train2.SIZE)
            w.advance()
            memo = {"when": "Mon 05 May 09:00 ET", "notes": "hold AAPL"}
            n += 1
            same += head.build(w.clone(), data, dd, memo, ml) == train2.build(w.clone(), data, dd, memo, ml)
    out["build_same_as_head"] = f"{same}/{n}"
    w = trader.SimWorld(data, d - 1, train2.SIZE)
    w.advance()
    sysn, usern = train2.build(w, data, d, {}, False, news=True)
    out["news_prompt_mentions_tape"] = "tape" in sysn and "call tape" in usern
    captured, saved = {}, (sys.argv, train2.main_async, train2.asyncio.run)
    try:
        sys.argv = ["train2.py", "--tag", "x"]
        train2.main_async = lambda cfg: captured.setdefault("cfg", cfg)
        train2.asyncio.run = lambda x: None
        train2.main()
    finally:
        sys.argv, train2.main_async, train2.asyncio.run = saved
    ap_args = captured["cfg"]
    out["default_news_flag"] = ap_args.news
    out["default_init"] = ap_args.init
    return out


def main():
    t0 = time.time()
    data = trader.Data(last="20261015")
    pre = simgw.Precomputed(data)
    A = sn.archive()
    dates = sorted(set(pool_dates(data)) | set(random_dates(data)))
    res = {"archive": {"symbols": len(A.sym), "edgar_filings": len(A.edgar), "macro_tape_items": len(A.macro),
                       "brfg_items": len(A.brfg), "ibkr_headlines": len(A.ibkr)}}
    res["sweep"] = sweep(data, pre, A, dates)
    print("sweep", json.dumps(res["sweep"])[:400], flush=True)
    pdates = random.Random(3).sample(dates, 30) + [data.ix[x] for x in ("20250502", "20250918", "20250430", "20251029", "20250501")]
    res["perturbation"] = perturbation(data, pre, A, sorted(set(pdates)))
    print("perturbation", json.dumps(res["perturbation"])[:400], flush=True)
    res["defaults"] = defaults(data, pre)
    print("defaults", res["defaults"], flush=True)
    p, s, df = res["perturbation"], res["sweep"], res["defaults"]
    res["pass"] = (s["violations"] == 0 and p["identical"] == p["dates"] and p["power_detected"] == p["dates"]
                   and not df["simgw_has_news"] and df["build_same_as_head"].split("/")[0] == df["build_same_as_head"].split("/")[1]
                   and df["default_news_flag"] is False and df["default_init"] == "")
    res["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print("PASS" if res["pass"] else "FAIL", res["seconds"], "s")
    return res


if __name__ == "__main__":
    main()
