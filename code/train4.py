"""rl4: GRPO on multi-look episodes in the intraday simulator (intrasim.py), from rl3-step008's training state.

Why (runs/ml/research/week2_sleeve_analysis.md): rl3's simulator scored one pre-open decision whose orders ran untouched for
10 sessions, so it could see none of week 2's errors (a 90%-of-NAV name held without a stop across looks and overnight,
exits mishandled at later looks, positions misread at submit). Here one sample is an episode: the same sleeve takes
intrasim.LOOKS research looks in one session with the live prompt, tools, state text (the week-3 presentation) and retry
rule, and its book and notes carry over. A group is G independent episodes from one burn-in book. Credit (criteria v1.1,
R5): one datum per look, with advantage = the look's reward-to-go (intrasim.Ledger's trading reward from that look on,
minus the penalties of that look and later ones) minus the mean reward-to-go at the same look over the group; a look with
no valid submit holds the book.

Losses since the start: live, the sleeves were 5-17% down by midweek and rl3-ml sized to make it back. The prompt no longer
has a dollar goal (Clayton, 24 Oct: a pace line), but each group still starts its burn-in book between LOSS_LO and LOSS_HI
below its $30,000 start, so the state line "Sleeve now ... since its $30,000 start" covers what live shows.

    python train4.py --tag rl4-smoke --steps 1 --groups 2 --group 2 --no-update --init <rl3-step008 state>
    python train4.py --tag rl4 --steps 8 --init tinker://aefed264-7565-52b6-b3bd-192fbbbc2fe6:train:0/weights/rl3-step008
"""

import argparse
import asyncio
import json
import random
import time
from datetime import date
from pathlib import Path

import tinker
from tinker import types

import agent
import intrasim
import simgw
import tools
import train2
import trader

ROOT = Path(__file__).resolve().parent
PROJECT_ID = train2.PROJECT_ID
SIZE = train2.SIZE
PRICE = train2.PRICE
TRAIN_FIRST, TRAIN_LAST = "20251020", "20260630"    # 30-minute bars start 17 Oct 2025 (analysis criteria, 24 Oct)
VAL_FIRST, VAL_LAST = "20260701", "20261015"        # validation (R2b); the OOS forecasts end with the close of 15 Oct
MALFORMED = 1.0                                     # a look with no valid submit
REJECT_PENALTY, REJECT_CAP = 0.5, 3                 # per rejected order, at most REJECT_CAP a look
SMALL_ENTRY = 0.0                                   # rl5: dropped (M9 is reported, not hard; rl4 step 1: 0.06 R an active look)
UNRESEARCHED = 0.1                                  # per symbol of an exposure-adding order with no quote/news/stats/forecast
                                                    # call on it that returned data in the look (R1's hard M8, criteria v1.1)
EMPTY_COST = 0.12                                   # rl5: an empty look (no valid order, no cancel) pays what an active look
                                                    # paid in penalties on average (rl4 step 1: 0.10-0.14 R), so inaction is
                                                    # no longer a way to avoid them (rl4 stopped on its empty-share guard)
RESEARCH_TOOLS = ("quote", "news", "stats", "forecast")
LOSS_LO, LOSS_HI = -0.02, 0.15                      # the burn-in book's start: SIZE x (1 - loss), loss ~ U(LOSS_LO, LOSS_HI)
NOISE_ABORT = None                                  # retired for rl5 (criteria review 24 Oct): the post-close part is expected, not realized
DAY = 86400
RL3_STATE = "tinker://aefed264-7565-52b6-b3bd-192fbbbc2fe6:train:0/weights/rl3-step008"


def iso(data, d):
    return date.fromisoformat(data.iso(d))


def header(data, d, hhmm):
    """The live look's first line (live.look)."""
    t = iso(data, d)
    day = f"Date: {t.isoformat()} ({t.strftime('%A')}), {hhmm} ET."
    if intrasim.minutes(hhmm) >= 570:
        return (f"{day} The market is open until 16:00 ET; prices are delayed about 15 minutes. Orders you submit now "
                f"work immediately.")
    return f"{day} Before the open. Orders you submit now work from the 09:30 open."


def build(world, data, d, hhmm, memo, ml):
    """(system, user, state) for a look at hhmm, as live.look builds it (the sleeve's start is SIZE, as live's start_cash)."""
    live = world.marks() if intrasim.minutes(hhmm) >= 570 else None
    text, st, feats = trader.state_text(world, data, d, header=header(data, d, hhmm), live=live)
    user = trader.live_user(text, trader.goal_line(SIZE, st["value"], None, iso(data, d)), memo, steps=trader.live_steps(ml))
    return trader.live_system(SIZE, iso(data, d), ml=ml), user, st


async def burn(sc, data, d, k, cash0, cfg):
    """k cheap pre-open decisions (submit only) on daily bars before session d, from a sleeve of cash0 against its SIZE
    start. Returns the daily world, the notes and the tokens used."""
    w = trader.SimWorld(data, d - k, cash0)
    memo, used = {}, {"prefill": 0, "sampled": 0}
    for i in range(k):
        di = d - k + i
        text, st, _ = trader.state_text(w, data, di, header=train2.header(data, di))
        user = trader.live_user(text, trader.goal_line(SIZE, st["value"], None, iso(data, di)), memo,
                                steps=trader.live_steps(False, news=False))
        system = trader.live_system(SIZE, iso(data, di), ml=False, news=False)
        p = train2.prompt_ints(train2.renderer(data, di, cfg.effort), [trader.TOOL_LIVE], system, user)
        res = await sc.sample_async(tinker.ModelInput.from_ints(p), 1, train2.params(cfg.max_tokens, len(p)))
        s = res.sequences[0]
        used["prefill"] += len(p)
        used["sampled"] += len(s.tokens)
        kind, name, val = agent.parse_any(list(s.tokens), s.stop_reason, {"submit"})
        if kind == "TOOL" and isinstance(val, dict) and isinstance(val.get("place", []), list) and isinstance(val.get("cancel", []), list):
            w.act({"place": val.get("place", []), "cancel": val.get("cancel", [])})
            memo = {"when": f"{iso(data, di):%a %d %b} 09:00 ET", "notes": str(val.get("notes") or "")[:2000]}
        w.advance()
    return w, memo, used


async def run_look(sc, gw, names, prompt, cfg, first=None):
    """One look from its prompt to a submit, with live's retry rule (agent.gptoss_look): a turn that is not a tool call is
    dropped and re-asked at the same point, the submit is opened after prose or a second miss, and the look fails on the
    third. Dropped turns are not trained. Returns the token record."""
    seq, mask, lps = list(prompt), [0] * len(prompt), [0.0] * len(prompt)
    calls, bad, trace, force = 0, 0, [], []
    cost = {"prefill": 0, "cached": 0, "sampled": 0}
    fresh = len(prompt)                           # tokens not yet seen by the sampler (for the cost estimate)
    for step in range(tools.MAX_CALLS + 4):
        if step == 0 and first is not None:
            toks, lp, stop = first
        else:
            if agent.CONTEXT - len(seq) - len(force) < agent.MIN_ROOM:
                trace.append({"step": step, "error": "context full"})
                break
            try:
                res = await sc.sample_async(tinker.ModelInput.from_ints(seq + force), 1,
                                            train2.params(cfg.max_tokens, len(seq) + len(force)))
            except Exception as e:                  # noqa: BLE001 - the tokens already paid for stay; the look fails
                trace.append({"step": step, "error": f"sampling error: {e!r}"[:200]})
                break
            s = res.sequences[0]
            toks, lp, stop = list(s.tokens), list(s.logprobs), s.stop_reason
            cost["prefill"] += fresh + len(force)
            cost["cached"] += len(seq) - fresh
        fresh = 0
        cost["sampled"] += len(toks)
        kind, name, val = agent.parse_any(force + toks, stop, names)
        if kind != "TOOL":
            bad += 1
            trace.append({"step": step, "error": f"{kind}: {str(val)[:200]}", "tokens": len(toks)})
            if bad >= 3:
                return {"call": None, "seq": seq, "mask": mask, "lps": lps, "trace": trace, "calls": calls, "bad": bad, "cost": cost}
            if kind == "FINAL" or bad == 2:
                force = agent.SUBMIT_OPEN()
            continue
        seq += force + toks
        mask += [0] * len(force) + [1] * len(toks)
        lps += [0.0] * len(force) + lp
        force = []
        if name == "submit":
            ok = isinstance(val, dict) and isinstance(val.get("place", []), list) and isinstance(val.get("cancel", []), list)
            trace.append({"step": step, "tool": "submit"})
            call = {"place": val.get("place", []), "cancel": val.get("cancel", []), "notes": str(val.get("notes") or "")} if ok else None
            return {"call": call, "seq": seq, "mask": mask, "lps": lps, "trace": trace, "calls": calls, "bad": bad, "cost": cost}
        if calls >= tools.MAX_CALLS:
            out = f"research limit reached ({tools.MAX_CALLS} calls): call submit now"
            force = agent.SUBMIT_OPEN()
        else:
            calls += 1
            out = tools.run(gw, name, val)
        trace.append({"step": step, "tool": name, "args": val, "result": out[:600]})
        rt = agent.tool_result_tokens(name, out)
        seq += rt
        mask += [0] * len(rt)
        lps += [0.0] * len(rt)
        fresh = len(rt)
    return {"call": None, "seq": seq, "mask": mask, "lps": lps, "trace": trace + [{"error": "no submit"}], "calls": calls,
            "bad": bad, "cost": cost}


def researched(trace):
    """Symbols a quote, news, stats or forecast call returned data on in this look (R1's hard M8: a refused call over the
    limit, an error, "no headlines" or a "not covered"/"error" row does not count)."""
    import re
    seen = set()
    for t in trace:
        name, res = t.get("tool"), str(t.get("result", ""))
        if name not in RESEARCH_TOOLS or res.startswith("research limit") or res.startswith("error"):
            continue
        for s in tools.symbols_in(name, t.get("args")):
            if name == "news":
                ok = res.strip() not in ("", "no headlines")
            else:
                ok = s in res and not re.search(rf"(^|\n){re.escape(s)}:? (error|not covered)", res)
            if ok:
                seen.add(s)
    return seen


def substantive(typ, action, qty, limit, stop, trail_pct, q0, last, r1):
    """Whether one order (placed now, or open and being cancelled) counts as acting for EMPTY_COST, and its level's
    distance from the last price in range1 units (None for a market-type order). Counts: a stop-type order (STP, STP LMT,
    TRAIL) covering at least half of a held position within 2 range1 (the stop credit's reach; a TRAIL by its trail %), or
    an order of at least $1,000 that is marketable (MKT/MOO/MOC) or prices within 1 range1 of the last price (LMT, LOC,
    STP LMT, a non-protective STP; about 13-15% of sessions touch a level 1 range1 away, Trader Training's check on
    vol2_oos), in a symbol with market data (last)."""
    if not last:
        return False, None
    r = r1 or 0.03
    q1 = q0 + (qty if action == "BUY" else -qty)
    protective = (typ in ("STP", "STP LMT", "TRAIL") and abs(q0) > 0 and abs(q1) < abs(q0)
                  and min(qty, abs(q0)) >= 0.5 * abs(q0))
    if typ == "TRAIL":
        d = (trail_pct or 0) / 100 / r if trail_pct else None
        return bool(protective and d is not None and d <= 2), d
    lvl = limit if typ in ("LMT", "LOC", "STP LMT") else stop if typ == "STP" else None
    d = abs(lvl - last) / last / r if lvl else None
    if protective:
        return d is not None and d <= 2, d
    return qty * (lvl or last) >= 1000 and (d is None or d <= 1), d


def order_stats(call, st, world, trace=(), r1=None):
    """Process facts about one decision against the book it saw: orders taking a name to 50%+ of the sleeve (M1), orders
    under $1,000 (M9) and entries among them, entries without a protective order for the same name in the same submit,
    protective (stop-type) orders, symbols of exposure-adding orders with no data-returning quote/news/stats/forecast call
    in the look (hard M8) and of any order with no such call (M8x), orders missing p_fill or p_win, and whether the look
    was empty for EMPTY_COST: no substantive order placed and no cancel of an open order that would itself be substantive
    now. Token actions (a few shares, levels far from the market, unfillable symbols, cancel-and-replace of far orders)
    are inaction in disguise (the analysis session and Trader Training, 24 Oct) and count as token_orders. r1: {symbol:
    range1 as a fraction}."""
    seen = researched(trace)
    r1 = r1 or {}
    nav = st["value"] or 1.0
    pos = {p["symbol"]: p["quantity"] for p in st["positions"]}
    open_by_ref = {o["ref"]: o for o in st["orders"]}
    real_cancels = 0
    for ref in ((call or {}).get("cancel") or []):
        o = open_by_ref.get(str(ref))
        if o:
            ok, _ = substantive(o["type"], o["action"], o["quantity"], o.get("limit"), o.get("stop"), o.get("trail_percent"),
                                pos.get(o["symbol"], 0), world.m.last(o["symbol"]), r1.get(o["symbol"]))
            real_cancels += ok
    big = small = small_entries = entries = unprotected_entries = prot = stop_type = no_probs = valid = subst = far = 0
    notional = entry_notional = 0.0
    lim_d = []
    place = [o for o in (call or {}).get("place", []) if isinstance(o, dict)]
    ordered, adding = set(), set()
    stops = {(str(o.get("symbol", "")).upper().strip(), str(o.get("action", "")).upper()) for o in place
             if str(o.get("type", "")).upper().replace("_", " ") in ("STP", "STP LMT", "TRAIL")}
    for o in place:
        c, err = trader.clean_order(o)
        if err:
            continue
        s, qty = c["symbol"], c["quantity"]
        ordered.add(s)
        valid += 1
        no_probs += c["p_fill"] is None or c["p_win"] is None
        px = c["limit"] or c["stop"] or world.m.last(s) or 0.0
        q0 = pos.get(s, 0)
        q1 = q0 + (qty if c["action"] == "BUY" else -qty)
        small += qty * px < 1000
        notional += qty * px
        ok, d = substantive(c["type"], c["action"], qty, c["limit"], c["stop"], c["trail_percent"], q0, world.m.last(s), r1.get(s))
        subst += ok
        far += not ok
        if d is not None and c["type"] in ("LMT", "LOC", "STP LMT"):
            lim_d.append(d)
        if c["type"] in ("STP", "STP LMT", "TRAIL"):
            stop_type += 1
            prot += abs(q1) < abs(q0)
            continue
        if abs(q1) > abs(q0):
            adding.add(s)
            entries += 1
            small_entries += qty * px < 1000
            entry_notional += qty * px
            unprotected_entries += (s, "SELL" if q1 > 0 else "BUY") not in stops
            big += abs(q1) * px >= 0.5 * nav
        pos[s] = q1
    return {"orders": len(place), "orders_ex_protective": len(place) - prot, "held": len(st["positions"]),
            "m1_big": big, "under_1k": small, "small_entries": small_entries, "entries": entries,
            "entries_no_stop": unprotected_entries, "protective": prot, "stop_type": stop_type, "no_probs": no_probs,
            "unresearched": len(adding - seen), "unresearched_any": len(ordered - seen),
            "empty": call is not None and not subst and not real_cancels, "token_orders": valid - subst,
            "far_orders": far, "notional": notional, "entry_notional_nav": entry_notional / nav, "limit_dist_r1": lim_d,
            "notes": bool(str((call or {}).get("notes") or "").strip())}


async def group(sc, data, pre, intra, prov, d, k, ml, loss, cfg):
    """One GRPO group: a burn-in book, then cfg.group episodes of cfg.looks looks from it."""
    w0, memo0, burn_used = await burn(sc, data, d, k, round(SIZE * (1 - loss), 2), cfg)
    members = await episodes(sc, data, pre, intra, prov, d, ml, intrasim.IntraWorld(w0, intra), memo0, cfg.group, cfg)
    return {"date": data.dates[d], "burn": k, "ml": ml, "loss": loss, "members": members, "burn_cost": burn_used}


async def episodes(sc, data, pre, intra, prov, d, ml, base, memo0, G, cfg, script=None):
    """G episodes from one book (base, an IntraWorld before session d's first look, and its notes memo0). script: a
    function hhmm -> call for a scripted policy (no sampling; eval4's hold baseline). Returns the scored members."""
    feats, sigma_m = data.features(d)
    got = prov(d) if prov else None
    fc_rows = got[1]["rows"] if got else {}
    specs = tools.available(intrasim.IntraGateway(data, d, pre, intra, "09:00"))
    if ml:
        import forecast as fc
        specs = specs + [fc.SPEC]
    names = {s["name"] for s in specs} | {"submit"}
    r1 = {k: v["range1_pct"] / 100 for k, v in fc_rows.items() if v.get("range1_pct")}
    worlds = [base.clone() for _ in range(G)]
    fwd = min(cfg.forward, max(0, 4 - iso(data, d).weekday()))     # the sessions left before Monday's flatten (25f93cc)
    ledgers = [intrasim.Ledger(wd, data, intra, d, feats, sigma_m, fc_rows, forward=fwd) for wd in worlds]
    memos = [dict(memo0) for _ in range(G)]
    looks = [[] for _ in range(G)]
    marks = [[] for _ in range(G)]
    for li, hhmm in enumerate(cfg.looks):
        prompts, sts = [], []
        for g in range(G):
            ledgers[g].trade_to(worlds[g], hhmm)
            if looks[g]:                             # orders of the last look rejected at fill time (margin)
                looks[g][-1]["rejected"] += worlds[g].rejected
            worlds[g].rejected = 0
            marks[g].append(ledgers[g].mark(worlds[g]))
            system, user, st = build(worlds[g], data, d, hhmm, memos[g], ml)
            sts.append(st)
            prompts.append(train2.prompt_ints(train2.renderer(data, d, cfg.effort), specs + [trader.TOOL_LIVE], system, user))
        if script:
            outs = [{"call": script(hhmm), "seq": [], "mask": [], "lps": [], "trace": [], "calls": 0, "bad": 0,
                     "cost": {"prefill": 0, "cached": 0, "sampled": 0}} for _ in range(G)]
        else:
            firsts = [None] * G
            if li == 0:                              # one prompt for every episode: sample it once
                res = await sc.sample_async(tinker.ModelInput.from_ints(prompts[0]), G, train2.params(cfg.max_tokens, len(prompts[0])))
                firsts = [(list(x.tokens), list(x.logprobs), x.stop_reason) for x in res.sequences]
            gws = [intrasim.IntraGateway(data, d, pre, intra, hhmm, forecast=prov if ml else None) for _ in range(G)]
            outs = await asyncio.gather(*(run_look(sc, gws[g], names, prompts[g], cfg, firsts[g]) for g in range(G)))
            if li == 0:
                for g, lk in enumerate(outs):
                    lk["cost"]["prefill" if g == 0 else "cached"] += len(prompts[0])
        for g, lk in enumerate(outs):
            call, st = lk["call"], sts[g]
            before = intrasim.book_summary(worlds[g].m)
            notes = worlds[g].act(call) if call else []
            stats = order_stats(call, st, worlds[g], lk["trace"], r1)
            if call:
                memos[g] = {"when": f"{iso(data, d):%a %d %b} {hhmm} ET", "notes": str(call.get("notes") or "")[:2000]}
            looks[g].append({**lk, "hhmm": hhmm, "nav": st["value"], "gross": st["gross"], "notes_placed": notes, "book": before,
                             **stats, "rejected": worlds[g].rejected,
                             "forced": any("research limit" in str(t.get("result", "")) for t in lk["trace"])})
            worlds[g].rejected = 0
    members = []
    for g in range(G):
        parts = ledgers[g].close(worlds[g])
        looks[g][-1]["rejected"] += worlds[g].rejected     # fill-time margin rejections after the last look
        for li, lk in enumerate(looks[g]):
            lk["P"] = marks[g][li]
            lk["penalty"] = (MALFORMED * (lk["call"] is None) + intrasim.BAD_TURN * lk["bad"]
                             + SMALL_ENTRY * lk["small_entries"] + UNRESEARCHED * lk["unresearched"]
                             + REJECT_PENALTY * min(lk["rejected"], REJECT_CAP) + EMPTY_COST * lk["empty"])
        later = parts["carry_charge"]                     # the reward-to-go of each look, from the last look back
        for lk in reversed(looks[g]):
            later += lk["penalty"]
            lk["rtg"] = parts["R_trade"] - lk["P"] - later
        navs = [lk["nav"] for lk in looks[g]] + [parts["nav_close"]]
        pen = sum(lk["penalty"] for lk in looks[g]) + parts["carry_charge"]
        members.append({"R": parts["R_trade"] - pen, **parts, "penalty": pen, "rejected": sum(lk["rejected"] for lk in looks[g]),
                        "worst_step_pp": 100 * min(b / a - 1 for a, b in zip(navs, navs[1:])), "looks": looks[g]})
    return members


def datums_for(win):
    """One datum per look: its reward-to-go minus the group's mean reward-to-go at the same look."""
    out = []
    for li in range(len(win["members"][0]["looks"])):
        lks = [m["looks"][li] for m in win["members"]]
        mean = sum(lk["rtg"] for lk in lks) / len(lks)
        out += [train2.datum(lk["seq"], lk["mask"], lk["lps"], float(lk["rtg"] - mean)) for lk in lks
                if any(lk["mask"]) and abs(lk["rtg"] - mean) > 1e-9]
    return out


def noise_share(wins):
    """R5's noise rule: over looks 2..L of every group, the share of the within-group variance of the reward-to-go that
    the realized overnight part (the closing book's gap to the next open, hedged) carries, pooled."""
    num = den = 0.0
    for w in wins:
        ms = w["members"]
        on = [m["overnight_pp"] for m in ms]
        mo = sum(on) / len(on)
        for li in range(1, len(ms[0]["looks"])):
            r = [m["looks"][li]["rtg"] for m in ms]
            mr = sum(r) / len(r)
            num += sum((x - mo) ** 2 for x in on)
            den += sum((x - mr) ** 2 for x in r)
    return num / den if den else None


def look_rows(wins):
    return [lk for w in wins for m in w["members"] for lk in m["looks"]]


def summarize(step, wins, n_datums, secs, cost):
    mem = [m for w in wins for m in w["members"]]
    lks = look_rows(wins)
    if not mem:
        return {"step": step, "samples": 0, "seconds": secs, "cost_usd": cost}
    n, nl = len(mem), max(1, len(lks))
    avg = lambda key: sum(float(m.get(key, 0)) for m in mem) / n
    lav = lambda key: sum(float(lk.get(key, 0)) for lk in lks) / nl
    used = {}
    for lk in lks:
        for t in lk["trace"]:
            if t.get("tool") and t["tool"] != "submit":
                used[t["tool"]] = used.get(t["tool"], 0) + 1
    orders = sum(lk["orders"] for lk in lks) or 1
    return {"step": step, "episodes": n, "looks": len(lks), "datums": n_datums, "R": avg("R"), "R_trade": avg("R_trade"),
            "book_pp": avg("book_pp"), "hedge_pp": avg("hedge_pp"), "risk_intra_pp": avg("risk_intra_pp"),
            "risk_close_pp": avg("risk_close_pp"), "premium_pp": avg("premium_pp"), "overnight_pp": avg("overnight_pp"),
            "penalty": avg("penalty"), "gross_close": avg("gross_close"), "noise_share": noise_share(wins),
            "malformed_looks": sum(lk["call"] is None for lk in lks) / nl, "bad_turns": lav("bad"),
            "rejected_per_look": sum(m["rejected"] for m in mem) / nl, "research_calls": lav("calls"), "tools": used,
            "forced_submit": lav("forced"),
            "orders_per_look": lav("orders"), "orders_ex_protective_per_look": lav("orders_ex_protective"),
            "protective_per_held_position": sum(lk["protective"] for lk in lks) / max(1, sum(lk["held"] for lk in lks)),
            "m1_big_per_look": lav("m1_big"), "under_1k_share": sum(lk["under_1k"] for lk in lks) / orders,
            "entries_no_stop_share": sum(lk["entries_no_stop"] for lk in lks) / max(1, sum(lk["entries"] for lk in lks)),
            "protective_per_look": lav("protective"),
            "look_top_weight_median": sorted(lk["book"]["top_weight"] for lk in lks)[nl // 2],
            "look_unprotected_75": sum(lk["book"]["unprotected_75"] for lk in lks) / nl,
            "close_unprotected_75": sum(m["at_close"]["unprotected_75"] for m in mem) / n,
            "entries_per_look": lav("entries"),
            "close_top_weight_median": sorted(m["at_close"]["top_weight"] for m in mem)[n // 2],
            "look_gross": lav("gross"), "stop_type_share": sum(lk["stop_type"] for lk in lks) / orders,
            "small_entries_per_look": lav("small_entries"), "unresearched_per_look": lav("unresearched"),
            "empty_share": lav("empty"), "token_orders_per_look": lav("token_orders"), "far_orders_per_look": lav("far_orders"),
            "mean_order_notional": sum(lk["notional"] for lk in lks) / orders,
            "limit_dist_r1_median": _median([x for lk in lks for x in lk["limit_dist_r1"]]), "entry_notional_nav_per_look": lav("entry_notional_nav"), "forced_empty_share": sum(lk["forced"] and lk["empty"] for lk in lks) / nl,
            "notes_share": lav("notes"), "no_probs_share": sum(lk["no_probs"] for lk in lks) / orders,
            "carried_unprotected_share": sum(m["carried_unprotected"] for m in mem) / max(1, sum(m["carried"] for m in mem)),
            "carry_charge": avg("carry_charge"), "R_sd": (sum((m["R"] - avg("R")) ** 2 for m in mem) / n) ** 0.5,
            "R_pnl": sum(m["book_pp"] - m["hedge_pp"] for m in mem) / n,
            "look_tokens": sum(lk["cost"]["sampled"] for lk in lks) / nl,
            "context_full": sum(any(t.get("error") == "context full" for t in lk["trace"]) for lk in lks) / nl,
            "ml_groups": sum(1 for w in wins if w["ml"]), "seconds": secs, "cost_usd": cost}


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def usd(c, trained=0, update=True):
    return (c["prefill"] * PRICE["prefill"] + c["cached"] * PRICE["cached"] + c["sampled"] * PRICE["sample"]
            + (trained * PRICE["train"] if update else 0)) / 1e6


def pool(data, first, last):
    """Sessions in [first, last] with a next session (the liquidation open) and room for the longest burn-in."""
    return [d for d in range(len(data.dates)) if first <= data.dates[d] <= last and d - 6 >= 260 and d + 1 < len(data.dates)]


async def main_async(cfg):
    out = ROOT / "runs" / "train" / cfg.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(cfg), indent=1))
    data = trader.Data()
    pre = simgw.Precomputed(data)
    intra = intrasim.Intra(data)
    import forecast as fc
    prov = fc.sim_provider(data)
    dates = [d for d in pool(data, TRAIN_FIRST, TRAIN_LAST) if intra.has(d)]
    service = tinker.ServiceClient(project_id=PROJECT_ID, user_metadata={"purpose": f"train:{cfg.tag}"})
    ckpts = out / "checkpoints.jsonl"
    start = 0
    if cfg.resume:
        rows = [json.loads(l) for l in ckpts.read_text().splitlines() if l.strip()]
        tc = await service.create_training_client_from_state_with_optimizer_async(rows[-1]["state"])
        start = rows[-1]["step"]
        for f in (out / "metrics.jsonl", out / "samples.jsonl"):
            if f.exists():
                keep = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l)["step"] <= start]
                f.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
    else:                                   # weights only: a fresh AdamW (rl3's second moments are on a ~8x larger reward scale)
        tc = await service.create_training_client_from_state_async(cfg.init)
    print(f"{'resumed at step ' + str(start) if start else 'starting from ' + cfg.init}; {len(dates)} training dates "
          f"{data.dates[dates[0]]}..{data.dates[dates[-1]]}", flush=True)
    adam = types.AdamParams(learning_rate=cfg.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    rng = random.Random(cfg.seed)
    draw = lambda: (rng.sample(dates, cfg.groups), [rng.randint(cfg.burn_min, cfg.burn) for _ in range(cfg.groups)],
                    [rng.random() < cfg.ml_frac for _ in range(cfg.groups)],
                    [round(rng.uniform(LOSS_LO, LOSS_HI), 4) for _ in range(cfg.groups)])
    for _ in range(start):
        draw()
    hist = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines() if l.strip()] if start else []
    rows = cfg.steps + (1 if cfg.measure_last else 0)
    for step in range(start, rows):
        t0 = time.time()
        update = step < cfg.steps and not cfg.no_update      # row step+1 samples checkpoint `step` (rl5: the last row only measures)
        ds, ks, mls, gaps = draw()
        sc = await tc.save_weights_and_get_sampling_client_async()
        res = await asyncio.gather(*(group(sc, data, pre, intra, prov, d, k, ml, gp, cfg) for d, k, ml, gp in zip(ds, ks, mls, gaps)),
                                   return_exceptions=True)
        errs = [w for w in res if isinstance(w, Exception)]
        wins = [w for w in res if not isinstance(w, Exception)]
        for e in errs[:3]:
            print("group error:", repr(e)[:600], flush=True)
        datums = [x for w in wins for x in datums_for(w)]
        if datums and update:
            fb = await tc.forward_backward_async(datums, loss_fn=cfg.loss)
            op = await tc.optim_step_async(adam)
            await fb.result_async()
            await op.result_async()
        c = {"prefill": 0, "cached": 0, "sampled": 0}
        for w in wins:
            c["prefill"] += w["burn_cost"]["prefill"]
            c["sampled"] += w["burn_cost"]["sampled"]
            for lk in look_rows([w]):
                for key in c:
                    c[key] += lk["cost"][key]
        cost = usd(c, sum(x.model_input.length for x in datums), update)
        row = summarize(step + 1, wins, len(datums), time.time() - t0, cost)
        row["group_errors"] = len(errs)
        row["updated"] = bool(datums and update)
        row["slots"] = slot_stats(look_rows(wins))
        row["adv_sd_by_slot"] = adv_sd_by_slot(wins)
        hist.append(row)
        row["drift"] = drift(row, hist[0])
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        with (out / "samples.jsonl").open("a") as f:
            for w in wins:
                for m in w["members"]:
                    f.write(json.dumps({"step": step + 1, "date": w["date"], "burn": w["burn"], "ml": w["ml"], "loss": w["loss"],
                                        **{k: v for k, v in m.items() if k != "looks"},
                                        "looks": [{k: v for k, v in lk.items() if k not in ("seq", "mask", "lps")} for lk in m["looks"]]},
                                       default=float) + "\n")
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items() if k != "slots"}), flush=True)
        if update and not cfg.no_save and (step + 1) % cfg.save_every == 0:
            name = f"{cfg.tag}-step{step + 1:03d}"
            st = await (await tc.save_state_async(name, ttl_seconds=cfg.state_ttl_days * DAY)).result_async()
            sp = await (await tc.save_weights_for_sampler_async(name, ttl_seconds=cfg.sampler_ttl_days * DAY)).result_async()
            with ckpts.open("a") as f:
                f.write(json.dumps({"step": step + 1, "state": st.path, "sampler": sp.path}) + "\n")
            print(f"saved {name}: {sp.path}", flush=True)
        cand = candidate(hist, ckpts)
        (out / "candidate.json").write_text(json.dumps(cand, indent=1))
        # drift guard (R5, criteria v1.1, as pre-registered in rl5_prereg.md): from row 3 on, two consecutive rows with any
        # key drifted, overall or in any look slot, stop the run; the candidate is then the last eligible checkpoint
        if len(hist) >= 3 and hist[-1]["drift"] and hist[-2]["drift"] and hist[-2]["step"] >= 2:
            print(f"DRIFT GUARD: {hist[-2]['drift']} at row {step} and {row['drift']} at row {step + 1}; stopping", flush=True)
            (out / "STOPPED").write_text(json.dumps({"row": step + 1, "drift": row["drift"], "candidate": cand}))
            break
        if cfg.cap_usd and _spent(out) > cfg.cap_usd:
            print(f"SPEND CAP: {_spent(out):.2f} > {cfg.cap_usd}; stopping", flush=True)
            (out / "STOPPED").write_text(json.dumps({"row": step + 1, "spend_cap": cfg.cap_usd, "candidate": cand}))
            break
    print("CANDIDATE:", json.dumps(candidate(hist, ckpts)), flush=True)


DRIFT_KEYS = ("research_calls", "orders_ex_protective", "empty")


def slot_stats(lks):
    """Per look slot and overall ("all"): (mean, sd, n) of research calls and orders excluding protective stops per look,
    and (share, n) of empty looks."""
    import statistics as stx
    by = {}
    for lk in lks:
        for key in (lk["hhmm"], "all"):
            by.setdefault(key, []).append(lk)
    out = {}
    for slot, xs in by.items():
        r = {}
        for k in ("calls", "orders_ex_protective"):
            v = [float(x[k]) for x in xs]
            r["research_calls" if k == "calls" else k] = [stx.mean(v), stx.pstdev(v), len(v)]
        r["empty"] = [sum(bool(x["empty"]) for x in xs) / len(xs), None, len(xs)]
        r["gross"] = [stx.mean(float(x["gross"]) for x in xs), None, len(xs)]
        out[slot] = r
    return out


def drift(row, first):
    """Keys drifted against row 1, overall or in any slot: a change of more than 30% of row 1's value that is also more than
    two standard errors of the difference (look-level sd for means, binomial for the empty share; 96 looks a slot make the
    empty share's slot noise about 3 points, so 30% alone would trip on noise); plus mean gross at looks overall up more
    than 30% or down more than 60% (criteria v1.1)."""
    out = []
    for slot, r in row["slots"].items():
        f = first["slots"].get(slot)
        if not f:
            continue
        for k in DRIFT_KEYS:
            (m2, s2, n2), (m1, s1, n1) = r[k], f[k]
            if k == "empty":
                se = ((m1 * (1 - m1)) / n1 + (m2 * (1 - m2)) / n2) ** 0.5
            else:
                se = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 0.5
            if abs(m2 - m1) > 0.30 * max(abs(m1), 0.01) and abs(m2 - m1) > 2 * se:
                out.append(f"{k}@{slot}")
    g1, g2 = first["slots"]["all"]["gross"][0], row["slots"]["all"]["gross"][0]
    if g1 and not (0.40 <= g2 / g1 <= 1.30):
        out.append("gross@all")
    return out


def candidate(hist, ckpts):
    """Checkpoint k is eligible only if rows 2..k+1 are all in bounds (row k+1 samples checkpoint k; criteria review, 24
    Oct); the candidate is the highest eligible saved checkpoint, or none."""
    saved = {json.loads(l)["step"]: json.loads(l) for l in ckpts.read_text().splitlines() if l.strip()} if ckpts.exists() else {}
    clean = {h["step"]: not h["drift"] for h in hist}
    best = None
    for k in sorted(saved):
        if all(clean.get(r, False) for r in range(2, k + 2)):
            best = k
    return {"checkpoint": best, **(saved.get(best) or {}), "rows_seen": len(hist)}


def adv_sd_by_slot(wins):
    """Within-group sd of the reward-to-go at each look slot, pooled over groups (R5's tracked advantage scale)."""
    acc = {}
    for w in wins:
        ms = w["members"]
        for li, lk0 in enumerate(ms[0]["looks"]):
            r = [m["looks"][li]["rtg"] for m in ms]
            mu = sum(r) / len(r)
            acc.setdefault(lk0["hhmm"], []).extend((x - mu) ** 2 for x in r)
    return {k: round((sum(v) / len(v)) ** 0.5, 4) for k, v in acc.items()}


def _spent(out):
    return sum(json.loads(l).get("cost_usd", 0) for l in (out / "metrics.jsonl").read_text().splitlines() if l.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--groups", type=int, default=12)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--burn", type=int, default=5)
    ap.add_argument("--burn-min", type=int, default=1)
    ap.add_argument("--looks", nargs="+", default=intrasim.LOOKS)
    ap.add_argument("--forward", type=float, default=intrasim.FORWARD)
    ap.add_argument("--ml-frac", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--loss", default="ppo")
    ap.add_argument("--max-tokens", type=int, default=agent.MAX_OUT)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--measure-last", action="store_true", help="one more row after the last update, sampling the final "
                    "checkpoint without updating, so it can be eligible (its row k+1)")
    ap.add_argument("--state-ttl-days", type=int, default=30)
    ap.add_argument("--sampler-ttl-days", type=int, default=45)
    ap.add_argument("--cap-usd", type=float, default=0.0, help="stop after a step once the run's estimated spend passes this")
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--no-update", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default=RL3_STATE)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
