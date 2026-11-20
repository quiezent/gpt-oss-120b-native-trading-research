"""Dividends in the sleeve books (F6 of runs/ml/research/sgov_tflo_framework_probe.md; Clayton, 26 Oct: "Yes but today
26 Oct not at 2 Nov"). Until now a sleeve's book took the ex-dividend price drop and never the dividend, so every long
held over an ex-date was understated and every short overstated. A sleeve holding a position at the close before an
ex-date is now credited (long) or charged (short) quantity x dividend, in runs/dividends.jsonl, which paper.book adds
to cash and realized P&L; the model is told in its next look's notes.

The ex-date and amount come from IBKR's own daily bars: the dividend-adjusted (ADJUSTED_LAST) close over the raw
(TRADES) close steps at each ex-date, and the step gives the amount, raw close before x (1 - factor before / factor
after). Checked 26 Oct against published distributions: SGOV 0.3005 and TFLO 0.1596 (ex 1 Oct), T 0.2775 (9 Oct),
PG 1.0885 (23 Oct). A step worth more than 20% of the price (bad data, or a split in a series IBKR has not adjusted) is
logged and not booked. IBKR's TRADES bars are split-adjusted as well (SOXX 3-for-1, ex 5 Nov 2026: no step was logged at the
16:15 record or the next morning, and the cached bars before 5 Nov are a third of the prices traded), so a split shows no step.

Splits (book_splits, run first by book): on 5 Nov the account held SOXX in post-split shares from the morning while the
books kept pre-split shares all day. A split is booked in runs/splits.jsonl, before the first look of its ex-date, when two
signals agree: the account's position over the books' (at the close before, plus today's fills) is a split ratio, and the
price of the latest earlier look over IBKR's re-based close of that day shows the same ratio (a news gap reads 1 there).
Otherwise one loud line is logged and nothing is booked.

    python dividends.py            # book what is due now (client 10; outside looks)
"""
import json
import math
from datetime import datetime
from fractions import Fraction
from zoneinfo import ZoneInfo

import paper

ET = ZoneInfo("America/New_York")
LOOKBACK = "1 M"          # ex-dates booked up to a month late (a missed run is caught up); each is booked once
MAX_SHARE = 0.20

# Split ratios, new shares per old: n-for-1, 3-for-2 and 5-for-4, and 1-for-n for the same n.
SPLIT_N = (2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 50)
RATIOS = ({Fraction(n): f"{n}-for-1" for n in SPLIT_N} | {Fraction(3, 2): "3-for-2", Fraction(5, 4): "5-for-4"}
          | {Fraction(1, n): f"1-for-{n}" for n in SPLIT_N})
# A price ratio shows split r when, of r, the other ratios and 1 (no split), r is the nearest (in log terms) and within 15%.
# The nearest-ratio rule keeps neighbours apart and 15% caps the rest: 3 is read for 2.55-3.45, 2 for 1.73-2.30, 3-for-2 for
# 1.37-1.72, 5-for-4 for 1.12-1.37. 15% leaves room for the move on top of the split (SOXX's 5 Nov close was 12% under a
# third of its 4 Nov close; its 4 Nov 15:30 look price, 447.50, over the 5 Nov close is 3.39).
PRICE_TOL = 0.15
LOOK_DIR = paper.RUNS / "live"   # look records <YYYY-MM-DD>/<sleeve>-<HHMM>.json: the prices a model saw, as they were
SUSPECTED = set()                # (symbol, day) already logged loud in this process: once a day per symbol
RESIDUE_NAMES = 3                # the account listing this many names no sleeve holds is a reset's residue (the TWS Gateway
                                 # Owner, 15 Nov: after a paper cash reset the Gateway can list the removed positions for a
                                 # session or two, and a stale quantity can fall on a split ratio by chance): its
                                 # quantities are then no split signal, and nothing is booked that day
RESIDUE_SAID = set()             # days that line was logged


def detect(gw, symbol):
    """[(ex_date YYYYMMDD, dividend a share, raw close before)] in the lookback, from the adjusted/raw close ratio."""
    bars = lambda what: {str(b["date"])[:8]: b["close"] for b in
                         gw.bars(symbol, duration=LOOKBACK, bar_size="1 day", what=what, rth=True) if b.get("close")}
    raw, adj = bars("TRADES"), bars("ADJUSTED_LAST")
    ds = sorted(set(raw) & set(adj))
    f = {d: adj[d] / raw[d] for d in ds}
    return [(b, raw[a] * (1 - f[a] / f[b]), raw[a]) for a, b in zip(ds, ds[1:]) if abs(f[a] / f[b] - 1) > 1e-4]


def held_overnight(sleeve, today):
    """Symbols the sleeve held at any close this week, through the close before today."""
    days = sorted({paper.et_date(f["time"]) for f in paper.fills() if f.get("sleeve") == sleeve} | {today})
    return set().union(*(paper.positions_at(sleeve, d) for d in days[1:])) if len(days) > 1 else set()


def price_class(x):
    """The split ratio (or 1, no split) a price ratio x shows, or None (see PRICE_TOL)."""
    if not x or x <= 0 or not math.isfinite(x):
        return None
    best = min(list(RATIOS) + [Fraction(1)], key=lambda r: abs(math.log(x / r)))
    return best if abs(x / best - 1) <= PRICE_TOL else None


def label(r):
    return RATIOS.get(r, "none") if r != 1 else "no split"


def looked_close(symbol, today):
    """(date, price) of the symbol's latest position price in a look record of the latest earlier day that has one (up to
    5 days back), or None. A look record is written at the look, so its price is not re-based by a later split."""
    days = sorted((p for p in LOOK_DIR.glob("????-??-??") if p.is_dir() and p.name.replace("-", "") < today), reverse=True)
    for p in days[:5]:
        for f in sorted(p.glob("*-[0-9][0-9][0-9][0-9].json"), key=lambda f: f.stem[-4:], reverse=True):
            try:
                st = json.loads(f.read_text(encoding="utf-8")).get("state") or {}
            except (OSError, ValueError):
                continue
            for x in st.get("positions") or []:   # a position without a mark shows its average cost: not a price
                if x.get("symbol") == symbol and x.get("close") and abs(x["close"] - (x.get("avg_cost") or 0)) > 1e-9:
                    return p.name.replace("-", ""), float(x["close"])
    return None


def price_ratios(gw, symbol, today, seen, quote=True):
    """([(what, price ratio, kind)] of prices before today over prices now, the price now, IBKR's close of the look's day).
    kind "rebase": the look price over IBKR's TRADES close of the same day. A look record keeps the price as it was; IBKR
    re-bases its history for a split (SOXX's 4 Nov close read 150.14, a third of 450.41, at the 07:30 refresh of 5 Nov), so
    this shows the split, and about 1 at a gap. kind "quote": the look price over the quote, which shows the split and a gap
    alike. The rest (kind None) are logged and decide nothing. `seen` collects what could not be read. quote=False: no quote."""
    out = []
    try:
        q = (gw.quotes([symbol]) or [{}])[0] if quote else {}
        if q.get("error"):
            raise RuntimeError(q["error"])
    except Exception as e:                          # noqa: BLE001
        seen.append(f"quote failed ({type(e).__name__}: {str(e)[:80]})")
        q = {}
    mid = (q["bid"] + q["ask"]) / 2 if q.get("bid") and q.get("ask") else None
    now = q.get("last") if q.get("last") and (q.get("last") != q.get("close") or not mid) else mid   # a last equal to
    # the previous close is no trade since it: the bid/ask is then the price now
    bars = {}
    for what in ("TRADES", "ADJUSTED_LAST"):
        try:
            bars[what] = {str(b["date"])[:8]: b["close"] for b in
                          gw.bars(symbol, duration="10 D", bar_size="1 day", what=what, rth=True) if b.get("close")}
        except Exception as e:                      # noqa: BLE001
            seen.append(f"{what} bars failed ({type(e).__name__}: {str(e)[:80]})")
            bars[what] = {}
    raw, adj = bars["TRADES"], bars["ADJUSTED_LAST"]
    prev = max((d for d in raw if d < today), default=None)
    looked = looked_close(symbol, today)
    base = raw.get(looked[0]) if looked else None
    if looked and base:
        out.append((f"look price {looked[1]:.2f} / IBKR close {base:.2f} on {looked[0]}", looked[1] / base, "rebase"))
    if looked and now:
        out.append((f"look price {looked[1]:.2f} on {looked[0]} / quote {now:.2f}", looked[1] / now, "quote"))
    if prev and now:
        out.append((f"IBKR close {raw[prev]:.2f} on {prev} / quote {now:.2f}", raw[prev] / now, None))
    if q.get("close") and now:
        out.append((f"quote's close {q['close']:.2f} / quote {now:.2f}", q["close"] / now, None))
    if prev and today in raw and prev in adj and today in adj:
        out.append((f"adjusted/raw factor step on {today}",
                    (adj[today] / raw[today]) / (adj[prev] / raw[prev]), None))
    if quote and not now:
        seen.append("no quote price")
    if not looked:
        seen.append("no look price before today")
    elif not base:
        seen.append(f"no IBKR close on {looked[0]}")
    return out, now, base


def book_splits(gw, today=None, log=print):
    """Book a split of a held symbol with today's ex-date when the account's quantity and IBKR's re-basing both show its
    ratio; log one loud line (once a day per symbol) when only one of them does, or the quote reads another ratio. Never
    raises. Returns the rows booked. On a day the account lists RESIDUE_NAMES or more names no sleeve holds (a reset's
    residue) nothing is booked: one line says so, and a name whose re-basing shows a ratio still gets its loud line; a
    quantity mismatch alone is then the reset and is not logged per name."""
    today = today or datetime.now(ET).strftime("%Y%m%d")
    booked = []
    try:
        sleeves = paper.sleeves()
        prior = {}                                  # symbol -> {sleeve: quantity at the close before today}
        for s in sleeves:
            for sym, q in paper.positions_at(s, today).items():
                prior.setdefault(sym, {})[s] = q
        todays = {}                                 # symbol -> the sleeves' net shares filled today (in new shares)
        for f in paper.fills():
            if f.get("sleeve") in sleeves and paper.et_date(f["time"]) == today:
                todays[f["symbol"]] = todays.get(f["symbol"], 0) + (f["shares"] if f["side"] == "BOT" else -f["shares"])
        if not prior:
            return booked
        try:
            acct = {}
            for p in gw.positions():
                if p.get("quantity"):
                    acct[p["symbol"]] = acct.get(p["symbol"], 0) + p["quantity"]
        except Exception as e:                      # noqa: BLE001
            log(f"splits: account positions failed ({type(e).__name__}: {str(e)[:120]}); no split check")
            return booked
        have = {(r["symbol"], r["ex_date"]) for r in paper.read(paper.SPLITS)}
        unheld = sorted(sym for sym in acct if sym not in prior and sym not in todays)   # listed, in no sleeve's book
        residue = len(unheld) >= RESIDUE_NAMES
        if residue and today not in RESIDUE_SAID:
            RESIDUE_SAID.add(today)
            log(f"splits: reset, not split: the account lists {len(unheld)} names no sleeve holds ({', '.join(unheld[:6])}"
                f"{', ...' if len(unheld) > 6 else ''}); its quantities are no split signal, so no split is booked today")
    except Exception as e:                          # noqa: BLE001
        log(f"splits: check failed ({type(e).__name__}: {str(e)[:150]})")
        return booked
    for sym in sorted(prior):
        try:
            row = _check(gw, sym, prior[sym], todays.get(sym, 0), acct.get(sym, 0), today, have, log, residue)
        except Exception as e:                      # noqa: BLE001 - one symbol never stops the rest
            log(f"splits: {sym} check failed ({type(e).__name__}: {str(e)[:150]})")
            continue
        if row:
            booked.append(row)
    return booked


def _check(gw, sym, by_sleeve, filled, account, today, have, log, residue=False):
    total = sum(by_sleeve.values())
    held = account - filled                         # the account's shares from before today, after the split
    if (sym, today) in have or (total and held == total):
        return None
    cands = [r for r in RATIOS if int(Fraction(total) * r) == held] if total and held else []
    seen = []
    # Without a quantity ratio (shares no sleeve holds, books that net to zero, an account at 0) only IBKR's re-basing is
    # read, so a real split there is logged loud; a mismatch it does not show stays reconcile's POSITION MISMATCH.
    ratios, now, base = price_ratios(gw, sym, today, seen, quote=bool(cands))
    shown = [(w, x, price_class(x), kind) for w, x, kind in ratios]
    rebase = next((c for _, _, c, k in shown if k == "rebase"), None)
    quoted = [c for _, _, c, k in shown if k == "quote"]
    qty = (f"account {account:+g} ({filled:+g} filled today) against the books' {total:+g} at the close before "
           f"({', '.join(f'{s} {q:+g}' for s, q in sorted(by_sleeve.items()))}): "
           f"{' or '.join(RATIOS[c] for c in cands) or 'no split ratio'}")
    prices = "; ".join(f"{w} = {x:.3f} ({label(c) if c is not None else 'no ratio'})" for w, x, c, _ in shown) or "none"
    # Both needed: the quantity's ratio, and the same ratio in IBKR's re-basing (a gap reads 1 there, a split its ratio). A
    # quote that reads another ratio, or none, stops it; a quote that reads 1 against a re-based history is a print from
    # before the split, and is not used.
    stale = bool(quoted) and quoted[0] == 1
    split = rebase in cands and (not quoted or quoted[0] in (rebase, 1))
    if not split or residue:        # residue: a reset's stale positions are listed, so the quantity is no signal today
        if (sym, today) not in SUSPECTED and (rebase not in (None, 1) or (cands and not residue)):
            SUSPECTED.add((sym, today))
            log(f"SPLIT SUSPECTED, NOT BOOKED: {sym} on {today}: quantity {qty}; prices {prices}"
                + ("; the account lists a reset's residue, so nothing is booked today" if residue else "")
                + (f"; {', '.join(seen)}" if seen else ""))
        return None
    r = rebase
    rem = float(Fraction(total) * r - held)
    row = {"symbol": sym, "ex_date": today, "ratio": r.numerator if r.denominator == 1 else float(r), "split": RATIOS[r],
           "source": f"dividends.book_splits: quantity {qty}; prices {prices}", "booked": paper.now_iso()}
    if rem:                                         # the price now, or IBKR's re-based close when the quote is stale
        row["cash_price"] = round(now if quoted and not stale else base, 4)
    paper.append(paper.SPLITS, row)
    have.add((sym, today))
    log(f"SPLIT BOOKED: {sym} {RATIOS[r]} ex {today}: quantity {qty}; prices {prices}"
        + ("; the quote is a price from before the split and is not used" if stale else "")
        + (f"; the account's {rem:+g} share left over from the books' total is paid in cash by the broker" if rem else "")
        + (f"; {', '.join(seen)}" if seen else ""))
    day = datetime.strptime(today, "%Y%m%d")
    for s, q in sorted(by_sleeve.items()):
        whole, frac = paper.split_qty(q, row["ratio"])
        text = f"{sym} split {RATIOS[r]} effective {day:%a} {day.day} {day:%b}: your {q:+g} shares are now {whole:+g}."
        if frac:
            px = row.get("cash_price") or paper.book(s)["positions"].get(sym, {}).get("avg_cost")
            text += (f" The {frac:+g} share left over was settled in cash at ${px:,.2f} a share ({frac * px:+,.2f} to your cash)."
                     if px else f" The {frac:+g} share left over was settled in cash.")
        paper.add_notes(s, [text])
    return row


def book(gw, today=None, log=print):
    """Book each split of today (book_splits), then each dividend not yet booked, with an ex-date on or before today, on
    the position each sleeve held at the close before it. Runs at the 07:30 refresh and the 16:15 record; idempotent.
    Returns the dividend rows booked."""
    today = today or datetime.now(ET).strftime("%Y%m%d")
    book_splits(gw, today, log)
    splits = paper.splits()
    have = {(r["sleeve"], r["symbol"], r["ex_date"]) for r in paper.read(paper.DIVIDENDS)}
    who = {}
    for s in paper.sleeves():
        for sym in held_overnight(s, today):
            who.setdefault(sym, []).append(s)
    booked = []
    for sym in sorted(who):
        try:
            steps = detect(gw, sym)
        except Exception as e:                      # noqa: BLE001 - one symbol's bars never stop the rest
            log(f"dividends: {sym} bars failed ({type(e).__name__}: {str(e)[:120]})")
            continue
        for ex, d, px in steps:
            if ex > today:
                continue
            if not 0 < d <= MAX_SHARE * px:
                log(f"dividends: {sym} step on {ex} of {d:.4f} a share on {px:.2f} not booked (a split or bad data?)")
                continue
            for s in who[sym]:
                q = paper.positions_at(s, ex).get(sym, 0)
                if not q or (s, sym, ex) in have:
                    continue
                k = math.prod(r["ratio"] for r in splits if r["symbol"] == sym and ex <= r["ex_date"] <= today)
                n = q * k                           # the bars are split-adjusted: d is a share of today's, so is n
                row = {"sleeve": s, "symbol": sym, "ex_date": ex, "quantity": q, "per_share": round(d, 4),
                       "amount": round(n * d, 2), "source": "IBKR ADJUSTED_LAST/TRADES close ratio", "booked": paper.now_iso()}
                if k != 1:
                    row["split_adjusted_quantity"] = n
                paper.append(paper.DIVIDENDS, row)
                have.add((s, sym, ex))
                booked.append(row)
                paper.add_notes(s, [f"Dividend: {sym} went ex-dividend on {ex[:4]}-{ex[4:6]}-{ex[6:]} at ${d:.4f} a share; "
                                    f"your {q:+,} shares at the close before"
                                    + (f" ({n:+,g} shares after the split)" if k != 1 else "")
                                    + f" were {'credited' if q > 0 else 'charged'} ${abs(n * d):,.2f} in cash."])
    return booked


if __name__ == "__main__":
    from tws import Gateway
    with Gateway("trader-training", timeout=60) as gw:
        print(book(gw))
