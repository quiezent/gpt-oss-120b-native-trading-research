"""Dividends in the sleeve books (F6 of runs/ml/research/sgov_tflo_framework_probe.md; Clayton, 26 Oct: "Yes but today
26 Oct not at 2 Nov"). Until now a sleeve's book took the ex-dividend price drop and never the dividend, so every long
held over an ex-date was understated and every short overstated. A sleeve holding a position at the close before an
ex-date is now credited (long) or charged (short) quantity x dividend, in runs/dividends.jsonl, which paper.book adds
to cash and realized P&L; the model is told in its next look's notes.

The ex-date and amount come from IBKR's own daily bars: the dividend-adjusted (ADJUSTED_LAST) close over the raw
(TRADES) close steps at each ex-date, and the step gives the amount, raw close before x (1 - factor before / factor
after). Checked 26 Oct against published distributions: SGOV 0.3005 and TFLO 0.1596 (ex 1 Oct), T 0.2775 (9 Oct),
PG 1.0885 (23 Oct). A step worth more than 20% of the price (a split, or bad data) is logged and not booked.

    python dividends.py            # book what is due now (client 10; outside looks)
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import paper

ET = ZoneInfo("America/New_York")
LOOKBACK = "1 M"          # ex-dates booked up to a month late (a missed run is caught up); each is booked once
MAX_SHARE = 0.20


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


def book(gw, today=None, log=print):
    """Book each dividend not yet booked, with an ex-date on or before today, on the position each sleeve held at the
    close before it. Runs at the 07:30 refresh and the 16:15 record; idempotent. Returns the rows booked."""
    today = today or datetime.now(ET).strftime("%Y%m%d")
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
                row = {"sleeve": s, "symbol": sym, "ex_date": ex, "quantity": q, "per_share": round(d, 4),
                       "amount": round(q * d, 2), "source": "IBKR ADJUSTED_LAST/TRADES close ratio", "booked": paper.now_iso()}
                paper.append(paper.DIVIDENDS, row)
                have.add((s, sym, ex))
                booked.append(row)
                paper.add_notes(s, [f"Dividend: {sym} went ex-dividend on {ex[:4]}-{ex[4:6]}-{ex[6:]} at ${d:.4f} a share; "
                                    f"your {q:+,} shares at the close before were {'credited' if q > 0 else 'charged'} "
                                    f"${abs(q * d):,.2f} in cash."])
    return booked


if __name__ == "__main__":
    from tws import Gateway
    with Gateway("trader-training", timeout=60) as gw:
        print(book(gw))
