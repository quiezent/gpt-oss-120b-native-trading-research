"""Checks on data/edgar/ (after edgar_fetch.py): time-zone evidence, coverage, counts. Prints a Markdown report.

    .venv\\Scripts\\python.exe ml_research\\edgar_check.py
"""

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "edgar"


def load():
    with open(OUT / "filings.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def tz_index(rows):
    """Compare our converted ET time with the filing index page's 'Accepted' (EDGAR's own Eastern time)."""
    n = same = 0
    worst = []
    naive_same = 0
    for r in rows:
        if not r.get("accepted_et_index"):
            continue
        n += 1
        a = datetime.strptime(r["accepted_et"], "%Y-%m-%d %H:%M:%S")
        b = datetime.strptime(r["accepted_et_index"], "%Y-%m-%d %H:%M:%S")
        if a == b:
            same += 1
        else:
            worst.append((abs((a - b).total_seconds()), r["accessionNumber"], r["accepted_et"], r["accepted_et_index"]))
        raw = datetime.strptime(r["acceptanceDateTime"][:19], "%Y-%m-%dT%H:%M:%S")
        if raw == b:
            naive_same += 1
    return n, same, naive_same, sorted(worst, reverse=True)[:5]


def bars():
    out = {}
    for p in (ROOT / "data" / "bars").glob("*.json"):
        b = json.loads(p.read_text())["bars"]
        out[p.stem.replace("_", " ")] = {f"{x['date'][:4]}-{x['date'][4:6]}-{x['date'][6:8]}": x for x in b}
    return out


def gap_check(rows, B):
    """Median |overnight gap| into the session we say first reflects a 2.02, vs. the session one earlier and later,
    and vs. all sessions. If the time zone were off, the big gap would sit on a neighbouring session."""
    spy = sorted(B["SPY"])
    pos = {d: i for i, d in enumerate(spy)}

    def gap(sym, d):
        b = B.get(sym)
        if not b or d not in pos or pos[d] == 0:
            return None
        prev = spy[pos[d] - 1]
        if d not in b or prev not in b:
            return None
        return abs(np.log(b[d]["open"] / b[prev]["close"]))
    res = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if "2.02" not in r["items"].split(",") or not r["first_session"] or r["first_session"] not in pos:
            continue
        i = pos[r["first_session"]]
        for sym in r["symbols"].split("|"):
            for k, off in (("first_session", 0), ("one_before", -1), ("one_after", 1)):
                if 0 < i + off < len(spy):
                    g = gap(sym, spy[i + off])
                    if g is not None:
                        res[r["session"]][k].append(g)
    allg = [gap(s, d) for s in list(B)[:200] for d in spy[1:]]
    allg = [g for g in allg if g is not None]
    return res, float(np.median(allg))


def main():
    rows = load()
    cmap = json.loads((OUT / "cik_map.json").read_text())
    comp = json.loads((OUT / "companies.json").read_text())
    earn = json.loads((OUT / "earnings_dates.json").read_text())["symbols"]
    print(f"## Counts\n\n- companies (non-ETF symbols): {len(cmap)}; unique CIKs: {len({v['cik'] for v in cmap.values()})}")
    print(f"- 8-K + 8-K/A filings since 2023-09-01: {len(rows)} (8-K/A: {sum(r['form'] == '8-K/A' for r in rows)})")
    items = Counter(it for r in rows for it in (r["items"] or "").split(",") if it)
    print("- by item: " + ", ".join(f"{k} {v}" for k, v in sorted(items.items())))
    st = Counter(r["text_status"] for r in rows if r["text_status"])
    print(f"- text targets (2.02/7.01/8.01): {sum(st.values())}; status {dict(st)}; text files {len(list((OUT / 'text').glob('*.txt')))}")
    kinds = Counter(r["text_kind"] for r in rows if r["text_status"] == "ok")
    print(f"- text source: {dict(kinds)}")
    e202 = [r for r in rows if "2.02" in r["items"].split(",")]
    print(f"- 2.02 filings: {len(e202)}; with text: {sum(r['text_status'] == 'ok' for r in e202)}")
    sess = Counter(r["session"] for r in e202)
    print(f"- 2.02 by session: {dict(sess)}")

    n, same, naive, worst = tz_index(rows)
    print(f"\n## Time zone\n\n- filings with an index-page 'Accepted' time: {n}; our ET == index ET: {same}; "
          f"raw field read as ET == index ET: {naive}")
    if worst:
        print(f"- largest mismatches: {worst}")

    B = bars()
    res, allmed = gap_check(rows, B)
    print(f"\n## Price check (median |open/prev close - 1| in bps; all sessions {allmed * 1e4:.0f})\n")
    print("| session | n | first_session | one before | one after |\n|---|---|---|---|---|")
    for s, d in res.items():
        print(f"| {s} | {len(d['first_session'])} | {np.median(d['first_session']) * 1e4:.0f} | "
              f"{np.median(d['one_before']) * 1e4:.0f} | {np.median(d['one_after']) * 1e4:.0f} |")

    # coverage: 2.02 per calendar quarter, 2023Q4..2026Q3, for quarters in which the stock has bars
    print("\n## Coverage\n")
    qs = [f"{y}Q{q}" for y in (2023, 2024, 2025, 2026) for q in (1, 2, 3, 4) if "2023Q4" <= f"{y}Q{q}" <= "2026Q3"]
    src = Counter(e.get("source") for v in earn.values() for e in v)
    print(f"- earnings_dates.json entries: {sum(src.values())} by source {dict(src)}")
    es = Counter(e["session"] for v in earn.values() for e in v)
    print(f"- earnings_dates.json by session: {dict(es)}")
    dist = Counter(len(v) for v in earn.values())
    print(f"- 2.02 count per symbol since 2023-09-01: {dict(sorted(dist.items()))}")
    # gaps: > 120 days between consecutive 2.02s, or from the first bar (>= 2023-10-16) / to 2026-10-16, while the stock traded
    from datetime import date
    gaps = {}
    for s, ev in earn.items():
        b = sorted(B.get(s, {}))
        if not b:
            continue
        lo, hi = max(b[0], "2023-10-16"), "2026-10-16"
        pts = [lo] + sorted(e["accepted_et"][:10] for e in ev if lo <= e["accepted_et"][:10] <= hi) + [hi]
        long = [(a, c) for a, c in zip(pts, pts[1:]) if (date.fromisoformat(c) - date.fromisoformat(a)).days > 135]
        if long:
            gaps[s] = long
    print(f"- symbols with a stretch of more than 135 days without a 2.02 while trading (2023-10-16..2026-10-16): {len(gaps)}")
    for s, m in sorted(gaps.items()):
        print(f"  - {s} ({cmap[s]['title']}): " + ", ".join(f"{a}..{c}" for a, c in m))
    conv = Counter(c.get("raw_convention") for c in comp.values())
    print(f"- acceptanceDateTime convention by company (symbols): {dict(conv)}")
    src = Counter(r["time_source"] for r in rows)
    print(f"- time source of filings: {dict(src)}")
    pred = {s: c["predecessors"] for s, c in comp.items() if c.get("predecessors")}
    print(f"- predecessor CIKs merged: {pred}")


if __name__ == "__main__":
    main()
