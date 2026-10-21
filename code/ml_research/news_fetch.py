"""Fetch and cache IBKR historical headlines for the news tool: data/news/<SYM>.json.

Read-only: connects as client "trader-research" and only calls news() and article(); never places or cancels orders.
IBKR's free providers here are BRFG (Briefing.com stock columns), BRFUPDN (Briefing.com analyst actions) and DJNL
(Dow Jones newsletters, no per-stock history). History reaches back about two (BRFUPDN) to three (BRFG) years.

    .venv\\Scripts\\python.exe ml_research\\news_fetch.py            # every symbol in data/universe523.txt
    .venv\\Scripts\\python.exe ml_research\\news_fetch.py AAPL MSFT  # just these
    .venv\\Scripts\\python.exe ml_research\\news_fetch.py --articles  # text of every BRFG/DJNL headline in data/news

Article text (for the simulator's article(), synthesis_wave_c.md S4): data/news/articles/<PROVIDER>/<id>.json with the
fetch time, about 1 request a second, resumable (cached ids are skipped; failures are retried on the next run).
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEWS = ROOT / "data" / "news"
ARTICLES = NEWS / "articles"
TEXT_PROVIDERS = ("BRFG", "DJNL")       # BRFUPDN analyst actions are self-contained headlines
LIMIT, PACE = 300, 1.0


def connect():
    from tws import Gateway
    return Gateway("trader-research", timeout=60)


def fetch_symbol(gw, sym, limit=LIMIT, stop_before=None):
    """All headlines for sym, newest first, paging back with before= while a page comes back full."""
    rows, seen, before = [], set(), ""
    while True:
        page = gw.news(sym, limit=limit, before=before)
        new = [r for r in page if (r["provider"], r["article_id"]) not in seen]
        for r in new:
            seen.add((r["provider"], r["article_id"]))
        rows += new
        if len(page) < limit or not new:
            break
        before = page[-1]["time"][:19].replace("T", " ")
        if stop_before and before < stop_before:
            break
        time.sleep(PACE)
    rows.sort(key=lambda r: r["time"], reverse=True)
    return rows


def save(sym, rows):
    NEWS.mkdir(parents=True, exist_ok=True)
    (NEWS / f"{sym.replace(' ', '_')}.json").write_text(json.dumps({"symbol": sym, "fetched_utc": datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"), "headlines": rows}), encoding="utf-8")


def main(symbols, skip_done=False):
    if skip_done:
        symbols = [s for s in symbols if not (NEWS / f"{s.replace(' ', '_')}.json").exists()]
    gw = connect()
    t0 = time.time()
    for i, sym in enumerate(symbols):
        for attempt in range(4):
            try:
                rows = fetch_symbol(gw, sym)
                save(sym, rows)
                print(f"{i + 1}/{len(symbols)} {sym} {len(rows)} {time.time() - t0:.0f}s", flush=True)
                break
            except Exception as e:                       # gateway restart or pacing: reconnect and retry
                if "No security definition" in str(e):
                    print(f"{sym} not found: {e!r}", flush=True)
                    break
                print(f"{sym} error {e!r}; reconnecting", flush=True)
                time.sleep(10 * (attempt + 1))
                try:
                    gw.close()
                except Exception:
                    pass
                try:
                    gw = connect()
                except Exception as e2:
                    print("reconnect failed", repr(e2), flush=True)
        time.sleep(PACE)


def article_path(provider, aid):
    return ARTICLES / provider / f"{aid}.json"


def article_ids():
    """(provider, article_id) of every BRFG / DJNL headline in data/news/*.json, oldest first."""
    rows = {}
    for f in NEWS.glob("*.json"):
        for h in json.loads(f.read_text(encoding="utf-8")).get("headlines", []):
            if h["provider"] in TEXT_PROVIDERS:
                rows.setdefault((h["provider"], h["article_id"]), h["time"])
    return sorted(rows, key=lambda k: rows[k])


def backfill_articles():
    todo = [k for k in article_ids() if not article_path(*k).exists()]
    print(f"{len(todo)} articles to fetch", flush=True)
    gw, t0, ok, bad = connect(), time.time(), 0, []
    for i, (p, aid) in enumerate(todo):
        t1 = time.time()
        try:
            a = gw.article(p, aid)
        except Exception as e:                          # noqa: BLE001 - logged and retried on the next run
            bad.append(aid)
            print(f"{p} {aid} error {e!r}"[:300], flush=True)
            if not gw.connected:
                time.sleep(10)
                try:
                    gw = connect()
                except Exception as e2:                 # noqa: BLE001
                    print("reconnect failed", repr(e2), flush=True)
            continue
        out = article_path(p, aid)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"provider": p, "article_id": aid, "fetched_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"), **{k: a[k] for k in ("text", "base64") if k in a}}), encoding="utf-8")
        ok += 1
        if i % 50 == 0:
            print(f"{i + 1}/{len(todo)} ok {ok} failed {len(bad)} {time.time() - t0:.0f}s", flush=True)
        time.sleep(max(0.0, PACE - (time.time() - t1)))
    print(f"done: {ok} fetched, {len(bad)} failed {bad[:10]}, {time.time() - t0:.0f}s", flush=True)
    gw.close()


if __name__ == "__main__":
    if "--articles" in sys.argv:
        backfill_articles()
        sys.exit()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    syms = args or [s.strip() for s in (ROOT / "data" / "universe523.txt").read_text().splitlines() if s.strip()]
    main(syms, skip_done="--resume" in sys.argv)
