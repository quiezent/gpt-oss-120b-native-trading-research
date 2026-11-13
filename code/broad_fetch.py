"""The broad universe (Clayton, 9 Nov 2026: "allow the simulation to pull from IBKR any names and the broader the better").

    python broad_fetch.py list            # build data/bars_broad/candidates.json from the NASDAQ Trader symbol directories
    python broad_fetch.py fetch [--client trader-research] [--max N]    # daily ADJUSTED_LAST bars, resumable

Candidates: US-listed common stocks, ADRs and ETFs from the public NASDAQ Trader directories (nasdaqlisted.txt,
otherlisted.txt), without test issues, warrants, rights, units, preferreds, notes or when-issued lines. Bars go to
data/bars_broad/<SYM>.json (not data/bars, the live cache), with the pull date: ADJUSTED_LAST is re-adjusted at every
pull. The TWS Gateway Owner's measurements (tws 1bd7d7c): duration "4 Y" (ADJUSTED_LAST takes no end date; day
durations over 365 are refused), about 1 s a name, back off a minute on a pacing violation, skip 05:55-06:15 ET.
IBKR has no bars for delisted names, so this universe is survivors (plus names still quoted OTC); that bias is measured
separately, never assumed away. Read-only on the account: market data requests only.
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "bars_broad"
ET = ZoneInfo("America/New_York")
DIRS = {"nasdaq": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "other": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"}
BAD_NAME = re.compile(r"\b(warrants?|rights?|units?|preferred|depositary shares? representing|notes? due|debentures?|"
                      r"when[- ]issued|trust preferred|subordinated|% series|series [a-z] |cumulative|perpetual|"
                      r"contingent value|acquisition corp|blank check)\b", re.I)


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "trader-training research (paper account)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def build_list():
    rows = []
    for src, url in DIRS.items():
        text = _get(url)
        lines = [x for x in text.splitlines() if x and not x.startswith("File Creation Time")]
        head = lines[0].split("|")
        for x in lines[1:]:
            f = dict(zip(head, x.split("|")))
            sym = f.get("Symbol") if src == "nasdaq" else f.get("ACT Symbol")
            name = f.get("Security Name", "")
            if not sym or f.get("Test Issue") == "Y":
                continue
            etf = f.get("ETF") == "Y"
            if not etf and BAD_NAME.search(name):
                continue
            if not re.fullmatch(r"[A-Z]{1,5}([.$][A-Z])?", sym) or "$" in sym:
                continue
            exch = "NASDAQ" if src == "nasdaq" else {"A": "AMEX", "N": "NYSE", "P": "ARCA", "Z": "BATS", "V": "IEX"}.get(
                f.get("Exchange", ""), f.get("Exchange", ""))
            rows.append({"symbol": sym.replace(".", " "), "name": name, "etf": etf, "exchange": exch, "source": src,
                         "financial_status": f.get("Financial Status")})
    seen, out = set(), []
    for r in rows:
        if r["symbol"] not in seen:
            seen.add(r["symbol"])
            out.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {"built_et": datetime.now(ET).isoformat(timespec="seconds"), "sources": DIRS, "n": len(out),
           "n_etf": sum(r["etf"] for r in out), "rows": out}
    (OUT / "candidates.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"{len(out)} candidates ({doc['n_etf']} ETFs) -> {OUT / 'candidates.json'}")


def _path(sym):
    return OUT / f"{sym.replace(' ', '_')}.json"


DOWN = ("1100", "2110", "connectivity", "not connected", "socket", "504", "connection")   # IBKR or the link is down


def _down(msg):
    m = msg.lower()
    return any(k in m for k in DOWN)


def _stop_at(until):
    """True once ET time is past HH:MM `until` (same calendar day as now, ET)."""
    if not until:
        return False
    h, m = map(int, until.split(":"))
    now = datetime.now(ET)
    return (now.hour, now.minute) >= (h, m)


def fetch(client, max_n=None, until=None):
    """Resumable: names with a file are skipped. An outage (1100, 2110, a dropped socket) doesn't burn the list: the
    fetch waits a minute and retries the same name, reconnecting, until the link is back or `until` (ET HH:MM) passes."""
    from tws import Gateway
    cands = json.loads((OUT / "candidates.json").read_text(encoding="utf-8"))["rows"]
    stocks = [r for r in cands if not r["etf"]]
    etfs = [r for r in cands if r["etf"]]
    todo = [r for r in stocks + etfs if not _path(r["symbol"]).exists()]      # stocks first; resumable
    if max_n:
        todo = todo[:max_n]
    log = OUT / "fetch_log.jsonl"
    print(f"{len(todo)} to fetch of {len(cands)}", flush=True)
    done, gw = 0, None
    try:
        for r in todo:
            if _stop_at(until):
                print(f"stopped at {until} ET", flush=True)
                break
            now = datetime.now(ET)
            if now.hour == 5 and now.minute >= 55 or now.hour == 6 and now.minute < 15:
                time.sleep(60 * 20)                    # the Gateway's 06:00 ET restart
            s, rec = r["symbol"], {"symbol": r["symbol"], "t": now.isoformat(timespec="seconds")}
            bars, tries = None, 0
            while True:
                try:
                    if gw is None:
                        gw = Gateway(client, timeout=30).__enter__()
                    bars = gw.bars(s, duration="4 Y", bar_size="1 day", what="ADJUSTED_LAST", rth=True, timeout=45)
                    rec.pop("error", None)
                    break
                except Exception as e:                  # noqa: BLE001
                    msg = str(e)
                    if "pacing" in msg.lower() or "420" in msg:
                        time.sleep(60)
                        continue
                    if _down(msg) and not _stop_at(until):
                        tries += 1
                        if tries == 1 or tries % 10 == 0:
                            print(f"{datetime.now(ET):%H:%M:%S} ET link down ({msg[:80]}); waiting", flush=True)
                        try:
                            gw.__exit__(None, None, None) if gw else None
                        except Exception:              # noqa: BLE001
                            pass
                        gw = None
                        time.sleep(60)
                        continue
                    bars, rec["error"] = None, msg[:200]
                    break
            if bars:
                rows = [{k: b.get(k) for k in ("date", "open", "high", "low", "close", "volume")} for b in bars]
                _path(s).write_text(json.dumps({"symbol": s, "pulled_et": now.isoformat(timespec="seconds"),
                                                "what": "ADJUSTED_LAST", "meta": r, "bars": rows}), encoding="utf-8")
                rec.update(n=len(rows), first=str(rows[0]["date"])[:8], last=str(rows[-1]["date"])[:8])
            else:
                rec.setdefault("error", "no bars")
            with open(log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            done += 1
            if done % 100 == 0:
                print(f"{done}/{len(todo)} {now:%H:%M:%S} ET", flush=True)
    finally:
        if gw is not None:
            try:
                gw.__exit__(None, None, None)
            except Exception:                           # noqa: BLE001
                pass
    print(f"done {done}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "fetch"])
    ap.add_argument("--client", default="trader-research")
    ap.add_argument("--max", type=int)
    ap.add_argument("--until", help="ET HH:MM: start no name after it (the P1 stops before the 17:00 UTC+8 cutover)")
    a = ap.parse_args()
    build_list() if a.cmd == "list" else fetch(a.client, a.max, a.until)
