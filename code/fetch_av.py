"""Alpha Vantage earnings data (key: user env var Alpha_Vantage; free tier 25 calls a day, 1 a second; personal research
use). Raw responses stay in data/av/ (gitignored, not redistributed). Never print or log a request URL: the key is in it.

  daily(n_history): one EARNINGS_CALENDAR call (every scheduled US report in the next 3 months: date, consensus EPS,
      before/after market) -> data/av/calendar/<YYYYMMDD>.csv, a point-in-time record that only accumulates forward; then
      EARNINGS (per-quarter report date and time, reported and consensus EPS, surprise) for up to n_history universe stocks
      not yet cached -> data/av/earnings/<SYM>.json. live.py's 08:15 refresh calls daily(22): 23 calls a day.

    python fetch_av.py            # one daily run now
"""

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
AV = ROOT / "data" / "av"
ET = ZoneInfo("America/New_York")


def _call(**params):
    from ml_research import keys
    key = keys.get("Alpha_Vantage")
    if not key:
        raise RuntimeError("no Alpha_Vantage key in the environment")
    q = urllib.parse.urlencode({**params, "apikey": key})
    try:
        with urllib.request.urlopen("https://www.alphavantage.co/query?" + q, timeout=60) as r:
            text = r.read().decode("utf-8")
    except Exception as e:                                   # noqa: BLE001
        raise RuntimeError(f"request failed: {type(e).__name__}") from None   # never the URL
    finally:
        time.sleep(1.2)                                     # free tier: at most 1 request a second
    if text.lstrip().startswith("{") and ('"Information"' in text[:200] or '"Note"' in text[:200] or '"Error Message"' in text[:200]):
        raise RuntimeError("Alpha Vantage refused: " + json.loads(text).get("Information", json.loads(text).get("Note", json.loads(text).get("Error Message", "")))[:120])
    return text


def daily(n_history=22):
    import trader
    out = {"calendar": None, "history": [], "errors": []}
    today = datetime.now(ET).strftime("%Y%m%d")
    cal = AV / "calendar" / f"{today}.csv"
    if not cal.exists():
        try:
            text = _call(function="EARNINGS_CALENDAR", horizon="3month")
            if not text.startswith("symbol"):
                raise RuntimeError("unexpected calendar response")
            cal.parent.mkdir(parents=True, exist_ok=True)
            cal.write_text(text, encoding="utf-8")
            out["calendar"] = text.count("\n")
        except RuntimeError as e:
            out["errors"].append(f"calendar: {e}")
    etf = trader.etfs()
    todo = [s for s in sorted(set(trader.universe()) - set(etf))
            if not (AV / "earnings" / f"{s.replace(' ', '_')}.json").exists()]
    for s in todo[:n_history]:
        try:
            text = _call(function="EARNINGS", symbol=s.replace(" ", "."))   # Alpha Vantage writes share classes BRK.B
            j = json.loads(text)
            if not j.get("quarterlyEarnings"):
                raise RuntimeError("no quarterly earnings")
            p = AV / "earnings" / f"{s.replace(' ', '_')}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
            out["history"].append(s)
        except RuntimeError as e:
            out["errors"].append(f"{s}: {e}")
            if "refused" in str(e):                       # daily limit reached: stop for today
                break
    out["remaining"] = max(0, len(todo) - len(out["history"]))
    return out


if __name__ == "__main__":
    r = daily(int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 22)
    print({k: (v if k != "history" else len(v)) for k, v in r.items()})
