"""The rate line in every live sleeve's state (Clayton, 25 Oct 23:05 ET: "As you recommended", then deployed before
week 3's first look; framework_issues_etf_probe.md item 4): the fed funds target range and its last change, the
3-month Treasury bill yield, and the next FOMC decision, as facts. Without it the models priced rates from memory
(bills at 5.0-5.25% against 4.14%; Opus expected cuts while the Fed was raising).

fetch() runs at the 07:30 refresh (FRED via ml_research.fred; the key is never printed or written) and keeps three
numbers in data/macro/rates.json. line() reads that file and the macro calendar and returns "" when either is missing,
so a look never fails on it.
"""
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
FILE = ROOT / "data" / "macro" / "rates.json"
CALENDAR = ROOT / "data" / "macro" / "calendar.json"
ET = ZoneInfo("America/New_York")
STALE_DAYS = 7          # a figure older than this is left out


def _obs(series, start):
    from ml_research import fred
    j = fred.get("series/observations", series_id=series, observation_start=start)
    return [(o["date"], float(o["value"])) for o in j["observations"] if o["value"] not in (".", "")]


def fetch(today=None):
    """FRED's DFEDTARL/DFEDTARU (target range, daily) and DGS3MO (3-month bill, constant maturity, a day behind)."""
    today = today or datetime.now(ET).date()
    start = (today - timedelta(days=400)).isoformat()
    lo, up, bill = _obs("DFEDTARL", start), _obs("DFEDTARU", start), _obs("DGS3MO", start)
    change = next(({"date": up[i][0], "bps": round(100 * (up[i][1] - up[i - 1][1]))}
                   for i in range(len(up) - 1, 0, -1) if up[i][1] != up[i - 1][1]), None)
    out = {"fetched_et": datetime.now(ET).isoformat(timespec="seconds"), "source": "FRED DFEDTARL, DFEDTARU, DGS3MO",
           "target": [lo[-1][1], up[-1][1]], "target_as_of": up[-1][0], "last_change": change,
           "bill3m": bill[-1][1], "bill3m_as_of": bill[-1][0]}
    FILE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def _day(iso):
    d = date.fromisoformat(iso)
    return f"{d:%a} {d.day} {d:%b}" + (f" {d.year}" if d.year != datetime.now(ET).year else "")


def line(now=None):
    now = now or datetime.now(ET)
    try:
        r = json.loads(FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    fresh = lambda iso: (now.date() - date.fromisoformat(iso)).days <= STALE_DAYS
    try:
        fomc = sorted((e for e in json.loads(CALENDAR.read_text(encoding="utf-8"))["events"] if e.get("key") == "fomc"),
                      key=lambda e: e["datetime_et"])
    except (OSError, ValueError, KeyError):
        fomc = []
    parts = []
    if fresh(r["target_as_of"]):
        lo, up = r["target"]
        t = f"fed funds target {lo:.2f}-{up:.2f}%"
        c = r.get("last_change")
        if c:                       # FRED dates a change by when it took effect, the day after the decision
            eff = date.fromisoformat(c["date"])
            dec = [e for e in fomc if 0 <= (eff - date.fromisoformat(e["date"])).days <= 3]
            t += (f" ({'raised' if c['bps'] > 0 else 'cut'} {abs(c['bps'])} bp at the FOMC decision of {_day(dec[-1]['date'])})"
                  if dec else f" (changed {c['bps']:+d} bp, effective {_day(c['date'])})")
        parts.append(t)
    if fresh(r["bill3m_as_of"]):    # DGS3MO: read in the probe as a fund's own yield; the label says what it is
        parts.append(f"3-month Treasury yield {r['bill3m']:.2f}% (constant maturity, {_day(r['bill3m_as_of'])})")
    ahead = [e for e in fomc if datetime.fromisoformat(e["datetime_et"]) > now][:2]
    past = [e for e in fomc if datetime.fromisoformat(e["datetime_et"]) <= now]
    if parts and past and datetime.fromisoformat(past[-1]["datetime_et"]) > datetime.fromisoformat(r["fetched_et"]):
        # a decision since the fetch (FOMC day after 14:00, until the next refresh): the figures above predate it
        parts.append(f"the FOMC decided on {_day(past[-1]['date'])} at {past[-1]['time_et']} ET, after these figures "
                     "(see tape)")
    if ahead:                       # with the market's pricing (Kalshi, from the calendar): in the probe Opus read the
        last = datetime.fromisoformat(past[-1]["datetime_et"]) if past else None   # 3-month yield's premium as
        odds = [_odds(e, now, last) for e in ahead]                                # a hike priced for the next decision
        t = f"next FOMC decision {_day(ahead[0]['date'])}, {ahead[0]['time_et']} ET" + (f" (Kalshi: {odds[0][0]})" if odds[0][0] else "")
        if len(ahead) > 1:
            t += f", then {_day(ahead[1]['date'])}" + (f" (Kalshi: {odds[1][0]})" if odds[1][0] else "")
        asof = next((a for _, a in odds if a), None)
        parts.append(t + (f"; Kalshi prices of {_day(asof)}" if asof else ""))
    return ("Rates: " + "; ".join(parts) + ".") if parts else ""


def _odds(e, now, last_decision=None):
    """('hold 91%, +25 bp 7%', as-of date) from a calendar row's Kalshi decision market; ('', None) if none, stale, or
    priced before the latest decision (an FOMC afternoon, until the nightly calendar rebuild)."""
    x = (e.get("expectation") or {}).get("rate_change_bps") or {}
    p = x.get("probabilities_bps") or {}
    try:
        at = datetime.fromisoformat(x["as_of_utc"])
        asof = at.astimezone(ET).date()
    except (KeyError, ValueError, TypeError):
        return "", None
    if (not p or x.get("quality", "ok") != "ok" or (now.date() - asof).days > STALE_DAYS
            or (last_decision is not None and at < last_decision)):
        return "", None
    lab = lambda k: "hold" if int(k) == 0 else f"{int(k):+d} bp"
    top = [f"{lab(k)} {100 * v:.0f}%" for k, v in sorted(p.items(), key=lambda kv: -kv[1])[:2] if v >= 0.05]
    return ", ".join(top), asof.isoformat()


if __name__ == "__main__":
    print(json.dumps(fetch(), indent=1))
    print(line())
