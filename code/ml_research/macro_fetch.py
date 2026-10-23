"""Build data/macro/calendar.json: scheduled US macro releases and FOMC decisions, 2023-10..2026-12, with
release time (ET), first-release values (ALFRED vintages) and the pre-release Kalshi market-implied expectation.

Sources (all public): FRED/ALFRED API (key via ml_research.keys, never printed or written), federalreserve.gov
FOMC calendar and statement pages, Kalshi public read endpoints (no key). Raw responses are cached in
data/macro/cache/ so a rerun only refetches what is missing or can still change: FRED (not cached), the FOMC calendar
page, Kalshi's event lists, and the markets of Kalshi events not yet settled. live.py reruns it each evening.

    .venv\\Scripts\\python.exe ml_research\\macro_fetch.py            # full build
    .venv\\Scripts\\python.exe ml_research\\macro_fetch.py --no-kalshi
"""

import json
import re
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import threading
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml_research import fred  # noqa: E402

OUT = ROOT / "data" / "macro"
CACHE = OUT / "cache"
ET = ZoneInfo("America/New_York")
UA = "Trader Training research contact via github.com/quiezent"
START, END = "2023-10-01", "2027-12-31"
TODAY = datetime.now(ET).date()

# key, name, FRED release id, conventional time ET, agency, tier, headline series that must update on a real release day,
# value specs (field, series, transform, frequency)
RELEASES = [
    ("cpi", "CPI", 10, "08:30", "BLS", 1, "CPIAUCSL",
     [("cpi_mom_pct", "CPIAUCSL", "pch", "M"), ("core_cpi_mom_pct", "CPILFESL", "pch", "M"), ("cpi_yoy_pct", "CPIAUCNS", "pch12", "M"),
      ("core_cpi_yoy_pct", "CPILFENS", "pch12", "M")]),
    ("nfp", "Employment Situation", 50, "08:30", "BLS", 1, "PAYEMS",
     [("nfp_change_k", "PAYEMS", "chg", "M"), ("unemployment_rate", "UNRATE", "lin", "M"), ("ahe_mom_pct", "CES0500000003", "pch", "M")]),
    ("gdp", "GDP", 53, "08:30", "BEA", 1, ("A191RL1Q225SBEA", "GDPC1", "GDP"),
     [("real_gdp_qoq_saar", "A191RL1Q225SBEA", "lin", "Q")]),
    ("pce", "Personal Income and Outlays (PCE)", 54, "08:30", "BEA", 1, "PCEPI",
     [("pce_mom_pct", "PCEPI", "pch", "M"), ("core_pce_mom_pct", "PCEPILFE", "pch", "M"), ("pce_yoy_pct", "PCEPI", "pch12", "M"),
      ("core_pce_yoy_pct", "PCEPILFE", "pch12", "M")]),
    ("ppi", "PPI", 46, "08:30", "BLS", 2, "PPIFIS",
     [("ppi_fd_mom_pct", "PPIFIS", "pch", "M"), ("core_ppi_mom_pct", "PPIFES", "pch", "M")]),
    ("retail", "Retail Sales (advance)", 9, "08:30", "Census", 2, "RSAFS",
     [("retail_mom_pct", "RSAFS", "pch", "M"), ("retail_ex_auto_mom_pct", "RSFSXMV", "pch", "M")]),
    ("jolts", "JOLTS", 192, "10:00", "BLS", 2, "JTSJOL",
     [("job_openings_k", "JTSJOL", "lin", "M")]),
    ("claims", "Initial Jobless Claims", 180, "08:30", "DOL", 2, "ICSA",
     [("initial_claims", "ICSA", "lin", "W")]),
    ("eci", "Employment Cost Index", 11, "08:30", "BLS", 2, "ECIALLCIV",
     [("eci_qoq_pct", "ECIALLCIV", "pch", "Q")]),
    ("adp", "ADP National Employment", 194, "08:15", "ADP", 3, "ADPMNUSNERSA",
     [("adp_change_k", "ADPMNUSNERSA", "chg_k", "M")]),
    ("umich", "Michigan Sentiment (final)", 91, "10:00", "UMich", 3, "UMCSENT",
     [("umich_sentiment", "UMCSENT", "lin", "M")]),
    ("indpro", "Industrial Production", 13, "09:15", "Fed", 3, "INDPRO",
     [("indpro_mom_pct", "INDPRO", "pch", "M")]),
    ("housing_starts", "Housing Starts", 27, "08:30", "Census", 3, "HOUST",
     [("housing_starts_k_saar", "HOUST", "lin", "M")]),
    ("new_home_sales", "New Home Sales", 97, "10:00", "Census", 3, "HSN1F",
     [("new_home_sales_k_saar", "HSN1F", "lin", "M")]),
]

# Kalshi series -> (release key, value field it prices, kind)
KALSHI = [
    ("KXCPI", "cpi", "cpi_mom_pct"), ("KXCPICORE", "cpi", "core_cpi_mom_pct"), ("KXCPIYOY", "cpi", "cpi_yoy_pct"),
    ("KXPAYROLLS", "nfp", "nfp_change_k"), ("KXU3", "nfp", "unemployment_rate"),
    ("KXGDP", "gdp", "real_gdp_qoq_saar"), ("KXPCECORE", "pce", "core_pce_mom_pct"),
    ("KXJOBLESSCLAIMS", "claims", "initial_claims"), ("KXJOLTSOPEN", "jolts", "job_openings_k"),
    ("KXUSRETAIL", "retail", "retail_mom_pct"), ("KXISMPMI", "ism_mfg", "ism_mfg_pmi"),
    ("KXFEDDECISION", "fomc", "rate_change_bps"),
]
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2/"

# Rows checked by hand against the agency's own release (bea.gov, 18 Oct 2026). main() applies them after every build,
# so a rebuild keeps them: the release day and time ET, marked verified, with time_uncertain and date_conflict cleared.
OVERRIDES = {
    "2025-12-05_pce": ("2025-12-05", "10:00", "bea.gov Personal Income and Outlays, September 2025: 'EMBARGOED UNTIL "
                                              "RELEASE AT 10:00 a.m. EST, Friday, December 5, 2025'"),
    "2025-12-23_gdp": ("2025-12-23", "08:30", "bea.gov GDP, 3rd Quarter 2025 (Initial Estimate): 'EMBARGOED UNTIL RELEASE "
                                              "AT 8:30 a.m. EST, Tuesday, December 23, 2025'"),
    "2025-12-23_pce": ("2025-12-23", "08:30", "bea.gov Personal Income and Outlays, Data Update, September 2025 (23 Dec "
                                              "2025): released with the 08:30 GDP initial estimate it accompanies"),
    "2026-01-22_gdp": ("2026-01-22", "08:30", "bea.gov GDP, 3rd Quarter 2025 (Updated Estimate): 'EMBARGOED UNTIL RELEASE "
                                              "AT 8:30 a.m. EST, Thursday, January 22, 2026'"),
    "2026-01-22_pce": ("2026-01-22", "10:00", "bea.gov Personal Income and Outlays, October and November 2025: 'EMBARGOED "
                                              "UNTIL RELEASE AT 10:00 a.m. EST, Thursday, January 22, 2026'"),
    "2026-10-29_gdp": ("2026-10-29", "08:30", "bea.gov release schedule (18 Oct 2026): 'GDP (Advance Estimate), 3rd Quarter "
                                              "2026', October 29, 8:30 AM (Kalshi's KXGDP-26OCT30 says 30 Oct)"),
    "2026-10-29_pce": ("2026-10-29", "08:30", "bea.gov release schedule (18 Oct 2026): 'Personal Income and Outlays, "
                                              "September 2026', October 29, 8:30 AM"),
}


def apply_overrides(events):
    """Set the hand-checked rows (OVERRIDES) in place; returns the ids applied."""
    done = []
    for e in events:
        o = OVERRIDES.get(e.get("id"))
        if o is None:
            continue
        day, hhmm, src = o
        e.update(date=day, time_et=hhmm, datetime_et=et_iso(date.fromisoformat(day), hhmm), id=f"{day}_{e['key']}")
        e.pop("date_conflict", None)
        tc = e.setdefault("time_check", {})
        for k in ("time_uncertain", "conflict"):
            tc.pop(k, None)
        tc.update(verified=True, via="primary source checked by hand", primary_source=src)
        done.append(e["id"])
    events.sort(key=lambda e: (e["datetime_et"], e["key"]))
    return done

# ---------------------------------------------------------------- http helpers


_tls = threading.local()
_rate_lock = threading.Lock()
_next_slot = [0.0]


def _session():
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
        _tls.s.headers["User-Agent"] = UA
    return _tls.s


def _throttle(pace):
    """Global limiter across threads: at most one request start per `pace` seconds."""
    with _rate_lock:
        now = time.time()
        slot = max(now, _next_slot[0])
        _next_slot[0] = slot + pace
    if slot > now:
        time.sleep(slot - now)


def _cached_get(url, path, pace=0.25, params=None, ok404=False, refresh=False):
    path = CACHE / path
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else path.read_text(encoding="utf-8")
    last = "no attempt"
    for attempt in range(5):
        _throttle(pace)
        try:
            r = _session().get(url, params=params, timeout=30)
        except requests.RequestException as e:
            time.sleep(3 * (attempt + 1))
            last = str(e)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            last = f"HTTP {r.status_code}"
            continue
        if r.status_code == 404 and ok404:
            return None
        if r.status_code != 200:
            raise RuntimeError(f"GET {url} HTTP {r.status_code}: {r.text[:200]}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text, encoding="utf-8")
        return r.json() if path.suffix == ".json" else r.text
    raise RuntimeError(f"GET {url}: {last}")


def et_iso(d, hhmm):
    h, m = map(int, hhmm.split(":"))
    return datetime(d.year, d.month, d.day, h, m, tzinfo=ET).isoformat()


# ---------------------------------------------------------------- FOMC

MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September",
                                      "October", "November", "December"], 1)}
MON3 = {k[:3]: v for k, v in MONTHS.items()}


def fomc_events():
    page = _cached_get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", "fed/fomccalendars.htm",
                       refresh=True)
    rows = []
    for ym in re.finditer(r'<h4><a id="\d+">(\d{4}) FOMC Meetings</a></h4>', page):
        year = int(ym.group(1))
        if year < 2023 or year > int(END[:4]):
            continue
        nxt = page.find("FOMC Meetings</a></h4>", ym.end())
        seg = page[ym.end(): nxt if nxt > 0 else len(page)]
        for blk in re.split(r'<div class="(?:fomc-meeting--shaded )?row fomc-meeting"', seg)[1:]:
            mon = re.search(r'fomc-meeting__month[^>]*><strong>([^<]+)</strong>', blk)
            dts = re.search(r'fomc-meeting__date[^>]*>([^<]+)</div>', blk)
            if not mon or not dts:
                continue
            mon, dts = mon.group(1).strip(), dts.group(1).strip()
            if "notation" in dts.lower() or "unscheduled" in dts.lower() and "-" not in dts:
                continue
            sep = "*" in dts
            stmt = re.search(r'/newsevents/pressreleases/monetary(\d{8})a\.htm', blk)
            if stmt:
                d = datetime.strptime(stmt.group(1), "%Y%m%d").date()
            else:
                last_mon = mon.split("/")[-1].strip()
                mnum = MONTHS.get(last_mon) or MON3.get(last_mon[:3])
                day = int(re.findall(r"\d+", dts)[-1])
                d = date(year, mnum, day)
            minutes = re.search(r"\(Released ([A-Z][a-z]+ \d{1,2}, \d{4})\)", blk)
            rows.append(dict(date=d, sep=sep, statement_url=("https://www.federalreserve.gov/newsevents/pressreleases/monetary"
                                                            f"{stmt.group(1)}a.htm") if stmt else None,
                             minutes=datetime.strptime(minutes.group(1), "%B %d, %Y").date() if minutes else None,
                             label=f"{mon} {dts}"))
    events = []
    for r in sorted(rows, key=lambda r: r["date"]):
        check = {"method": "conventional 14:00 ET"}
        if r["statement_url"]:
            html = _cached_get(r["statement_url"], f"fed/statement_{r['date']:%Y%m%d}.htm", pace=1.0)
            m = re.search(r"For (?:release|immediate release)[^<]{0,40}?(\d{1,2}:\d{2}\s*[ap]\.?m\.?)\s*(E[SD]T)?", html, re.I)
            check = {"method": "federalreserve.gov statement page", "stated": (m.group(0).strip() if m else None)}
            if m:
                t = m.group(1).replace(".", "").replace(" ", "").lower()
                hh, mm = map(int, t[:-2].split(":"))
                hh = hh + 12 if t.endswith("pm") and hh != 12 else hh
                check["consistent"] = f"{hh:02d}:{mm:02d}" == "14:00"
        released = r["date"] < TODAY or (r["date"] == TODAY and datetime.now(ET).hour >= 14)
        events.append(dict(key="fomc", name="FOMC statement", tier=1, date=r["date"].isoformat(), time_et="14:00",
                           datetime_et=et_iso(r["date"], "14:00"), status="released" if released else "scheduled",
                           reference_period=r["label"], first_release={"sep_meeting": r["sep"], "press_conference_et": "14:30"},
                           expectation=None, time_check=check,
                           source={"dates": "federalreserve.gov/monetarypolicy/fomccalendars.htm", "url": r["statement_url"]}))
        if r["minutes"]:
            events.append(dict(key="fomc_minutes", name="FOMC minutes", tier=2, date=r["minutes"].isoformat(), time_et="14:00",
                               datetime_et=et_iso(r["minutes"], "14:00"),
                               status="released" if r["minutes"] < TODAY else "scheduled", reference_period=r["label"],
                               first_release=None, expectation=None, time_check={"method": "conventional 14:00 ET"},
                               source={"dates": "federalreserve.gov/monetarypolicy/fomccalendars.htm"}))
    return events


# ---------------------------------------------------------------- FRED / ALFRED

_vint_cache = {}


def vintages(series):
    """All vintages since 2023-09-01 of observations since 2022-01-01: list of (obs_date, value, rt_start, rt_end)."""
    if series in _vint_cache:
        return _vint_cache[series]
    j = fred.get("series/observations", series_id=series, realtime_start="2023-09-01", realtime_end="9999-12-31",
                 observation_start="2022-01-01", limit=100000)
    rows = [(o["date"], None if o["value"] == "." else float(o["value"]), o["realtime_start"], o["realtime_end"])
            for o in j["observations"]]
    _vint_cache[series] = rows
    return rows


def asof(series, d):
    ds = d.isoformat()
    return {od: v for od, v, rs, re_ in vintages(series) if rs <= ds <= re_ and v is not None}


def shift(obs, freq, n):
    d = date.fromisoformat(obs)
    if freq == "W":
        return (d - timedelta(days=7 * n)).isoformat()
    months = n * (3 if freq == "Q" else 1)
    y, m = divmod(d.month - 1 - months, 12)
    return date(d.year + y, m + 1, 1).isoformat()


def transform(v, ref, how, freq):
    cur = v.get(ref)
    if cur is None:
        return None
    if how == "lin":
        return round(cur, 3)
    prev = v.get(shift(ref, freq, 12 if how == "pch12" else 1))
    if prev is None:
        return None
    if how == "chg":
        return round(cur - prev, 3)
    if how == "chg_k":
        return round((cur - prev) / 1000.0, 1)
    return round((cur / prev - 1) * 100, 3)


def fred_events():
    events, report = [], {}
    for key, name, rid, t, agency, tier, check, specs in RELEASES:
        rd = fred.get("release/dates", release_id=rid, realtime_start=START, realtime_end="9999-12-31",
                      include_release_dates_with_no_data="true", sort_order="asc", limit=10000)["release_dates"]
        dates = sorted({x["date"] for x in rd if START <= x["date"] <= END})
        checks = check if isinstance(check, tuple) else (check,)
        check = checks[0]
        vd = set()
        for c in checks:   # a release day must change at least one of these (an unchanged GDP growth rate makes no vintage)
            vd |= set(fred.get("series/vintagedates", series_id=c, realtime_start="2023-09-01")["vintage_dates"])
        rep = dict(fred_release_id=rid, fred_dates=len(dates), dropped_no_headline_update=[], added_from_vintages=[])
        past = [d for d in dates if date.fromisoformat(d) < TODAY]
        fut = [d for d in dates if date.fromisoformat(d) >= TODAY]
        keep = []
        for d in past:
            (keep if d in vd else rep["dropped_no_headline_update"]).append(d)
        for d in sorted(vd):
            if START <= d < TODAY.isoformat() and d not in past:
                keep.append(d)
                rep["added_from_vintages"].append(d)
        seen_ref = defaultdict(int)
        for d in sorted(keep):
            dd = date.fromisoformat(d)
            v_now, v_before = asof(check, dd), asof(check, dd - timedelta(days=1))
            ref = max(v_now) if v_now else None
            new_period = ref is not None and ref not in v_before
            vals = {}
            for field, ser, how, freq in specs:
                vv = asof(ser, dd)
                r = max(vv) if vv else None
                if r != ref and ser != check:
                    vals[field] = None if r is None else transform(vv, r, how, freq)
                    if r is not None and r != ref:
                        vals[field + "_ref"] = r
                else:
                    vals[field] = transform(vv, ref, how, freq) if ref else None
            seen_ref[ref] += 1
            kind = "new_period" if new_period else "revision_only"
            if key == "gdp":
                vals["estimate_n"] = seen_ref[ref]
                vals["prior_vintage_value"] = v_before.get(ref)
                kind = ["advance", "second", "third"][min(seen_ref[ref], 3) - 1] if seen_ref[ref] <= 3 else "later_revision"
            events.append(dict(key=key, name=name, tier=tier, date=d, time_et=t, datetime_et=et_iso(dd, t), status="released",
                               reference_period=ref, kind=kind, first_release=vals, expectation=None,
                               time_check={"method": f"conventional {t} ET ({agency})"},
                               source={"dates": f"FRED release_id {rid} release/dates, checked against {'/'.join(checks)} vintage dates",
                                       "values": "ALFRED vintage as of release date"}))
        for d in fut:
            dd = date.fromisoformat(d)
            events.append(dict(key=key, name=name, tier=tier, date=d, time_et=t, datetime_et=et_iso(dd, t), status="scheduled",
                               reference_period=None, first_release=None, expectation=None,
                               time_check={"method": f"conventional {t} ET ({agency})"},
                               source={"dates": f"FRED release_id {rid} release/dates (scheduled)"}))
        lu = fred.get("series", series_id=check)["seriess"][0]["last_updated"]      # e.g. "2026-10-14 08:08:29-05"
        lu_et = datetime.strptime(lu[:19], "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone(timedelta(hours=int(lu[19:])))).astimezone(ET)
        for e in events:
            if e["key"] == key and e["date"] == lu_et.date().isoformat():
                lag = (lu_et - datetime.fromisoformat(e["datetime_et"])).total_seconds() / 60
                e["time_check"]["fred_posted_et"] = lu_et.strftime("%H:%M")
                e["time_check"]["fred_posted_consistent"] = 0 <= lag <= 120
        rep.update(released=len(keep), scheduled=len(fut), latest_fred_post_et=lu_et.isoformat())
        report[key] = rep
        print(f"  {key}: {len(keep)} released, {len(fut)} scheduled, dropped {len(rep['dropped_no_headline_update'])},"
              f" added {len(rep['added_from_vintages'])}", flush=True)
    return events, report


def ism_events():
    """ISM is not on FRED (licence). Manufacturing PMI: 1st business day, Services: 3rd business day, 10:00 ET (rule)."""
    import pandas as pd
    from pandas.tseries.holiday import USFederalHolidayCalendar
    hol = set(USFederalHolidayCalendar().holidays("2023-01-01", "2027-01-31").date)
    out = []
    m = date(2023, 10, 1)
    while m <= date(2026, 12, 1):
        bdays = [d.date() for d in pd.date_range(m, m + timedelta(days=10)) if d.weekday() < 5 and d.date() not in hol]
        for key, name, k in (("ism_mfg", "ISM Manufacturing PMI", 0), ("ism_services", "ISM Services PMI", 2)):
            d = bdays[k]
            out.append(dict(key=key, name=name, tier=2, date=d.isoformat(), time_et="10:00", datetime_et=et_iso(d, "10:00"),
                            status="released" if d < TODAY else "scheduled", reference_period=None, first_release=None,
                            expectation=None, time_check={"method": "conventional 10:00 ET (ISM)"},
                            source={"dates": "rule: ISM Manufacturing 1st, Services 3rd business day (federal holidays skipped);"
                                             " not on FRED"}))
        m = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    return out


# ---------------------------------------------------------------- Kalshi


def kget(path, cache, params=None, refresh=False):
    return _cached_get(KALSHI_BASE + path, "kalshi/" + cache, pace=0.2, params=params, ok404=True, refresh=refresh)


def _settled(cache):
    """A cached Kalshi markets file whose markets all have a result (nothing left to change)."""
    p = CACHE / "kalshi" / cache
    try:
        ms = json.loads(p.read_text(encoding="utf-8")).get("markets") or []
    except (OSError, ValueError, AttributeError):
        return False
    return bool(ms) and all(m.get("result") in ("yes", "no") for m in ms)


def _num(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def candle_prob(c):
    """Mid of yes bid/ask close if the spread is sane, else last trade close. Handles both payload versions."""
    def f(block, k):
        b = c.get(block) or {}
        return _num(b.get(k + "_dollars", b.get(k)))
    bid, ask = f("yes_bid", "close"), f("yes_ask", "close")
    if bid is not None and ask is not None and 0 <= bid <= ask and ask - bid <= 0.15:
        return (bid + ask) / 2, "mid"
    p = f("price", "close")
    if p is not None:
        return p, "last"
    if bid is not None and ask is not None and ask >= bid:
        return (bid + ask) / 2, "wide_mid"
    return None, None


def market_prob(series, m, cutoff_ts):
    """Pre-release yes probability of one market from hourly candles ending at or before cutoff_ts."""
    t = m["ticker"]
    start = cutoff_ts - 3 * 86400
    path = f"historical/markets/{t}/candlesticks" if m.get("_hist") else f"series/{series}/markets/{t}/candlesticks"
    j = kget(path, f"candles/{t}_{cutoff_ts}.json", dict(start_ts=start, end_ts=cutoff_ts, period_interval=60))
    cs = [c for c in ((j or {}).get("candlesticks") or []) if c.get("end_period_ts", 0) <= cutoff_ts]
    for c in reversed(cs):
        p_, how = candle_prob(c)
        if p_ is not None:
            return p_, how, c["end_period_ts"]
    return None, None, None


def kalshi_ref(event_ticker, key):
    """Reference period of a Kalshi event as our reference_period string, e.g. KXCPI-25SEP -> 2025-09-01;
    KXGDP-25OCT30 (advance release date) -> 2025-07-01 (the quarter before)."""
    mm = re.search(r"-(\d{2})([A-Z]{3})(\d{2})?$", event_ticker)
    if not mm or mm.group(2).title() not in MON3:
        return None
    y, mo = 2000 + int(mm.group(1)), MON3[mm.group(2).title()]
    if key == "gdp":
        q0 = (mo - 1) // 3 * 3 + 1 - 3
        y, q0 = (y - 1, q0 + 12) if q0 < 1 else (y, q0)
        return date(y, q0, 1).isoformat()
    if mm.group(3):
        return None
    return date(y, mo, 1).isoformat()


def implied_from_ladder(markets):
    """markets: list of (strike, P(X > strike)). Returns mean, median of the discrete implied distribution."""
    pts = sorted(markets)
    ks = [k for k, _ in pts]
    ps = [p for _, p in pts]
    for i in range(1, len(ps)):            # survival must be non-increasing in the strike
        ps[i] = min(ps[i], ps[i - 1])
    step = min((b - a for a, b in zip(ks, ks[1:])), default=0.1)
    mass = [(ks[0], 1 - ps[0])] + [(ks[i + 1], ps[i] - ps[i + 1]) for i in range(len(ks) - 1)] + [(ks[-1] + step, ps[-1])]
    tot = sum(w for _, w in mass) or 1.0
    mean = sum(k * w for k, w in mass) / tot
    med = next((k for k, p in zip(ks, ps) if p < 0.5), ks[-1] + step)
    if ps[0] < 0.5:
        med = ks[0]
    return round(mean, 4), med


FED_OUT = [(r"(?<![0-9])0\s*bps|maintain|hold|no change", 0), (r"cut\s*>\s*25|cut\s*50|cut\s*>25", -50), (r"cut\s*25", -25),
           (r"hike\s*>\s*25|hike\s*50", 50), (r"hike\s*25", 25)]


def kalshi_expectations(events):
    by_key = defaultdict(list)
    for e in events:
        by_key[e["key"]].append(e)
    log = {}
    now_ts = int(time.time())
    for series, key, field in KALSHI:
        evs, cur, pg = [], None, 0
        while True:
            prm = dict(series_ticker=series, limit=200)
            if cur:
                prm["cursor"] = cur
            j = kget("events", f"events_{series}_{pg}.json", prm, refresh=True) or {}
            evs += j.get("events", [])
            cur, pg = j.get("cursor"), pg + 1
            if not cur or not j.get("events") or pg > 10:
                break
        n_ev = n_hit = 0
        for ev in evs:
            et = ev["event_ticker"]
            ms = []
            fresh = not any(_settled(f"markets/{et}_{i}.json") for i in (0, 1))
            for i, p in enumerate(("historical/markets", "markets")):
                got = (kget(p, f"markets/{et}_{i}.json", dict(event_ticker=et, limit=200), refresh=fresh) or {}).get("markets", [])
                ms += [dict(mm, _hist=(i == 0)) for mm in got]
            if not ms:
                continue
            close = min(datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")) for m in ms)
            close_et = close.astimezone(ET)
            if not (START <= close_et.date().isoformat() <= END):
                continue
            n_ev += 1
            # match 1: same release type on the date the market closed (its scheduled release day)
            cands = [e for e in by_key.get(key, []) if e["date"] == close_et.date().isoformat()]
            delayed = False
            if not cands:   # match 2: same reference period (release moved, e.g. the Oct-Nov 2025 shutdown)
                ref = kalshi_ref(et, key)
                cands = [e for e in by_key.get(key, []) if ref and e.get("reference_period") == ref
                         and e.get("kind") in ("new_period", "advance")]
                delayed = bool(cands)
            if not cands:
                log.setdefault(series + "_unmatched", []).append(f"{et} close {close_et:%Y-%m-%d %H:%M}")
                for e in by_key.get(key, []):
                    if e["status"] == "scheduled" and 0 < abs((date.fromisoformat(e["date"]) - close_et.date()).days) <= 3:
                        e["date_conflict"] = dict(kalshi_event=et, kalshi_scheduled_date=close_et.date().isoformat(),
                                                  note="Kalshi's scheduled release day differs from FRED's")
                continue
            target = cands[0]
            rel_ts = int(datetime.fromisoformat(target["datetime_et"]).timestamp())
            cutoff = min(int(close.timestamp()), rel_ts - 60, now_ts)
            probs, how_used, asof_ts = [], set(), []
            with ThreadPoolExecutor(4) as pool:
                got = list(pool.map(lambda m: market_prob(series, m, cutoff), ms))
            for m, (p, how, ts) in zip(ms, got):
                if p is None:
                    continue
                how_used.add(how)
                asof_ts.append(ts)
                probs.append((m, p))
            if not probs:
                log.setdefault(series + "_noprice", []).append(et)
                continue
            exp = dict(source="kalshi", series=series, event_ticker=et, field=field,
                       market_close_et=close_et.strftime("%Y-%m-%d %H:%M"), delayed_release=delayed,
                       as_of_utc=datetime.fromtimestamp(max(asof_ts), timezone.utc).isoformat(), price_basis=sorted(how_used))
            if series == "KXFEDDECISION":
                dist = {}
                for m, p in probs:
                    lab = (m.get("yes_sub_title") or m.get("subtitle") or m["ticker"]).lower()
                    for pat, bps in FED_OUT:
                        if re.search(pat, lab):
                            dist[bps] = dist.get(bps, 0) + p
                            break
                tot = sum(dist.values())
                if tot <= 0:
                    continue
                dist = {k: round(v / tot, 4) for k, v in sorted(dist.items())}
                exp.update(probabilities_bps=dist, implied_mean=round(sum(k * v for k, v in dist.items()), 2),
                           raw_sum=round(tot, 3), quality="ok" if 0.85 <= tot <= 1.2 else "poor (bucket prices do not sum to 1)")
                res = [m for m in ms if m.get("result") == "yes"]
                if res:
                    exp["settled_outcome"] = res[0].get("yes_sub_title")
            else:
                ladder = []
                for m, p in probs:
                    st, fl, cp = m.get("strike_type"), _num(m.get("floor_strike")), _num(m.get("cap_strike"))
                    if st in ("greater", "greater_or_equal") and fl is not None:
                        ladder.append((fl, p))
                    elif st in ("less", "less_or_equal") and cp is not None:
                        ladder.append((cp, 1 - p))
                if len(ladder) < 2:
                    log.setdefault(series + "_noladder", []).append(et)
                    continue
                if field in ("nfp_change_k", "job_openings_k"):
                    ladder = [(k / 1000.0, p) for k, p in ladder]
                mean, med = implied_from_ladder(ladder)
                exp.update(implied_mean=mean, implied_median=med, n_strikes=len(ladder),
                           p_above={str(k): round(p, 3) for k, p in sorted(ladder)})
                ev_val = next((m.get("expiration_value") for m in ms if m.get("expiration_value")), None)
                if ev_val:
                    exp["kalshi_settlement_value"] = ev_val
            target.setdefault("expectations", []).append(exp)
            # release-time evidence: the contract rules state when the market closes on the scheduled release day
            stated = None
            for m in ms:
                mt = re.search(r"clos\w*\s+(?:at|on)\s+(\d{1,2}):(\d{2})\s*([AP])\.?M", m.get("rules_secondary") or "", re.I)
                if mt:
                    hh = int(mt.group(1)) % 12 + (12 if mt.group(3).upper() == "P" else 0)
                    stated = hh * 60 + int(mt.group(2))
                    break
            basis = "contract rules"
            if stated is None and close_et.hour * 60 + close_et.minute < int(target["time_et"][:2]) * 60 + int(target["time_et"][3:]):
                stated, basis = close_et.hour * 60 + close_et.minute, "market close_time"
            if not delayed and stated is not None:
                th, tm = divmod(-(-stated // 15) * 15, 60)      # the release is at the next quarter hour
                target.setdefault("time_check", {}).setdefault("kalshi", {})[series] = dict(
                    close_et=f"{stated // 60:02d}:{stated % 60:02d}", basis=basis, implied_release_et=f"{th:02d}:{tm:02d}",
                    consistent=f"{th:02d}:{tm:02d}" == target["time_et"])
            if series == "KXISMPMI":
                exp["kalshi_close_date"] = close_et.date().isoformat()
                if exp.get("kalshi_settlement_value") and target["status"] == "released":
                    target["first_release"] = {"ism_mfg_pmi": _num(str(exp["kalshi_settlement_value"]).strip("%")),
                                               "source": "Kalshi settlement value (ISM is not on FRED)"}
            n_hit += 1
        log[series] = dict(events_in_window=n_ev, matched_with_price=n_hit)
        print(f"  kalshi {series}: {n_hit}/{n_ev}", flush=True)
    for e in events:
        if e.get("expectations"):
            e["expectation"] = {x["field"]: x for x in e.pop("expectations")}
    return log


# ---------------------------------------------------------------- empirical time check from own 30-minute bars


def empirical_time_check(events):
    """Median |30-min return| (TLT and SPY) in the bar that should contain the release, relative to the same slot on days
    with no tier-1/2 event, and the same for the overnight gap (for pre-open releases)."""
    import numpy as np
    res = {}
    for sym in ("TLT", "SPY"):
        p = ROOT / "data" / "bars30" / f"{sym}.json"
        if not p.exists():
            continue
        bars = json.loads(p.read_text())["bars"]
        by_day = defaultdict(dict)
        for b in bars:
            d, t = b["date"][:8], b["date"][9:14]
            by_day[d][t] = b
        days = sorted(by_day)
        slot, gap = defaultdict(dict), {}
        for i, d in enumerate(days):
            for t, b in by_day[d].items():
                slot[t][d] = abs(np.log(b["close"] / b["open"]))
            if i and "09:30" in by_day[d] and by_day[days[i - 1]]:
                prev_close = by_day[days[i - 1]][max(by_day[days[i - 1]])]["close"]
                gap[d] = abs(np.log(by_day[d]["09:30"]["open"] / prev_close))
        busy = defaultdict(set)
        for e in events:
            if e["tier"] <= 2 and e["status"] == "released":
                busy[e["date"].replace("-", "")].add(e["time_et"])
        quiet = [d for d in days if not busy.get(d)]
        base = {t: float(np.median([slot[t][d] for d in quiet if d in slot[t]])) for t in ("10:00", "14:00", "09:30")}
        base_gap = float(np.median([gap[d] for d in quiet if d in gap]))
        out = {}
        for key in sorted({e["key"] for e in events}):
            ds = [e["date"].replace("-", "") for e in events if e["key"] == key and e["status"] == "released"
                  and e["date"].replace("-", "") in by_day]
            if len(ds) < 5:
                continue
            r = dict(n_days=len(ds))
            r["gap_ratio"] = round(float(np.median([gap[d] for d in ds if d in gap])) / base_gap, 2)
            for t in ("09:30", "10:00", "14:00"):
                r[f"bar_{t}_ratio"] = round(float(np.median([slot[t][d] for d in ds if d in slot[t]])) / base[t], 2)
            out[key] = r
        res[sym] = dict(first_day=days[0], last_day=days[-1], quiet_days=len(quiet), by_release=out)
        if sym == "TLT":    # per-event evidence for tier-1 releases the Kalshi rules do not cover
            for e in events:
                d = e["date"].replace("-", "")
                if e["tier"] == 1 and e["status"] == "released" and d in by_day and not e["time_check"].get("kalshi"):
                    e["time_check"]["tlt_bars30"] = dict(
                        gap_ratio=round(gap[d] / base_gap, 2) if d in gap else None,
                        **{f"bar_{t}_ratio": round(slot[t][d] / base[t], 2) for t in ("09:30", "10:00", "14:00") if d in slot[t]})
    return res


# ---------------------------------------------------------------- main


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("FOMC", flush=True)
    events = fomc_events()
    print("FRED/ALFRED", flush=True)
    fe, report = fred_events()
    events += fe + ism_events()
    events.sort(key=lambda e: (e["datetime_et"], e["key"]))
    klog = {}
    if "--no-kalshi" not in sys.argv:
        print("Kalshi", flush=True)
        klog = kalshi_expectations(events)
    print("empirical time check", flush=True)
    emp = empirical_time_check(events)
    for e in events:
        tc = e["time_check"]
        ks = tc.get("kalshi") or {}
        ev_ok = [tc.get("consistent")] if "consistent" in tc else []
        ev_ok += [x["consistent"] for x in ks.values()]
        ev_ok += [tc["fred_posted_consistent"]] if "fred_posted_consistent" in tc else []
        tc["verified"] = bool(ev_ok) and all(ev_ok)
        if ev_ok and not all(ev_ok):
            tc["conflict"] = True
    # BEA publishes GDP and Personal Income and Outlays together when they share a day: one verified time covers both
    bea = defaultdict(list)
    for e in events:
        if e["key"] in ("gdp", "pce"):
            bea[e["date"]].append(e)
    for same in bea.values():
        if len(same) == 2 and any(x["time_check"]["verified"] for x in same):
            for x in same:
                if not x["time_check"]["verified"]:
                    x["time_check"].update(verified=True, via="same-day BEA release verified")
    for e in events:   # rescheduled after the Oct-Nov 2025 shutdown and not verified: say so
        if e["status"] == "released" and "2025-10-15" <= e["date"] <= "2026-02-28" and e["tier"] == 1                 and not e["time_check"]["verified"]:
            e["time_check"]["time_uncertain"] = ("release rescheduled after the 2025 shutdown; no allowed source states its"
                                                 " time, so the conventional time is assumed (see tlt_bars30)")
    for i, e in enumerate(events):
        e["id"] = f"{e['date']}_{e['key']}"
    print(f"overrides applied: {apply_overrides(events)}", flush=True)
    meta = dict(built_et=datetime.now(ET).isoformat(timespec="seconds"), window=[START, END], today_et=TODAY.isoformat(),
                n_events=len(events), fred_report=report, kalshi_log=klog, empirical_time_check=emp,
                notes="first_release values are computed from the ALFRED vintage in force on the release date (index levels ->"
                      " % change to 3 dp); the agency's published 1-dp figure can differ by rounding. Expectations are Kalshi"
                      " pre-release prices (last hourly candle before the market's close / the release).")
    (OUT / "calendar.json").write_text(json.dumps(dict(meta=meta, events=events), indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'calendar.json'}: {len(events)} events", flush=True)


if __name__ == "__main__":
    main()
