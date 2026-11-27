"""Each sleeve's book carried over the close and what the next open did to it: one record per sleeve and night.

    python carry_record.py --all                 # rebuild every night the ledgers cover
    python carry_record.py --date 2026-11-13     # the night ending at that day's open
    python carry_record.py --all --out DIR       # write somewhere else (a check; never runs/carry)

Writes runs/carry/positions.csv (one row per sleeve, night and name) and runs/carry/nights.csv (one row per sleeve and
night). Re-running a night replaces that night's rows. No model reads these files; they are a record. Only files are
read: the ledgers (runs/archive/<week>/ and runs/), the look records (runs/live/<date>/), data/bars, data/bars_broad and
data/bars30. Nothing is fetched.

The definitions are the Trading Model Researcher's (Trader-models runs/ml/research/week6_prep/carry/live, 15 Nov 2026),
written again here:
- A run is one sleeve from one start (sleeves.json "start"); base-ml's run from 25 Oct is one run across weeks. Its fills
  are the sleeve's fills, one per exec_id (the latest record wins, as paper.fills), on its days. A restarted sleeve's
  Friday book is in no run: the TWS Gateway Owner closes it before Monday's open, outside every sleeve's value, so the
  night into Monday is recorded only for a run that continues.
- The book at the close of P: that run's fills on P and before, a split with P < ex_date <= D multiplying the quantity
  carried into D and dividing its close (as paper.py applies splits).
- Prices are unadjusted. data/bars holds ADJUSTED_LAST bars; a day's factor k = (09:30 30-minute TRADES bar open) / (daily
  open), the two being the same print, un-adjusts that day's open and close. A day with no 30-minute bar takes k from the
  nearest day that has one; a name with no 30-minute bars takes k from the orders' arrival quotes (arrival.close is the
  previous session's raw close), else k = 1. Name-days with no bar at all take the close from the framework's own marks
  (next session's 09:00 look states, weekend looks, arrival closes, else the day's latest look) and the open from that
  day's opening-auction fills (tif OPG), else its first fill 09:30:00-09:34:59.
- Gap leg of a carried position = q x (open of D - close of P). The open is the 09:30 print: open_src "bars30" (the
  name's 09:30 30-minute bar) is exact; "daily_k", "bars30_late", "fill" and "last_fill" are approximate (approx_open
  True); a name with no open at all has gap 0 and noopen True, and its whole close-to-close goes to the session leg.
- Dividends are their own column (div_usd, from every dividends.jsonl), never in the gap.
- Session leg (pnl_sess) = q x (close of D - open of D), plus, for shares exited on D, (exit price - close of D) x shares.

Exposure measures, both the repo's own:
- beta_net: the look state's "beta-weighted" figure (trader.py Data.features / outside_features): 60 daily log returns
  of the adjusted closes through P against SPY, at least 40; universe stocks shrunk 2/3 raw + 1/3, ETFs (lists/etfs.txt)
  and names outside the universe raw; 1.0 where there is none. It is what the sleeve's model was shown at D's 09:00 look,
  here on the close book at the close of P.
- chip_tilt: the Kelly audit's (runs/analysis/kelly_audit_20261107/queries/b2_books.py betas): each name's loading on
  SMH's residual after SPY, from 120 daily simple returns before P (at least 60, else 0), weighted by value at the close
  of P over the sleeve's value.

Labels: fam and ml_tool of a night are those of the close of P, from P's daily record (runs/live/daily.jsonl and the
archived ones: model, ml_tool as live.py wrote them, e.g. "ML2.1"), else from the sleeves.json of the first ledger set
that holds P; a held sleeve (model "hold") is labelled by its model_held. bars_ok (positions.csv) is True when the name
had daily bars for both P and D when the row was written.

The night into D can be recorded once D's daily bars are in data/bars (live.py's 07:30 refresh of the next trading
day) and D's 30-minute bars are in data/bars30 (live.py's 16:15 record of D, for universe names). update() is the call
for the loop: it records nothing while a ledger is missing or has an unreadable line, and re-records a night of the
last REDO_DAYS trading days when a daily bar missing at the time has since come.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import traceback
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
WEEK1_MONDAY = date(2026, 10, 12)           # week 1 = 13-16 Oct 2026
EXACT_OPEN = {"bars30"}
HOLD_MODEL = "hold"                         # live.HOLD_MODEL: a held sleeve's sleeves.json model; its own is model_held
REDO_DAYS = 5                               # update() re-records a night whose daily bars came late, for this many trading days

POS_COLS = ["run", "fam", "sleeve", "P", "D", "night", "wk", "sym", "q", "side", "w", "noopen", "weekend", "g", "oc", "cc",
            "sg", "soc", "scc", "beta", "pnl_gap", "pnl_sess", "pnl_cc", "exited", "approx_open", "open_src", "close_src",
            "beta_chip", "O", "CP", "C", "bars_ok"]
NIGHT_COLS = ["P", "D", "night", "sleeve", "run", "wk", "fam", "ml_tool", "weekend", "value_P", "positions", "gross_usd",
              "net_usd", "gross_x", "net_x", "beta_net", "chip_tilt", "top5", "gap_usd", "gap_pct", "div_usd", "n_approx",
              "approx_usd", "n_noopen", "last_look_ok", "first_look_ok", "last_look", "first_look"]


# ---------------------------------------------------------------------------------------------------- reading files
def jl(p, bad=None):
    """Rows of a .jsonl file; a line that does not parse (e.g. half-written) is skipped and counted in bad."""
    out = []
    p = Path(p)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            if bad is not None:
                bad[p.name] = bad.get(p.name, 0) + 1
    return out


def jload(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def fname(sym):
    return sym.replace(" ", "_") + ".json"


def ymd(s):
    return str(s).replace("-", "")[:8]


def ddate(s):
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def week_of(d):
    """W1 for the week of 12 Oct 2026; a Saturday or Sunday counts with the Monday after it."""
    x = ddate(d)
    if x.weekday() >= 5:
        x += timedelta(days=7 - x.weekday())
    return f"W{(x - WEEK1_MONDAY).days // 7 + 1}"


def fill_et(t):
    """IBKR execution time -> (YYYYMMDD, HH:MM:SS) in US Eastern, as paper.et_date reads it: an explicit zone is used as
    is, a bare time is the Gateway's UTC+8."""
    s = " ".join(str(t).split())
    parts = s.split(" ")
    try:
        if len(parts) >= 3:
            dt = datetime.strptime(parts[0] + " " + parts[1], "%Y%m%d %H:%M:%S").replace(tzinfo=ZoneInfo(parts[2]))
        else:
            dt = datetime.strptime(s, "%Y%m%d %H:%M:%S").replace(tzinfo=timezone(timedelta(hours=8)))
        dt = dt.astimezone(ET)
        return dt.strftime("%Y%m%d"), dt.strftime("%H:%M:%S")
    except (ValueError, KeyError):
        return s[:8], s[9:17]


def utc_et_day(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(ET).strftime("%Y%m%d")


def sgn(x):
    return (x > 0) - (x < 0)


def set_order(p):
    """runs/archive/<set> in the order the weeks ran: week2 before week10; a name with no number after the numbered ones."""
    m = re.search(r"(\d+)", p.name)
    return (0, int(m.group(1)), p.name) if m else (1, 0, p.name)


class Ledgers:
    """Every ledger set (runs/archive/<week>/ in week order, then runs/) read once."""

    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.data = self.root / "data"
        self.bad = {}
        sets = sorted((p for p in (self.runs_dir / "archive").glob("*") if p.is_dir()), key=set_order) if (self.runs_dir / "archive").exists() else []
        self.sets = [(p.name, p, p / "daily.jsonl", False) for p in sets] + [("live", self.runs_dir, self.runs_dir / "live" / "daily.jsonl", True)]
        # each set's daily records: (sleeve, YYYYMMDD) -> the day's model and ml_tool, and the days each set holds
        self.day_rec, self.set_days = {}, {}
        for name, d, daily, _ in self.sets:
            days = defaultdict(set)
            for r in jl(daily, self.bad):
                if not (isinstance(r, dict) and r.get("sleeve") and r.get("date")):
                    continue
                k = (r["sleeve"], ymd(r["date"]))
                days[r["sleeve"]].add(k[1])
                rec = self.day_rec.setdefault(k, {})
                for f in ("model", "ml_tool"):
                    if r.get(f) and not rec.get(f):
                        rec[f] = r[f]
            self.set_days[name] = days
        by = {}
        for _, d, _, _ in self.sets:
            for r in jl(d / "fills.jsonl", self.bad):
                if r.get("exec_id") is not None and r.get("time"):
                    by[r["exec_id"]] = r
        self.fills = sorted(by.values(), key=lambda f: (f["time"], f["exec_id"]))
        for f in self.fills:
            f["d"], f["t"] = fill_et(f["time"])
        self.orders = {}
        for _, d, _, _ in self.sets:
            for r in jl(d / "orders.jsonl", self.bad):
                ref = r.get("ref")
                if not ref or "symbol" not in r:
                    continue
                if ref not in self.orders or (self.orders[ref].get("arrival") is None and r.get("arrival")):
                    self.orders[ref] = r
        dv = {}
        for _, d, _, _ in self.sets:
            for r in jl(d / "dividends.jsonl", self.bad):
                if r.get("sleeve") and r.get("symbol") and r.get("ex_date"):
                    dv[(r["sleeve"], r["symbol"], ymd(r["ex_date"]))] = r
        self.dividends = list(dv.values())
        self.splits, seen = [], set()               # each (symbol, ex_date) once, the first row, as paper.splits()
        for r in jl(self.runs_dir / "splits.jsonl", self.bad):
            if r.get("symbol") and r.get("ex_date") and (r["symbol"], ymd(r["ex_date"])) not in seen:
                r["ex_date"] = ymd(r["ex_date"])
                seen.add((r["symbol"], r["ex_date"]))
                self.splits.append(r)
        self.runs = self._runs()

    def _runs(self):
        """[{id, sleeve, start, first, last (None: still running), start_cash, meta}]"""
        rs = {}
        for name, d, daily, live in self.sets:
            sl = jload(d / "sleeves.json")
            if sl is None:
                if (d / "sleeves.json").exists():
                    raise ValueError(f"{d / 'sleeves.json'} does not parse")
                continue
            span = sorted({ymd(r["date"]) for r in jl(daily, self.bad) if r.get("date")})
            if not span:
                fd = [f["d"] for f in self.fills if any(f.get("sleeve") == s for s in sl)]
                span = sorted(fd)
            end = None if live else (span[-1] if span else None)
            for s, m in sl.items():
                k = (s, m.get("start"))
                r = rs.setdefault(k, {"sleeve": s, "start": m.get("start"), "start_cash": float(m["start_cash"]), "meta": m,
                                      "first": utc_et_day(m["start"]) if m.get("start") else (span[0] if span else "00000000"),
                                      "last": end, "live": live, "metas": []})
                r["meta"] = m
                r["metas"].append((name, m))
                if live:
                    r["live"], r["last"] = True, None
                elif not r["live"] and end and (r["last"] is None or end > r["last"]):
                    r["last"] = end
        out = sorted(rs.values(), key=lambda r: (r["sleeve"], r["first"]))
        for a, b in zip(out, out[1:]):
            if a["sleeve"] == b["sleeve"] and (a["last"] is None or a["last"] >= b["first"]):
                a["last"] = (ddate(b["first"]) - timedelta(days=1)).strftime("%Y%m%d")
        for r in out:
            r["id"] = f"{r['sleeve']}/{week_of(r['first'])}"
        return out

    def label(self, run, P):
        """(fam, ml_tool) of the book carried over the close of P, from that day's own records: the daily record of
        (sleeve, P) for model and ml_tool, else the sleeves.json of the first ledger set that holds P (a held sleeve's
        model is its model_held). Never from a later sleeves.json: a later change does not relabel earlier nights."""
        s = run["sleeve"]
        metas = run.get("metas") or [("", run["meta"])]
        m = next((mm for name, mm in metas if P in self.set_days.get(name, {}).get(s, ())), None)
        if m is None:                       # no daily record of P: the first set whose days go past P, else the newest set
            later = [mm for name, mm in metas if self.set_days.get(name, {}).get(s) and max(self.set_days[name][s]) >= P]
            m = later[0] if later else metas[-1][1]
        rec = self.day_rec.get((s, P), {})
        model = rec.get("model") or m.get("model", "")
        if model == HOLD_MODEL:
            model = m.get("model_held") or (m.get("model") if m.get("model") != HOLD_MODEL else HOLD_MODEL)
        # ml_tool: P's daily record (the revision, e.g. "ML2.1"), else its set's sleeves.json, else the first set of the
        # same run that names one (a run keeps its tool version: week 3's sleeves.json predates the field)
        tool = rec.get("ml_tool") or m.get("ml_tool") or (m.get("ml") and next((mm["ml_tool"] for _, mm in metas if mm.get("ml_tool")), ""))
        return family({**m, "model": model}), tool or ""


def family(meta):
    m = str(meta.get("model", ""))
    if m == HOLD_MODEL:
        return "hold"
    if m == "base":
        return "base+tool" if meta.get("ml") else "base"
    if m.startswith("claude"):
        return "claude"
    if m.startswith("thinkingmachines/Inkling"):   # Inkling-2-Small on base weights (inkling.py, 20 Nov 2026)
        return "inkling+tool" if meta.get("ml") else "inkling"
    return "trained"


# ---------------------------------------------------------------------------------------------------- prices
class Prices:
    """Unadjusted open and close per name and day, with where each came from."""

    def __init__(self, L, calendar):
        self.L = L
        self.cal = calendar
        self.calset = set(calendar)
        self.day, self.b30, self.ks, self.flag = {}, {}, {}, {}
        self.arr_close = defaultdict(dict)       # sym -> {ET date of the order: raw close of the session before}
        self.fb_close_src = defaultdict(list)    # (sym, P) -> arrival closes
        for o in L.orders.values():
            a = o.get("arrival") or {}
            if a.get("close") and o.get("time"):
                d = utc_et_day(o["time"])
                self.arr_close[o["symbol"]][d] = a["close"]
                p = self.prev_day(d)
                if p:
                    self.fb_close_src[(o["symbol"], p)].append(a["close"])
        self._dirs = {}
        self.fb_open = {}
        opg = {ref for ref, o in L.orders.items() if o.get("tif") == "OPG"}
        op, early = defaultdict(list), defaultdict(list)
        for f in L.fills:
            if f.get("ref") in opg and "09:30:00" <= f["t"] < "09:31:00":
                op[(f["symbol"], f["d"])].append(f["price"])
            elif "09:30:00" <= f["t"] < "09:35:00":
                early[(f["symbol"], f["d"])].append((f["t"], f["price"]))
        for k, v in op.items():
            self.fb_open[k] = sorted(v)[len(v) // 2]
        for k, v in early.items():
            self.fb_open.setdefault(k, sorted(v)[0][1])

    def prev_day(self, d):
        import bisect
        i = bisect.bisect_left(self.cal, d)
        return self.cal[i - 1] if i > 0 else None

    def _load(self, s):
        if s in self.day:
            return
        daily = {}
        for sub in ("bars_broad", "bars"):                       # data/bars wins
            doc = jload(self.L.data / sub / fname(s))
            if doc:
                daily.update({ymd(b["date"]): (b["open"], b["close"]) for b in doc.get("bars") or []
                              if b.get("close") and b.get("volume", 1)})
        b30 = {}
        doc = jload(self.L.data / "bars30" / fname(s))
        if doc:
            for b in doc.get("bars") or []:
                ds = str(b["date"])
                b30.setdefault(ds[:8], []).append((ds[9:14], b["open"], b["close"]))
        ks = {}
        for d, (o, c) in daily.items():
            bb = b30.get(d)
            if bb and bb[0][0] == "09:30" and o:
                ks[d] = bb[0][1] / o
        flag = "k_bars30"
        if not ks and daily:
            dl = sorted(daily)
            import bisect
            for d_next, ac in self.arr_close.get(s, {}).items():
                i = bisect.bisect_left(dl, d_next)
                if i > 0 and daily[dl[i - 1]][1]:
                    ks.setdefault(dl[i - 1], ac / daily[dl[i - 1]][1])
            flag = "k_from_arrival" if ks else "k_is_1"
        self.day[s], self.b30[s], self.ks[s], self.flag[s] = daily, b30, ks, flag

    def has_daily(self, s, d):
        """d's daily bar of s is in data/bars or data/bars_broad."""
        self._load(s)
        return d in self.day[s]

    def _k(self, s, d):
        """(k, exact): exact when d has its own 09:30 30-minute bar, so the open is that print."""
        ks = self.ks[s]
        if d in ks:
            return ks[d], self.flag[s] == "k_bars30"
        if not ks:
            return 1.0, False
        x = ddate(d)
        best = min(ks, key=lambda y: (abs((ddate(y) - x).days), y))
        return ks[best], False

    def raw(self, s, d):
        """(open, close, open_src, close_src) or None."""
        self._load(s)
        daily, b30 = self.day[s], self.b30[s]
        if d in daily:
            k, exact = self._k(s, d)
            o, c = daily[d]
            return (o * k if o else None, c * k, "bars30" if exact else "daily_k", "daily" if exact else "daily_k")
        if d in b30:
            bb = b30[d]
            return (bb[0][1], bb[-1][2], "bars30" if bb[0][0] == "09:30" else "bars30_late", "bars30_last")
        return None

    def _dir(self, X):
        """[(HHMM, sleeve file stem, {sym: close mark})] of the look records of day X."""
        if X not in self._dirs:
            rows = []
            dd = self.L.runs_dir / "live" / f"{X[:4]}-{X[4:6]}-{X[6:]}"
            if dd.is_dir():
                for f in sorted(dd.glob("*.json")):
                    doc = jload(f)
                    st = doc.get("state") if isinstance(doc, dict) else None
                    if not isinstance(st, dict) or not isinstance(st.get("positions") or [], list):
                        continue
                    marks = {x["symbol"]: x["close"] for x in st.get("positions") or []
                             if isinstance(x, dict) and x.get("symbol") and x.get("close")}
                    rows.append((f.stem[-4:], f.stem, marks))
            self._dirs[X] = rows
        return self._dirs[X]

    def _fb_close(self, s, P):
        v = list(self.fb_close_src.get((s, P), []))
        x = ddate(P) + timedelta(days=1)
        nxt = None
        for _ in range(10):                                       # look dirs after P up to the next trading day
            X = x.strftime("%Y%m%d")
            trading = X in self.calset
            for hm, stem, marks in self._dir(X):
                if (not trading or (hm == "0900" and stem[-5:] == "-0900")) and s in marks:
                    v.append(marks[s])
            if trading:
                nxt = X
                break
            x += timedelta(days=1)
        return sorted(v)[len(v) // 2] if v else None

    def _fb_late(self, s, X):
        best = None
        for hm, stem, marks in self._dir(X):
            if hm.isdigit() and hm != "0900" and s in marks and (best is None or best[0] < hm):
                best = (hm, marks[s])
        return best[1] if best else None

    def oc(self, s, d):
        """(open or None, close, open_src, close_src) or None, fallbacks included."""
        r = self.raw(s, d)
        if r is not None:
            return r
        c, o = self._fb_close(s, d), self.fb_open.get((s, d))
        csrc = "marks"
        if c is None:
            c, csrc = self._fb_late(s, d), "late_look_mark"
        if c is None:
            return None
        return (o, c, "fill" if o is not None else "none", csrc)


# ---------------------------------------------------------------------------------------------------- exposure
class Exposure:
    """Per-name betas as the repo computes them (see the module docstring)."""

    def __init__(self, L, calendar):
        self.L, self.cal = L, calendar
        self.idx = {d: i for i, d in enumerate(calendar)}
        u = L.root / "lists" / "universe523.txt"
        e = L.root / "lists" / "etfs.txt"
        self.universe = {x.strip() for x in u.read_text().splitlines() if x.strip()} if u.exists() else set()
        self.etfs = {x.strip() for x in e.read_text().splitlines() if x.strip()} if e.exists() else set()
        self._c, self._cache = {}, {}

    def closes(self, s):
        if s not in self._c:
            doc = jload(self.L.data / "bars" / fname(s)) or jload(self.L.data / "bars_broad" / fname(s)) or {}
            self._c[s] = {ymd(b["date"]): b["close"] for b in doc.get("bars") or [] if b.get("close")}
        return self._c[s]

    def look_beta(self, s, P):
        """trader.py's beta for the look after the close of P (None where it has none: 1.0 is used)."""
        t = self.idx.get(P)
        if t is None or t < 60:
            return None
        c, m = self.closes(s), self.closes("SPY")
        win = self.cal[t - 60:t + 1]
        cs = np.array([c.get(d, np.nan) for d in win], float)
        ms = np.array([m.get(d, np.nan) for d in win], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r, x = np.diff(np.log(cs)), np.diff(np.log(ms))
        if s in self.universe:
            if np.isnan(cs[-1]):
                return None
        else:
            have = [d for d in c if d <= P]
            if not have or max(have) < self.cal[t - 5]:
                return None
        ok = ~np.isnan(r) & ~np.isnan(x)
        if ok.sum() < 40 or np.var(x[ok], ddof=1) <= 0:
            return None
        raw = np.cov(x[ok], r[ok], ddof=1)[0, 1] / np.var(x[ok], ddof=1)
        if s in self.universe and s not in self.etfs:
            return float(2 / 3 * raw + 1 / 3)
        return float(raw)

    def _rets(self, s):
        c = self.closes(s)
        ds = sorted(c)
        return {b: c[b] / c[a] - 1 for a, b in zip(ds, ds[1:]) if c[a]}

    def chip(self, s, P):
        """b2_books.betas' loading on SMH's residual after SPY, 120 daily returns before P (None under 60)."""
        k = (s, P)
        if k not in self._cache:
            ri, rm, rs = self._rets(s), self._rets("SPY"), self._rets("SMH")
            ds = sorted(d for d in ri if d < P and d in rm and d in rs)[-120:]
            if len(ds) < 60:
                self._cache[k] = None
            else:
                i = np.array([ri[d] for d in ds]); m = np.array([rm[d] for d in ds]); sm = np.array([rs[d] for d in ds])
                X = np.column_stack([np.ones(len(ds)), m])
                e = sm - X @ np.linalg.lstsq(X, sm, rcond=None)[0]
                X2 = np.column_stack([np.ones(len(ds)), m, e])
                self._cache[k] = float(np.linalg.lstsq(X2, i, rcond=None)[0][2])
        return self._cache[k]


# ---------------------------------------------------------------------------------------------------- the nights
def calendar_of(L):
    doc = jload(L.data / "bars" / "SPY.json") or {}
    return sorted({ymd(b["date"]) for b in doc.get("bars") or [] if b.get("close")})


def look_files(L, sleeve, d):
    dd = L.runs_dir / "live" / f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return sorted(dd.glob(f"{sleeve}-[0-9][0-9][0-9][0-9].json")) if dd.is_dir() else []


def look_positions(f):
    lk = jload(f)
    if not isinstance(lk, dict) or not isinstance(lk.get("state"), dict):
        return None, None
    ps = lk["state"].get("positions") or []
    if not isinstance(ps, list):
        return None, None
    return ({x["symbol"]: x["quantity"] for x in ps if isinstance(x, dict) and x.get("symbol") and x.get("quantity")},
            str(lk.get("time", "")))


def run_nights(L, run, PR, EX, cal, upto):
    """Every night (P, D) of one run with D <= upto: ([position rows], [sleeve-night rows])."""
    sleeve, first, last = run["sleeve"], run["first"], run["last"]
    days = [d for d in cal if d >= first and (last is None or d <= last) and d <= upto]
    fl = defaultdict(list)
    for f in L.fills:
        if f.get("sleeve") == sleeve and days and days[0] <= f["d"] <= days[-1]:
            fl[f["d"]].append(f)
    pos, prevC, csrc = {}, {}, {}
    cash = run["start_cash"]
    nav_prev = cash
    prow, nrow = [], []
    for i, D in enumerate(days):
        P = days[i - 1] if i else None
        for sp in L.splits:
            if P is not None and P < sp["ex_date"] <= D and sp["symbol"] in pos:
                pos[sp["symbol"]] *= sp["ratio"]
                prevC[sp["symbol"]] /= sp["ratio"]
        bysym = defaultdict(list)
        for f in fl[D]:
            bysym[f["symbol"]].append(f)
        carried = []
        close_pos, close_src = {}, {}
        for s in sorted(set(pos) | set(bysym)):
            q0 = pos.get(s, 0)
            oc = PR.oc(s, D)
            if oc is None:
                fx = bysym.get(s)
                px = fx[-1]["price"] if fx else prevC.get(s)
                oc = (px, px, "last_fill", "last_fill")
            O, C, osrc, cs = oc
            noopen = O is None
            if noopen:
                O = prevC.get(s, C)
            r = q0
            n = 0
            nt = None
            if q0:
                CP = prevC[s]
                nt = dict(sym=s, q=q0, CP=CP, O=O, C=C, gap=q0 * (O - CP), sess=q0 * (C - O), exited=0, noopen=noopen,
                          osrc="none" if noopen else osrc, csrc=csrc.get(s, ""))
            for f in bysym.get(s, []):
                x = f["shares"] if f["side"] == "BOT" else -f["shares"]
                px = f["price"]
                com = f.get("commission") or 0.0
                cash -= x * px + com
                rem = x
                if r and sgn(x) != sgn(r):
                    e = sgn(x) * min(abs(x), abs(r))
                    nt["sess"] += e * (C - px)
                    nt["exited"] += -e
                    r += e
                    rem = x - e
                n += rem
            if nt is not None:
                carried.append(nt)
            q1 = r + n
            if q1:
                close_pos[s] = q1
                prevC[s] = C
                close_src[s] = cs
        div = 0.0
        for dv in L.dividends:
            if dv["sleeve"] == sleeve and ymd(dv["ex_date"]) == D:
                div += dv["amount"]
                cash += dv["amount"]
        nav = cash + sum(q * prevC[s] for s, q in close_pos.items())
        if P is not None:
            prow_d, nr = night_rows(L, run, P, D, carried, nav_prev, div, PR, EX)
            prow += prow_d
            nrow.append(nr)
        nav_prev = nav
        pos = close_pos
        prevC = {s: prevC[s] for s in close_pos}
        csrc = close_src
    return prow, nrow


def night_rows(L, run, P, D, carried, value_P, div, PR, EX):
    sleeve = run["sleeve"]
    wk = week_of(D)
    weekend = (ddate(D) - ddate(P)).days > 1
    sp = PR.oc("SPY", P), PR.oc("SPY", D)
    sg = soc = scc = None
    if sp[0] and sp[1] and sp[1][0]:
        sg, soc, scc = sp[1][0] / sp[0][1] - 1, sp[1][1] / sp[1][0] - 1, sp[1][1] / sp[0][1] - 1
    fam, ml_tool = L.label(run, P)
    rows = []
    for n in carried:
        s, q = n["sym"], n["q"]
        b = EX.look_beta(s, P)
        bc = EX.chip(s, P)
        rows.append(dict(run=run["id"], fam=fam, sleeve=sleeve, bars_ok=PR.has_daily(s, P) and PR.has_daily(s, D), P=P, D=D, night=f"{P}-{D}", wk=wk, sym=s, q=q,
                         side=1 if q > 0 else -1, w=abs(q) * n["CP"], noopen=n["noopen"], weekend=weekend,
                         g=n["O"] / n["CP"] - 1, oc=n["C"] / n["O"] - 1 if n["O"] else 0.0, cc=n["C"] / n["CP"] - 1,
                         sg=sg, soc=soc, scc=scc, beta=1.0 if b is None else b, pnl_gap=n["gap"], pnl_sess=n["sess"],
                         pnl_cc=q * (n["C"] - n["CP"]), exited=n["exited"],
                         approx_open=(not n["noopen"]) and n["osrc"] not in EXACT_OPEN, open_src=n["osrc"],
                         close_src=n["csrc"], beta_chip=0.0 if bc is None else bc, O=n["O"], CP=n["CP"], C=n["C"]))
    v = value_P
    gross = sum(r["w"] for r in rows)
    net = sum(r["side"] * r["w"] for r in rows)
    gap = sum(r["pnl_gap"] for r in rows)
    top = sorted(rows, key=lambda r: (-r["w"], r["sym"]))[:5]
    # the book checks: P's last look against the fills before it; D's first look against the close book of P
    last_ok = first_ok = None
    lf = look_files(L, sleeve, P)
    if lf:
        lp, lt = look_positions(lf[-1])
        if lp is not None:
            tt = lt[11:19]
            book = defaultdict(float)
            for f in L.fills:
                if f.get("sleeve") == sleeve and run["first"] <= f["d"] <= P and (f["d"] < P or f["t"] < tt):
                    k = 1
                    for x in L.splits:
                        if x["symbol"] == f["symbol"] and f["d"] < x["ex_date"] <= P:
                            k *= x["ratio"]
                    book[f["symbol"]] += k * (f["shares"] if f["side"] == "BOT" else -f["shares"])
            last_ok = {s: q for s, q in book.items() if q} == lp
    ff = look_files(L, sleeve, D)
    if ff:
        lp, _ = look_positions(ff[0])
        if lp is not None:
            want = {r["sym"]: r["q"] for r in rows}
            first_ok = want == lp
    nr = dict(P=P, D=D, night=f"{P}-{D}", sleeve=sleeve, run=run["id"], wk=wk, fam=fam, ml_tool=ml_tool, weekend=weekend, value_P=v, positions=len(rows), gross_usd=gross,
              net_usd=net, gross_x=gross / v if v else None, net_x=net / v if v else None,
              beta_net=sum(r["side"] * r["w"] * r["beta"] for r in rows) / v if v else None,
              chip_tilt=sum(r["side"] * r["w"] * r["beta_chip"] for r in rows) / v if v else None,
              top5="; ".join(f"{r['sym']} {'L' if r['side'] > 0 else 'S'} {r['w'] / v:.3f}" for r in top) if v else "",
              gap_usd=gap, gap_pct=100 * gap / v if v else None, div_usd=div,
              n_approx=sum(r["approx_open"] for r in rows), approx_usd=sum(r["w"] for r in rows if r["approx_open"]),
              n_noopen=sum(r["noopen"] for r in rows), last_look_ok=last_ok, first_look_ok=first_ok,
              last_look=lf[-1].name if lf else "", first_look=ff[0].name if ff else "")
    return rows, nr


def build(root=ROOT, dates=None, upto=None):
    """Position rows and sleeve-night rows for the nights into `dates` (YYYYMMDD set; None: every night)."""
    L = Ledgers(root)
    cal = calendar_of(L)
    if not cal:
        raise ValueError("no SPY daily bars (data/bars/SPY.json)")
    upto = upto or (max(dates) if dates else cal[-1])
    PR, EX = Prices(L, cal), Exposure(L, cal)
    P_all, N_all = [], []
    for run in L.runs:
        p, n = run_nights(L, run, PR, EX, cal, upto)
        P_all += [r for r in p if dates is None or r["D"] in dates]
        N_all += [r for r in n if dates is None or r["D"] in dates]
    return P_all, N_all, L


# ---------------------------------------------------------------------------------------------------- writing
def _read_csv(p):
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(p, cols, rows):
    tmp = p.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    os.replace(tmp, p)


def write(out, prow, nrow, nights):
    """Replace the rows of `nights` (a set of D) in out/positions.csv and out/nights.csv with these."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, cols, new, key in (("positions.csv", POS_COLS, prow, ("D", "run", "sym")), ("nights.csv", NIGHT_COLS, nrow, ("D", "run"))):
        keep = [r for r in _read_csv(out / name) if r.get("D") not in nights]
        rows = keep + new
        rows.sort(key=lambda r: tuple(str(r.get(k)) for k in key))
        _write_csv(out / name, cols, rows)


def recorded(out):
    return {r["D"] for r in _read_csv(Path(out) / "nights.csv")}


def ready(L, cal, D):
    """D's daily SPY bar is in data/bars, and D's 30-minute bars are in data/bars30 (or a later day's are: D's will not
    come)."""
    if D not in cal:
        return False
    doc = jload(L.data / "bars30" / "SPY.json")
    if not doc:
        return True
    ds = {str(b["date"])[:8] for b in doc.get("bars") or []}
    return D in ds or any(x > D for x in ds)


def record(root=ROOT, out=None, dates=None):
    out = Path(out) if out else Path(root) / "runs" / "carry"
    prow, nrow, L = build(root, dates)
    nights = {r["D"] for r in nrow} | (set(dates) if dates else set())
    write(out, prow, nrow, nights)
    return prow, nrow


def status(prow, nrow, nights):
    if not nrow:
        return "carry: nothing new to record"
    ds = sorted(nights)
    gap = sum(r["pnl_gap"] for r in prow)
    return (f"carry: recorded {len(ds)} night(s) into {ds[0]}" + (f"..{ds[-1]}" if len(ds) > 1 else "") +
            f": {len(nrow)} sleeve-nights, {len(prow)} positions, gap ${gap:+,.0f}, "
            f"{sum(r['approx_open'] for r in prow)} approximate opens, {sum(r['noopen'] for r in prow)} with no open")


def missing_inputs(L, cal, day):
    """The ledgers update() needs and cannot read: every set's sleeves.json and fills.jsonl (the live fills.jsonl only
    once a live run has a night to record: a fresh week has none until its first fill), and any line that does not
    parse."""
    miss = []
    for name, d, _, live in L.sets:
        if not isinstance(jload(d / "sleeves.json"), dict):
            miss.append(f"{name}/sleeves.json")
        needs_fills = not live or any(r["live"] and len([x for x in cal if r["first"] <= x <= day]) > 1 for r in L.runs)
        if needs_fills and not (d / "fills.jsonl").exists():
            miss.append(f"{name}/fills.jsonl")
    miss += [f"unreadable lines {L.bad}"] if L.bad else []
    return miss


def late_bars(L, out, cal, day):
    """Recorded nights into the last REDO_DAYS trading days with a position whose daily bar of P or D was missing then
    (bars_ok False) and is in data/bars or data/bars_broad now."""
    recent = set([d for d in cal if d <= day][-REDO_DAYS:])
    PR = Prices(L, cal)
    redo = set()
    for r in _read_csv(Path(out) / "positions.csv"):
        if r.get("D") in recent and r.get("bars_ok") == "False" and PR.has_daily(r["sym"], r["P"]) and PR.has_daily(r["sym"], r["D"]):
            redo.add(r["D"])
    return redo


def update(day=None, root=ROOT, out=None):
    """Record every night into a day <= `day` (default today, ET) not yet in out/nights.csv whose inputs are present
    (ready(), missing_inputs()), and re-record a recent night whose late daily bars have come (late_bars()). Returns a
    one-line status; never raises."""
    try:
        root = Path(root)
        out = Path(out) if out else root / "runs" / "carry"
        day = ymd(day) if day else datetime.now(ET).strftime("%Y%m%d")
        L = Ledgers(root)
        cal = calendar_of(L)
        if not cal:
            return "carry: no SPY daily bars; nothing recorded"
        miss = missing_inputs(L, cal, day)
        if miss:
            return f"carry: nothing recorded; waiting for ledgers: {', '.join(miss)}"
        have = recorded(out)
        want = set()
        for run in L.runs:
            ds = [d for d in cal if d >= run["first"] and (run["last"] is None or d <= run["last"]) and d <= day]
            want |= set(ds[1:])
        todo = sorted(d for d in want - have if ready(L, cal, d))
        waiting = sorted(d for d in want - have if not ready(L, cal, d))
        redo = sorted(late_bars(L, out, cal, day) - set(todo))
        tail = f"; waiting for bars: {', '.join(waiting)}" if waiting else ""
        tail += f"; re-recorded for late daily bars: {', '.join(redo)}" if redo else ""
        if not todo and not redo:
            return "carry: nothing new to record" + tail
        prow, nrow = record(root, out, set(todo) | set(redo))
        return status(prow, nrow, set(todo) | set(redo)) + tail
    except Exception as e:                              # noqa: BLE001 - the loop goes on
        where = traceback.extract_tb(e.__traceback__)[-1]
        return f"carry: failed ({type(e).__name__}: {str(e)[:200]}; line {where.lineno} in {where.name})"


def recent_names(days=10, root=ROOT, today=None):
    """Names with a fill in the last `days` calendar days, in any ledger set (the archives too: a week's fills are archived at
    Monday's reset). live.refresh() adds them to its bars refresh, so a name outside the universe still gets the daily bars
    of its last carried night after it is sold: all 13 position-nights of weeks 2 to 5 with no open were such names, since
    the refresh fetched only the universe and names still held. Never raises: an empty set on any failure."""
    try:
        today = ymd(today) if today else datetime.now(ET).strftime("%Y%m%d")
        lo = (datetime.strptime(today, "%Y%m%d") - timedelta(days=days)).strftime("%Y%m%d")
        return {f["symbol"] for f in Ledgers(root).fills if f.get("symbol") and lo <= f["d"] <= today}
    except Exception:                                   # noqa: BLE001 - the refresh goes on with the names it has
        return set()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true")
    g.add_argument("--date", help="YYYY-MM-DD: the night ending at that day's open")
    ap.add_argument("--out", help="output folder (default runs/carry)")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    dates = None if a.all else {ymd(a.date)}
    prow, nrow = record(Path(a.root), a.out, dates)
    print(status(prow, nrow, {r["D"] for r in nrow} or (dates or set())))


if __name__ == "__main__":
    main()
