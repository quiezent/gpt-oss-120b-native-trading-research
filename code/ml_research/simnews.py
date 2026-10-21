"""Point-in-time news, filings and market tape for rl2's training simulator.

rl2 learned to stop researching because the simulator's tools held nothing new. This module adds the three news
tools of the live gateway to `simgw.SimGateway`, answered from cached archives as they stood at the decision time
(09:00 ET on session d, the same moment simgw's bars stop):

- news(symbol, providers=None, limit=10, before=''): IBKR headlines (data/news, BRFG + BRFUPDN) and EDGAR 8-K /
  8-K/A filings (provider "EDGAR", article_id = accession, headline from the company name and the 8-K items),
  newest first, in the gateway's row format {time (UTC), provider, article_id, headline, meta}.
- article(provider, article_id): EDGAR exhibit text; macro release detail; IBKR (BRFG) article text from the backfill
  in data/news/articles/<provider>/<id>.json (ml_research/news_fetch.py --articles), else "full text not available
  for past dates".
- tape(): market-wide items: first the macro rows (macro_feats.tape_rows, the same rows the live tape prepends:
  releases of the last 7 days before today with first-release values, FOMC decisions, today's economic calendar),
  then Briefing.com (BRFG) headlines across the universe, newest first.

Visibility: an item is returned only if its time is strictly before the decision time. EDGAR uses accepted_utc
(the raw acceptanceDateTime is in two conventions; see data/edgar/README.md). Macro releases show from the day after
(the live calendar is rebuilt each evening); rows whose release time is unverified (time_uncertain) count as released
at 10:00 ET, the latest plausible time. A Kalshi expectation is shown only if it was priced before the release.
article() refuses an id that isn't visible yet, with the same message as an unknown id.

The live tools (ml_research/livenews.py) serve the same EDGAR rows, EDGAR article text and macro rows next to the
gateway's own, so training and live see one source set in one format.

    gw = simnews.NewsSimGateway(data, d, pre, forecast=None)    # a SimGateway with news, article and tape
    tools.available(gw)                                          # now offers news, tape and article

Known limits: the IBKR archive is thin (median 16 headlines a symbol, 83% analyst actions, dense only from Nov
2024); there is no Dow Jones tape (so the tape description here leaves Dow Jones out); company names are today's; the 8-K time is the SEC acceptance, which for about
2.6% of earnings filings lags a pre-market press release (late, never early).
"""

import bisect
import csv
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import simgw  # noqa: E402
from ml_research import macro_feats  # noqa: E402
ET = ZoneInfo("America/New_York")
DECISION_HOUR = 9                     # the simulator's look: 09:00 ET, before the open
TAPE_DAYS = macro_feats.TAPE_DAYS     # tape lookback, calendar days
ARTICLES = ROOT / "data" / "news" / "articles"
NO_TEXT = "full text not available for past dates"
NOT_FOUND = "10172: no article with this id is available"

ITEMS = {"1.01": "Material Agreement", "1.02": "Agreement Terminated", "1.03": "Bankruptcy or Receivership",
         "1.04": "Mine Safety", "1.05": "Cybersecurity Incident", "2.01": "Acquisition or Disposition Completed",
         "2.02": "Results of Operations", "2.03": "New Debt Obligation", "2.04": "Debt Acceleration Trigger",
         "2.05": "Exit or Restructuring Costs", "2.06": "Material Impairment", "3.01": "Delisting Notice",
         "3.02": "Unregistered Equity Sale", "3.03": "Change to Shareholder Rights", "4.01": "Auditor Change",
         "4.02": "Prior Financials Not to Be Relied On", "5.01": "Change in Control", "5.02": "Director or Officer Change",
         "5.03": "Bylaw or Fiscal Year Change", "5.04": "Benefit Plan Trading Suspension", "5.05": "Code of Ethics Change",
         "5.06": "Shell Status Change", "5.07": "Shareholder Vote Results", "5.08": "Director Nomination Dates",
         "7.01": "Regulation FD Disclosure", "8.01": "Other Events", "9.01": "Financial Statements and Exhibits"}


def ts(s):
    """Epoch seconds from an ISO time ('...Z', '+00:00', or naive = UTC) or 'YYYY-MM-DD HH:MM:SS' (UTC)."""
    s = str(s).strip().replace(" ", "T")
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def iso_utc(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def decision_ts(yyyymmdd, hour=DECISION_HOUR):
    s = str(yyyymmdd).replace("-", "")
    return int(datetime(int(s[:4]), int(s[4:6]), int(s[6:8]), hour, 0, tzinfo=ET).timestamp())


def _et(t, fmt="%Y-%m-%d %H:%M ET"):
    return datetime.fromtimestamp(t, ET).strftime(fmt)


# ---------------------------------------------------------------- archive
def _order(x):
    return (x[0], x[1]["provider"], x[1]["article_id"])


class Archive:
    """Every dated item, sorted by time. Built once from disk; tests build perturbed copies by hand."""

    def __init__(self, sym_items, edgar, macro, brfg, names=None, edgar_text_dir=None, article_dir=None):
        self.sym = {s: sorted(v, key=_order) for s, v in sym_items.items()}  # sym -> [(t, row)]
        self.sym_t = {s: [x[0] for x in v] for s, v in self.sym.items()}
        self.edgar = edgar                    # accession -> {"t", "symbols", "form", "items", "name", "has_text"}
        self.macro = sorted(macro, key=_order)                   # [(t, row)] tape items incl. calendar lines
        self.macro_t = [x[0] for x in self.macro]
        self.macro_by_id = {r["article_id"]: (t, r) for t, r in self.macro}
        self.brfg = sorted(brfg, key=_order)
        self.brfg_t = [x[0] for x in self.brfg]
        self.ibkr = {(r["provider"], r["article_id"]): t for v in self.sym.values() for t, r in v if r["provider"] != "EDGAR"}
        self.names = names or {}
        self.edgar_text_dir = Path(edgar_text_dir) if edgar_text_dir else ROOT / "data" / "edgar" / "text"
        self.article_dir = Path(article_dir) if article_dir else ARTICLES

    @classmethod
    def load(cls, root=ROOT):
        names = company_names(root)
        sym_items, brfg, seen = {}, [], set()
        for f in sorted((root / "data" / "news").glob("*.json")):
            x = json.loads(f.read_text(encoding="utf-8"))
            sym = x.get("symbol") or f.stem.replace("_", " ")
            rows = sym_items.setdefault(sym, [])
            for h in x.get("headlines", []):
                r = {"time": h["time"], "provider": h["provider"], "article_id": h["article_id"],
                     "headline": h["headline"], "meta": h.get("meta", "")}
                t = ts(h["time"])
                rows.append((t, r))
                if h["provider"] == "BRFG" and h["article_id"] not in seen:
                    seen.add(h["article_id"])
                    brfg.append((t, r))
        edgar = {}
        with open(root / "data" / "edgar" / "filings.csv", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                t = ts(row["accepted_utc"])
                syms = [s for s in row["symbols"].split("|") if s]
                name = names.get(syms[0]) if syms else None
                items = [i.strip() for i in row["items"].split(",") if i.strip()]
                acc = row["accessionNumber"]
                edgar[acc] = {"t": t, "symbols": syms, "form": row["form"], "items": items, "name": name,
                              "has_text": row.get("text_status") == "ok"}
                for s in syms:
                    sym_items.setdefault(s, []).append((t, edgar_row(acc, edgar[acc], s, names.get(s))))
        cal = json.loads((root / "data" / "macro" / "calendar.json").read_text(encoding="utf-8"))["events"]
        return cls(sym_items, edgar, macro_feats.tape_items(cal), brfg, names)

    # ---- queries (cutoff: epoch seconds; items strictly before it)
    def symbol(self, sym, cutoff):
        rows, tt = self.sym.get(sym, []), self.sym_t.get(sym, [])
        return rows[:bisect.bisect_left(tt, cutoff)]

    def tape(self, cutoff, since):
        m = self.macro[bisect.bisect_left(self.macro_t, since):bisect.bisect_left(self.macro_t, cutoff)]
        b = self.brfg[bisect.bisect_left(self.brfg_t, since):bisect.bisect_left(self.brfg_t, cutoff)]
        return m + b


_ARCHIVE = None


def archive():
    global _ARCHIVE
    if _ARCHIVE is None:
        _ARCHIVE = Archive.load()
    return _ARCHIVE


def company_names(root=ROOT):
    """symbol -> display name: the S&P 500 list's 'Security' (data/sp500.html), else the SEC title."""
    names = {}
    try:
        cm = json.loads((root / "data" / "edgar" / "cik_map.json").read_text(encoding="utf-8"))
        for s, v in cm.items():
            t = str(v.get("title") or s)
            names[s] = t.title() if t.isupper() else t
    except (OSError, ValueError):
        pass
    try:
        import pandas as pd
        t = pd.read_html(root / "data" / "sp500.html")[0]
        for s, n in zip(t["Symbol"], t["Security"]):
            names[str(s).replace(".", " ")] = str(n)
    except Exception:
        pass
    return {s: re.sub(r"\s*\(Class [A-Z]\)", "", n) for s, n in names.items()}


def edgar_row(acc, f, sym, name=None):
    items = [i for i in f["items"] if i != "9.01"] or f["items"]
    what = "; ".join(f"{ITEMS.get(i, 'Item')} ({i})" for i in items) or "no items listed"
    name = name or f.get("name") or sym
    form = f["form"] + (" (amendment)" if f["form"].endswith("/A") else "")
    return {"time": iso_utc(f["t"]), "provider": "EDGAR", "article_id": acc,
            "headline": f"{name} ({sym}) files {form}: {what}", "meta": ""}


def edgar_article(f, body):
    """article() text of an 8-K (simulator and live): company, form, SEC acceptance time (ET) and items, then the press
    release or filing text (up to edgar_fetch.TEXT_CHARS)."""
    items = "; ".join(f"{i} {ITEMS.get(i, '')}".strip() for i in f["items"])
    head = (f"{f.get('name') or ''} ({', '.join(f['symbols'])}) {f['form']}, accepted by the SEC {_et(f['t'])}. "
            f"Items: {items}.")
    body = re.sub(r"\A\s*(?:EX-[\d.]+|8-K(?:/A)?)[ \t]*\n+", "", body or "")   # a leftover document-type line (most archive texts)
    return head + "\n\n" + (body or "No press release or exhibit text is cached for this filing; the items above are what it covers.")


# ---------------------------------------------------------------- the gateway mixin
class NewsMixin:
    """Adds news, article and tape to a simulated gateway with .data and .d (the session about to trade)."""

    _archive = None

    @property
    def news_archive(self):
        return self._archive or archive()

    @property
    def cutoff(self):
        return decision_ts(self.data.dates[self.d]) if self.d < len(self.data.dates) else decision_ts(
            self.data.next_date)

    def news(self, symbol, providers=None, limit=10, before=""):
        sym = str(symbol).upper().strip().replace(".", " ")
        A = self.news_archive
        if sym not in A.sym and sym not in self.data.k:
            raise ValueError(f"200: No security definition has been found for {symbol}")
        cut = self.cutoff
        if before:
            cut = min(cut, ts(before) + 1)          # IBKR's before is inclusive (the boundary headline may repeat)
        rows = A.symbol(sym, cut)
        if providers:
            want = set(providers.split("+") if isinstance(providers, str) else providers)
            rows = [x for x in rows if x[1]["provider"] in want]
        rows = rows[::-1][:max(0, int(limit))]
        return [{k: v for k, v in r.items() if not k.startswith("_")} for _, r in rows]

    def article(self, provider, article_id):
        p, aid, A, cut = str(provider).upper().strip(), str(article_id).strip(), self.news_archive, self.cutoff
        if p == "EDGAR":
            f = A.edgar.get(aid)
            if not f or f["t"] >= cut:
                raise ValueError(NOT_FOUND)
            body = ""
            if f["has_text"]:
                try:
                    body = (A.edgar_text_dir / f"{aid}.txt").read_text(encoding="utf-8")
                except OSError:
                    body = ""
            return {"provider": "EDGAR", "article_id": aid, "text": edgar_article(f, body)}
        if p == "MACRO":
            x = A.macro_by_id.get(aid)
            if not x or x[0] >= cut or not macro_feats.on_tape(x[0], aid, datetime.fromtimestamp(cut, ET)):
                raise ValueError(NOT_FOUND)
            r = x[1]
            return {"provider": "MACRO", "article_id": aid,
                    "text": macro_feats.macro_detail(r["_detail"]) if "_detail" in r else r["headline"]}
        t = A.ibkr.get((p, aid))
        if t is None or t >= cut:
            raise ValueError(NOT_FOUND)
        try:
            text = json.loads((A.article_dir / p / f"{aid}.json").read_text(encoding="utf-8")).get("text")
        except (OSError, ValueError):
            text = None
        return {"provider": p, "article_id": aid, "text": text or f"{NO_TEXT} (headline only)."}

    def tape(self, providers=None, seconds=3.0, limit=50):
        cut, A = self.cutoff, self.news_archive
        mac = macro_feats.tape_rows(datetime.fromtimestamp(cut, ET), A.macro)
        rows = A.brfg[bisect.bisect_left(A.brfg_t, cut - TAPE_DAYS * 86400):bisect.bisect_left(A.brfg_t, cut)]
        if providers:
            want = set(providers.split("+") if isinstance(providers, str) else providers)
            rows = [x for x in rows if x[1]["provider"] in want]
        rows = (mac + sorted(rows, key=_order, reverse=True))[:max(0, int(limit))]
        return [{k: v for k, v in r.items() if not k.startswith("_")} for _, r in rows]


class NewsSimGateway(NewsMixin, simgw.SimGateway):
    """simgw.SimGateway plus point-in-time news, article and tape."""

    tape_sources = "Briefing.com"         # tools.available: the archive has no Dow Jones tape (DJNL)

    def __init__(self, data, d, pre, forecast=None, news_archive=None):
        super().__init__(data, d, pre, forecast=forecast)
        self._archive = news_archive


# ---------------------------------------------------------------- information density
TIER1 = re.compile(r"^(CPI|Payrolls|PCE|GDP|FOMC decision|FOMC statement)")


def window_items(A, sym, hi, lo):
    return [r for t, r in A.symbol(sym, hi) if t >= lo]


def density(data, dates_idx, archive_=None, sessions=5, symbols=None):
    """Per date: for each candidate (the market table's stocks, or `symbols`), the providers with items in
    [09:00 ET of session d-sessions, 09:00 ET of session d), whether one was an earnings 8-K (2.02), and the
    tier-1 macro headlines on the tape in the same window."""
    A = archive_ or archive()
    out = []
    for d in dates_idx:
        if symbols is None:
            rows, _ = data.table_rows(d)
            cands = [s for s in rows if s not in data.etf]
        else:
            cands = symbols
        hi, lo = decision_ts(data.dates[d]), decision_ts(data.dates[d - sessions])
        per = {}
        for s in cands:
            rs = window_items(A, s, hi, lo)
            per[s] = {"providers": sorted({r["provider"] for r in rs}),
                      "earnings": any(r["provider"] == "EDGAR" and "(2.02)" in r["headline"] for r in rs)}
        tier1 = [r["headline"] for t, r in A.tape(hi, lo) if r["provider"] == "MACRO" and TIER1.match(r["headline"])]
        out.append({"date": data.dates[d], "cands": per, "tier1": tier1})
    return out


def density_summary(rows, boot=2000, seed=0):
    """D1 (dates with any candidate item), D2 (candidate-date pairs with an item, by source; date-clustered
    bootstrap 95% interval), earnings share, D4 (dates with a tier-1 macro item on the tape)."""
    import numpy as np
    n = len(rows)
    any_d = np.array([any(c["providers"] for c in r["cands"].values()) for r in rows], float)
    keys = {"any": lambda c: bool(c["providers"]), "EDGAR": lambda c: "EDGAR" in c["providers"],
            "BRFUPDN": lambda c: "BRFUPDN" in c["providers"], "BRFG": lambda c: "BRFG" in c["providers"],
            "earnings_8k": lambda c: c["earnings"]}
    hits = {k: np.array([sum(f(c) for c in r["cands"].values()) for r in rows], float) for k, f in keys.items()}
    size = np.array([len(r["cands"]) for r in rows], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (boot, n))
    out = {"dates": n, "mean_candidates": round(float(size.mean()), 1),
           "D1_any_candidate_item": round(float(any_d.mean()), 4),
           "D4_tier1_macro_on_tape": round(float(np.mean([bool(r["tier1"]) for r in rows])), 4)}
    for k, h in hits.items():
        est = h.sum() / size.sum()
        bs = h[idx].sum(1) / size[idx].sum(1)
        out[f"D2_{k}"] = {"share": round(float(est), 4), "ci95": [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)]}
    return out


def density_run():
    import random
    import trader
    data = trader.Data(last="20261015")
    first = next(i for i, x in enumerate(data.dates) if x >= "20250103")
    last = max(i for i, x in enumerate(data.dates) if x <= "20251212")
    pool = [d for d in range(first, last + 1) if d - 5 >= 260 and d + 10 < len(data.dates)]
    rng = random.Random(20261017)                       # train2's date stream: seed, 16 groups x 20 steps, burn 5
    stream = []
    for _ in range(20):
        stream += rng.sample(pool, 16)
        [rng.randint(0, 5) for _ in range(16)]
        [rng.random() < 0.5 for _ in range(16)]
    stocks = [s for s in data.sym if s not in data.etf]
    res = {"window": "items in [09:00 ET of session d-5, 09:00 ET of session d)", "pool": [data.dates[pool[0]], data.dates[pool[-1]]]}
    res["a_pool_table"] = density_summary(density(data, pool))
    res["b_stream_table"] = density_summary(density(data, stream))
    res["c_pool_universe"] = density_summary(density(data, pool, symbols=stocks))
    for half, (lo, hi) in {"2025H1": ("20250101", "20250630"), "2025H2": ("20250701", "20251231")}.items():
        res[f"a_pool_table_{half}"] = density_summary(density(data, [d for d in pool if lo <= data.dates[d] <= hi]))
    out = ROOT / "runs" / "ml" / "research" / "simnews_density.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    if "--density" in sys.argv:
        density_run()
        sys.exit()
    import tools
    import trader
    data = trader.Data(last="20261015")
    d = data.ix[sys.argv[1]] if len(sys.argv) > 1 else data.ix["20250506"]
    gw = NewsSimGateway(data, d, simgw.Precomputed(data))
    print("decision", _et(gw.cutoff), [s["name"] for s in tools.available(gw)])
    print(tools.run(gw, "tape", {}))
    for s in sys.argv[2:] or ["AAPL"]:
        print(tools.run(gw, "news", {"symbol": s}))
