"""Live news, article and tape for tools.run: the IBKR gateway's rows plus the same EDGAR 8-K and macro rows that the
training simulator serves (ml_research/simnews.py), so live looks and training see one source set in one format
(synthesis_wave_c.md, Step 0, S1-S3). Every addition falls back to the plain IBKR result plus a short note.

- news(gw, symbol, limit): IBKR headlines merged with the company's 8-K / 8-K/A rows (provider EDGAR, article_id =
  accession), newest first. 8-Ks come from data/edgar/filings.csv and, for newer ones, from the submissions JSON that
  events.live fetches (cached per look in data/edgar/live/<day>/); times from companies.json's raw_convention.
- article(gw, provider, article_id): EDGAR = the filing's EX-99.1 (or main document) text, extracted as edgar_fetch
  does, capped at 6,000 characters, with simnews's header; cached in data/edgar/live/text/. MACRO = the release detail.
  Anything else goes to the gateway.
- tape(gw): the macro rows (macro_feats.tape_rows) first, then the gateway's tape.
"""

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EDGAR = ROOT / "data" / "edgar"
LIVE = EDGAR / "live"
NEW_FROM = "2026-09-01"            # submissions rows from here on are checked against the archive (events.live does the same)
_SEEN = {}                          # accession -> filing dict, for article() of filings found live
_COMP = None


def _companies():
    global _COMP
    if _COMP is None:
        _COMP = json.loads((EDGAR / "companies.json").read_text(encoding="utf-8"))
    return _COMP


def _live_filing(acc, rc, k, c, syms, name):
    from ml_research import edgar_fetch as ef
    t = int(ef.raw_to_utc(rc["acceptanceDateTime"][k], c.get("raw_convention") or "shifted").timestamp())
    return {"t": t, "symbols": syms, "form": rc["form"][k], "name": name, "cik": c["cik"], "accessionNumber": acc,
            "items": [i.strip() for i in (rc.get("items", [""] * (k + 1))[k] or "").split(",") if i.strip()],
            "primaryDocument": rc.get("primaryDocument", [""] * (k + 1))[k]}


def edgar_rows(sym, now=None):
    """([(t, row)] of the company's 8-K / 8-K/A filings in simnews's row format, sorted by time; notes).
    Empty for symbols with no SEC filer in the universe (ETFs), with no request."""
    from ml_research import events
    from ml_research import simnews as sn
    c = _companies().get(sym)
    if c is None:
        return [], []
    A = sn.archive()
    rows = [(t, r) for t, r in A.sym.get(sym, []) if r["provider"] == "EDGAR"]
    seen = {r["article_id"] for _, r in rows}
    notes = []
    j, _ = events._submissions_live(c["cik"], now or datetime.now(timezone.utc), notes, sym)
    rc = (j or {}).get("filings", {}).get("recent", {})
    syms = [s for s, v in _companies().items() if v.get("cik") == c["cik"]]
    for k, acc in enumerate(rc.get("accessionNumber", [])):
        if rc["form"][k] not in ("8-K", "8-K/A") or acc in seen or rc["filingDate"][k] < NEW_FROM:
            continue
        f = _SEEN[acc] = _live_filing(acc, rc, k, c, syms, A.names.get(syms[0]))
        rows.append((f["t"], sn.edgar_row(acc, f, sym, A.names.get(sym))))
    return sorted(rows, key=sn._order), notes


def news(gw, sym, limit):
    """(rows, note): the gateway's headlines merged with the company's 8-K rows, newest first, at most `limit`."""
    from ml_research import simnews as sn
    rows = gw.news(sym, limit=limit)
    try:
        ed, notes = edgar_rows(sym)
    except Exception as e:                                  # noqa: BLE001 - never fail the tool over EDGAR
        return rows, f"note: SEC 8-K rows unavailable ({type(e).__name__})"
    now = time.time()
    merged = sorted([(sn.ts(r["time"]), r) for r in rows] + [x for x in ed if x[0] < now], key=sn._order, reverse=True)
    return [r for _, r in merged[:max(0, int(limit))]], ("note: " + "; ".join(notes) if notes else "")


def _find(acc):
    """The filing dict for an accession: the archive, this process's live rows, or the cached submissions JSON."""
    from ml_research import simnews as sn
    A = sn.archive()
    if acc in A.edgar:
        return dict(A.edgar[acc], archive=True)
    if acc in _SEEN:
        return _SEEN[acc]
    cut = (datetime.now(timezone.utc) - timedelta(days=8)).strftime("%Y%m%d")
    by_cik = {}
    for s, v in _companies().items():
        by_cik.setdefault(v.get("cik"), []).append(s)
    for day in sorted((d for d in LIVE.glob("2*") if d.name >= cut), reverse=True):
        for p in day.glob("CIK*.json"):
            rc = json.loads(p.read_text(encoding="utf-8")).get("filings", {}).get("recent", {})
            if acc in rc.get("accessionNumber", []):
                k = rc["accessionNumber"].index(acc)
                syms = by_cik.get(int(p.stem[3:]), [])
                if not syms or rc["form"][k] not in ("8-K", "8-K/A"):
                    return None
                c = _companies()[syms[0]]
                return _live_filing(acc, rc, k, c, syms, A.names.get(syms[0]))
    return None


def edgar_text(acc):
    """article() text for an 8-K accession, or None if unknown. Text as in the simulator: the archive's text file, or
    for a newer filing the same extraction (edgar_fetch.fetch_text) cached in data/edgar/live/."""
    from ml_research import edgar_fetch as ef
    from ml_research import simnews as sn
    f = _find(acc)
    if f is None:
        return None
    body = ""
    if f.get("archive"):
        if f.get("has_text"):
            p = EDGAR / "text" / f"{acc}.txt"
            body = p.read_text(encoding="utf-8") if p.exists() else ""
    elif ef.wants_text({"items": ",".join(f["items"])}):
        try:
            m = ef.fetch_text({"accessionNumber": acc, "cik": f["cik"], "primaryDocument": f["primaryDocument"]},
                              idx=LIVE / "index", text=LIVE / "text")
            if m.get("status") == "ok":
                body = (LIVE / "text" / f"{acc}.txt").read_text(encoding="utf-8")
        except Exception as e:                              # noqa: BLE001
            body = f"Filing text unavailable ({type(e).__name__}); the items above are what it covers."
    return sn.edgar_article(f, body)


def article(gw, provider, article_id):
    from ml_research import macro_feats
    from ml_research import simnews as sn
    p, aid = str(provider).upper().strip(), str(article_id).strip()
    if p in ("EDGAR", "MACRO"):
        text = edgar_text(aid) if p == "EDGAR" else macro_feats.tape_article(aid)
        if text is None:
            raise ValueError(sn.NOT_FOUND)
        return {"provider": p, "article_id": aid, "text": text}
    return gw.article(provider, article_id)


def tape(gw):
    """(rows, note): the macro rows (newest first), then the gateway's tape."""
    from ml_research import macro_feats
    rows = gw.tape(seconds=5)
    try:
        mac = [{k: v for k, v in r.items() if not k.startswith("_")} for _, r in macro_feats.tape_rows()]
    except Exception as e:                                  # noqa: BLE001 - never fail the tool over the calendar
        return rows, f"note: macro rows unavailable ({type(e).__name__})"
    return mac + rows, ""
