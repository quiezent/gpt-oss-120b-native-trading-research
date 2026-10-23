"""Build data/edgar/: SEC 8-K filings (items, acceptance time) for the universe, plus press-release text.

Sources (free, no key): https://www.sec.gov/files/company_tickers.json, https://data.sec.gov/submissions/CIK##########.json
(+ its older "files" pages) and the filing index / documents under https://www.sec.gov/Archives/edgar/data/.
Polite: one shared limiter at <= 4.5 requests/s with the declared User-Agent; backs off on 429/403/5xx.
Every response is cached on disk, so a rerun only fetches what is missing (and refreshes the submissions JSON with --refresh).

acceptanceDateTime in the submissions JSON is not reliably UTC although it ends in "Z": per company it is either true UTC
or true UTC + the Eastern offset. Times here come from the filing index page ("Accepted", Eastern) where fetched, else from
the field under the company's calibrated convention. See data/edgar/README.md and calibrate().

    .venv\\Scripts\\python.exe ml_research\\edgar_fetch.py              # everything
    .venv\\Scripts\\python.exe ml_research\\edgar_fetch.py --no-text    # filings only
    .venv\\Scripts\\python.exe ml_research\\edgar_fetch.py --refresh    # re-pull submissions JSON (new filings)
"""

import csv
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "edgar"
RAW = OUT / "raw"
SUBS = RAW / "submissions"
IDX = OUT / "index"
TEXT = OUT / "text"
UA = "Trader Training research contact via github.com/quiezent"
SINCE = "2023-09-01"          # keep filings from here (bars start 2023-10-16; a few weeks earlier gives event age)
TEXT_ITEMS = ("2.02", "7.01", "8.01")
TEXT_CHARS = 6000
RATE = 4.5                    # requests per second, all threads together (SEC limit is 10; task limit 5)
ET = ZoneInfo("America/New_York")
UTC = timezone.utc

# Current CIK -> predecessor CIKs whose 8-Ks (before the successor's first 8-K) belong to the same listed stock:
# Exxon Mobil Corp -> ExxonMobil Holdings (Jul 2026), BlackRock Inc. (now BlackRock Finance) -> new BlackRock (Oct 2024),
# Bunge Ltd -> Bunge Global SA (Nov 2023). Found by checking every symbol's 2.02 count and filing history start.
PREDECESSORS = {2115436: [34088], 2012383: [1364742], 1996862: [1144519]}

# NYSE early closes (13:00 ET) in the data window (2026-2027 checked against nyse.com/markets/hours-calendars, 18 Oct 2026).
EARLY = {"2023-11-24", "2024-07-03", "2024-11-29", "2024-12-24", "2025-07-03", "2025-11-28", "2025-12-24",
         "2026-11-27", "2026-12-24", "2027-11-26"}
# NYSE full holidays after the last daily bar, through 2027 (same source).
HOLIDAYS_AFTER = {"2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
                  "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"}
CAL_END = date(2027, 12, 31)
HOLIDAYS_BEFORE = {"2023-09-04"}


class Limiter:
    def __init__(self, rate):
        self.dt, self.next, self.lock = 1.0 / rate, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.dt
        if t > now:
            time.sleep(t - now)


LIM = Limiter(RATE)
_local = threading.local()
STATS = {"requests": 0, "retries": 0, "fail": 0}


def session():
    s = getattr(_local, "s", None)
    if s is None:
        s = _local.s = requests.Session()
        s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    return s


def get(url, binary=False):
    """GET with the shared rate limit; backs off on 429/403/5xx. Returns text/bytes, or None on 404 / give-up."""
    delay = 10
    for attempt in range(7):
        LIM.wait()
        STATS["requests"] += 1
        try:
            r = session().get(url, timeout=60)
        except requests.RequestException:
            STATS["retries"] += 1
            time.sleep(delay)
            delay = min(delay * 2, 300)
            continue
        if r.status_code == 200:
            return r.content if binary else r.text
        if r.status_code == 404:
            return None
        if r.status_code in (403, 429) or r.status_code >= 500:
            STATS["retries"] += 1
            print(f"  {r.status_code} on {url}; backing off {delay}s", file=sys.stderr, flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 600)
            continue
        break
    STATS["fail"] += 1
    print(f"  gave up on {url}", file=sys.stderr, flush=True)
    return None


def cached(path, url, refresh=False):
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    t = get(url)
    if t is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(t, encoding="utf-8")
    return t


# ---------------------------------------------------------------- time

def to_utc(raw):
    """True UTC from a submissions-JSON acceptanceDateTime written in the "shifted" convention (about 57% of companies).

    There the field is the true UTC instant shifted forward by the Eastern UTC offset (it equals ET wall clock + 2 x offset),
    e.g. Apple's 8-K accepted 2024-08-01 16:30:26 ET (filing index "Accepted"; 20:30:26 UTC) shows "2024-08-02T00:30:26.000Z".
    So: true UTC = U such that U + offset_ET(U) == raw."""
    s = datetime.strptime(raw[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
    for h in (4, 5):
        u = s - timedelta(hours=h)
        if -u.astimezone(ET).utcoffset() == timedelta(hours=h):
            return u
    return s - timedelta(hours=4)   # unreachable except inside a DST gap


def raw_to_utc(raw, convention):
    """convention "utc": the field is true UTC; "shifted": see to_utc()."""
    if convention == "utc":
        return datetime.strptime(raw[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
    return to_utc(raw)


def index_time(f):
    """The filing index page's 'Accepted' time (EDGAR's Eastern time), cached in data/edgar/index/."""
    acc, cik = f["accessionNumber"], f.get("cik_filer") or f["cik"]
    p = IDX / f"{acc}.json"
    if p.exists():
        m = json.loads(p.read_text())
        if m.get("accepted_et_index"):
            return m["accepted_et_index"]
    html = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{acc}-index.htm")
    a = parse_index(html)["accepted_et"] if html else None
    if not p.exists():
        IDX.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"accepted_et_index": a, "status": "index_only"}), encoding="utf-8")
    return a


def calibrate(filings):
    """Per filer CIK, which convention its submissions JSON uses for acceptanceDateTime.

    It is not uniform: for some companies the field is true UTC, for others true UTC + the Eastern offset (see README).
    Decided against the filing index pages ('Accepted', Eastern); a filer with none in hand gets its latest 8-K's page."""
    by = {}
    for f in filings:
        by.setdefault(f["cik_filer"], []).append(f)
    conv = {}
    for cik, fs in by.items():
        known = [f for f in fs if f.get("accepted_et_index")]
        if not known:
            f = max(fs, key=lambda x: x["filingDate"])
            f["accepted_et_index"] = index_time(f)
            known = [f] if f["accepted_et_index"] else []
        votes = {"utc": 0, "shifted": 0}
        for f in known:
            idx = datetime.strptime(f["accepted_et_index"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=ET).astimezone(UTC)
            for c in votes:
                votes[c] += raw_to_utc(f["acceptanceDateTime"], c) == idx
        conv[cik] = max(votes, key=votes.get) if any(votes.values()) else None
        if votes["utc"] and votes["shifted"]:
            print(f"  CIK {cik}: mixed conventions {votes}", flush=True)
    return conv


def trading_calendar():
    bars = json.loads((ROOT / "data" / "bars" / "SPY.json").read_text())["bars"]
    days = [f"{b['date'][:4]}-{b['date'][4:6]}-{b['date'][6:8]}" for b in bars]
    d, before = date.fromisoformat(SINCE), []      # weekdays before the first bar (NYSE: Labor Day 4 Sep 2023 shut)
    while d.isoformat() < days[0]:
        if d.weekday() < 5 and d.isoformat() not in HOLIDAYS_BEFORE:
            before.append(d.isoformat())
        d += timedelta(days=1)
    days = before + days
    d = date.fromisoformat(days[-1])
    while d < CAL_END:
        d += timedelta(days=1)
        if d.weekday() < 5 and d.isoformat() not in HOLIDAYS_AFTER:
            days.append(d.isoformat())
    return days


def classify(et, cal, calset):
    """Session relative to the regular NYSE session: before_open / during / after_close / closed_day.
    first_session = the first regular session whose prices can reflect the filing."""
    day = et.date().isoformat()
    hm = et.hour * 60 + et.minute + et.second / 60
    close = 13 * 60 if day in EARLY else 16 * 60
    nxt = next((d for d in cal if d > day), None)
    if day not in calset:
        return "closed_day", nxt
    if hm < 9 * 60 + 30:
        return "before_open", day
    if hm < close:
        return "during", day
    return "after_close", nxt


# ---------------------------------------------------------------- tickers and submissions

def universe():
    U = [l.strip() for l in (ROOT / "lists" / "universe523.txt").read_text().splitlines() if l.strip()]
    E = {l.strip() for l in (ROOT / "data" / "etfs.txt").read_text().splitlines() if l.strip()}
    return U, E


def wiki_ciks():
    """CIK column of data/sp500.html (Wikipedia S&P 500 list), only as a cross-check."""
    h = (ROOT / "data" / "sp500.html").read_text(encoding="utf-8")
    out = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", h, re.S):
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        if len(cells) >= 7 and re.fullmatch(r"\d{10}", cells[6]):
            out[cells[0].replace(".", "-")] = int(cells[6])
    return out


def cik_map(refresh):
    t = cached(RAW / "company_tickers.json", "https://www.sec.gov/files/company_tickers.json", refresh)
    ct = json.loads(t)
    m = {}
    for v in ct.values():
        m.setdefault(v["ticker"].upper(), v)
    U, E = universe()
    wk = wiki_ciks()
    out, missing, mismatch = {}, [], []
    for s in U:
        if s in E:
            continue
        k = s.replace(" ", "-")
        if k not in m:
            missing.append(s)
            continue
        out[s] = {"cik": int(m[k]["cik_str"]), "sec_ticker": k, "title": m[k]["title"]}
        if k in wk and wk[k] != out[s]["cik"]:
            mismatch.append((s, out[s]["cik"], wk[k]))
    return out, missing, mismatch


def submissions(cik, refresh):
    """All filings (recent + older pages reaching back to SINCE) as a list of dicts."""
    t = cached(SUBS / f"CIK{cik:010d}.json", f"https://data.sec.gov/submissions/CIK{cik:010d}.json", refresh)
    if t is None:
        return None, []
    j = json.loads(t)
    blocks = [j["filings"]["recent"]]
    pages = []
    for f in j["filings"].get("files", []):
        if f["filingTo"] >= SINCE:
            p = cached(SUBS / f["name"], f"https://data.sec.gov/submissions/{f['name']}", refresh)
            if p:
                blocks.append(json.loads(p))
                pages.append(f["name"])
    rows = []
    for b in blocks:
        n = len(b["accessionNumber"])
        for i in range(n):
            rows.append({k: b[k][i] for k in ("accessionNumber", "form", "items", "filingDate", "reportDate",
                                               "acceptanceDateTime", "primaryDocument") if k in b})
    earliest = min((r["filingDate"] for r in rows), default=None)
    return {"name": j.get("name"), "pages": pages, "earliest_filing": earliest, "n_all": len(rows)}, rows


# ---------------------------------------------------------------- filing index and text

def parse_index(html):
    acc = re.search(r'infoHead">Accepted</div>\s*<div class="info">([^<]+)<', html)
    docs = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        if len(cells) < 5:
            continue
        href = re.search(r'href="([^"]+)"', cells[2])
        if not href:
            continue
        typ = re.sub(r"<[^>]+>|&nbsp;", "", cells[3]).strip()
        docs.append({"seq": re.sub(r"<[^>]+>|&nbsp;", "", cells[0]).strip(), "type": typ,
                     "href": href.group(1).replace("/ix?doc=", "")})
    return {"accepted_et": acc.group(1).strip() if acc else None, "docs": docs}


def pick_doc(docs, primary):
    """EX-99.1 (or EX-99.01 / EX-99 / the lowest EX-99.x) press release; else the main document."""
    def ok(d):
        return d["href"].lower().endswith((".htm", ".html", ".txt"))
    ex = [d for d in docs if d["type"].upper().startswith("EX-99") and ok(d)]
    for want in ("EX-99.1", "EX-99.01", "EX-99"):
        for d in ex:
            if d["type"].upper() == want:
                return d, "exhibit"
    if ex:
        def num(d):
            m = re.search(r"EX-99\.?(\d+)", d["type"].upper())
            return int(m.group(1)) if m else 999
        return sorted(ex, key=num)[0], "exhibit"
    main = [d for d in docs if d["href"].rsplit("/", 1)[-1] == primary and ok(d)] or \
           [d for d in docs if d["type"].upper().startswith("8-K") and ok(d)]
    return (main[0], "main") if main else (None, None)


BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "article", "ul", "ol",
         "center", "blockquote", "pre", "hr"}


def items(f):
    return {x for x in (f.get("items") or "").split(",") if x}


def wants_text(f):
    """2.02 / 7.01 / 8.01, plus 8-Ks tagged only 2.01 and/or 9.01 (or nothing): filers sometimes tag an earnings
    release that way (AMAT May 2024 "2.01,9.01", PSX Jan 2024 "9.01")."""
    it = items(f)
    return bool(it & set(TEXT_ITEMS)) or it <= {"2.01", "9.01"}


PERIOD = r"(quarter|quarterly|fiscal|full[- ]year|year[- ]end|[1-4]Q|Q[1-4])"
EARN_TITLE = re.compile(r"\b(reports?|reported|announces?|announced|delivers?|posts?)\b[^\n]{0,120}?\b" + PERIOD +
                        r"\b[^\n]{0,80}?\b(results|earnings)\b|^[^\n]{0,30}\b" + PERIOD + r"\b[^\n]{0,40}?\b(results|earnings)\b",
                        re.I | re.M)
NOT_EARN = re.compile(r"\b(will (report|announce|release|host)|to (report|announce|release|host)|conference call|"
                      r"dividend|preliminary)\b", re.I)


def inferred_earnings(f):
    """Earnings release filed without item 2.02 (only for the 2.01/9.01-only group): a results headline in the first
    800 characters of the press release, not a call notice, dividend or preliminary result."""
    if items(f) & {"2.02", "7.01", "8.01"} or f.get("text_status") != "ok":
        return False
    head = (TEXT / f"{f['accessionNumber']}.txt").read_text(encoding="utf-8")[:800]
    m = EARN_TITLE.search(head)
    return bool(m) and not NOT_EARN.search(head[:m.end()])


def strip_header(t):
    """Drop the SGML <TYPE>/<SEQUENCE>/<FILENAME> lines that EDGAR keeps at the top of a served document."""
    lines = t.split("\n")
    if len(lines) >= 3 and re.fullmatch(r"[A-Za-z0-9\-./ ]{2,24}", lines[0]) and lines[1].strip().isdigit() \
            and re.search(r"\.(htm|html|txt)$", lines[2].strip(), re.I):
        t = "\n".join(lines[3:]).lstrip()
    return t


def html_to_text(raw):
    import lxml.html
    head = re.sub(r"^\s*<DOCUMENT>.*?<TEXT>\s*", "", raw[:5000], flags=re.S | re.I)
    raw = head + raw[5000:]
    if not re.search(r"<(html|body|div|p|table|font)\b", raw[:20000], re.I):
        t = re.sub(r"<[^>]+>", " ", raw)      # plain-text filing
    else:
        raw = re.sub(r"<\?xml[^>]*\?>", "", raw)
        doc = lxml.html.fromstring(raw.encode("utf-8", "ignore"), parser=lxml.html.HTMLParser(encoding="utf-8", recover=True))
        for el in list(doc.iter()):
            if not isinstance(el.tag, str):
                continue
            tag = el.tag.lower().split(":")[-1] if ":" in el.tag else el.tag.lower()
            style = (el.get("style") or "").replace(" ", "").lower()
            if tag in ("script", "style", "head", "title") or el.tag.lower() == "ix:header" or "display:none" in style:
                if el.getparent() is not None:
                    el.drop_tree()
        parts = []

        def walk(el):
            if not isinstance(el.tag, str):
                if el.tail:
                    parts.append(el.tail)
                return
            tag = el.tag.lower().split(":")[-1]
            if tag in BLOCK:
                parts.append("\n")
            if tag in ("td", "th"):
                parts.append(" \t ")
            if el.text:
                parts.append(el.text)
            for c in el:
                walk(c)
            if tag in BLOCK:
                parts.append("\n")
            if el.tail:
                parts.append(el.tail)
        walk(doc)
        t = "".join(parts)
    t = t.replace("\xa0", " ").replace("​", "")
    t = re.sub(r"&#\d+;|&[a-z]+;", " ", t)
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return strip_header(t.strip())


def fetch_text(f, idx=IDX, text=TEXT):
    """Index page -> chosen document -> text file. Returns a status dict (also cached in data/edgar/index/; live
    fetches pass data/edgar/live/ folders instead)."""
    acc, cik = f["accessionNumber"], f.get("cik_filer") or f["cik"]
    meta_path = idx / f"{acc}.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("status") == "ok" and (text / f"{acc}.txt").exists():
            return meta
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/"
    html = get(base + f"{acc}-index.htm")
    if html is None:
        meta = {"status": "no_index"}
    else:
        ix = parse_index(html)
        d, kind = pick_doc(ix["docs"], f["primaryDocument"])
        meta = {"accepted_et_index": ix["accepted_et"], "doc": d["href"] if d else None, "doc_type": d["type"] if d else None,
                "kind": kind, "status": "no_doc"}
        if d:
            url = "https://www.sec.gov" + d["href"] if d["href"].startswith("/") else base + d["href"]
            b = get(url, binary=True)
            if b is None:
                meta["status"] = "doc_fail"
            else:
                try:
                    raw = b.decode("utf-8")
                except UnicodeDecodeError:
                    raw = b.decode("cp1252", "replace")
                txt = html_to_text(raw)[:TEXT_CHARS]
                text.mkdir(parents=True, exist_ok=True)
                (text / f"{acc}.txt").write_text(txt, encoding="utf-8")
                meta.update(status="ok", chars=len(txt))
    idx.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta


# ---------------------------------------------------------------- main

def main(argv):
    refresh, do_text = "--refresh" in argv, "--no-text" not in argv
    OUT.mkdir(parents=True, exist_ok=True)
    cmap, missing, mismatch = cik_map(refresh)
    print(f"{len(cmap)} companies mapped, missing {missing}, CIK mismatches vs sp500.html {mismatch}", flush=True)
    (OUT / "cik_map.json").write_text(json.dumps(cmap, indent=1), encoding="utf-8")

    cal = trading_calendar()
    calset = set(cal)
    by_cik = {}
    for s, v in cmap.items():
        by_cik.setdefault(v["cik"], []).append(s)

    companies, filings = {}, []
    t0 = time.time()

    def one(cik):
        info, rows = submissions(cik, refresh)
        for p in PREDECESSORS.get(cik, []):
            # The predecessor's 8-Ks up to the successor's first 8-K; joint filings share an accession number.
            first = min((r["filingDate"] for r in rows if r["form"].startswith("8-K")), default="9999")
            pinfo, prows = submissions(p, refresh)
            have = {r["accessionNumber"] for r in rows}
            add = [dict(r, cik_filer=p) for r in prows if r["filingDate"] < first and r["accessionNumber"] not in have]
            rows = rows + add
            if info is not None:
                info.setdefault("predecessors", []).append({"cik": p, "name": pinfo and pinfo["name"], "until": first,
                                                            "added_filings": len(add)})
        return cik, (info, rows)
    with ThreadPoolExecutor(4) as ex:
        for i, (cik, (info, rows)) in enumerate(ex.map(one, sorted(by_cik))):
            for s in by_cik[cik]:
                companies[s] = {"cik": cik, **(info or {"error": "no submissions"})}
            seen = set()
            for r in rows:
                if r["accessionNumber"] in seen:
                    continue
                seen.add(r["accessionNumber"])
                r = dict(r, cik_filer=r.get("cik_filer", cik))
                if r["form"] in ("8-K", "8-K/A") and r["filingDate"] >= SINCE:
                    filings.append({"cik": cik, "symbols": "|".join(by_cik[cik]), **r})
            if i % 50 == 0:
                print(f"  submissions {i}/{len(by_cik)}  {time.time() - t0:.0f}s  requests {STATS['requests']}", flush=True)
    filings.sort(key=lambda f: f["acceptanceDateTime"])
    print(f"{len(filings)} 8-K/8-K/A filings since {SINCE}", flush=True)

    if do_text:
        todo = [f for f in filings if wants_text(f)]
        todo.sort(key=lambda f: (0 if "2.02" in f["items"].split(",") else 1, f["acceptanceDateTime"]))
        print(f"text: {len(todo)} filings with items {TEXT_ITEMS}", flush=True)
        metas = {}
        t1 = time.time()
        with ThreadPoolExecutor(4) as ex:
            for i, (f, m) in enumerate(zip(todo, ex.map(fetch_text, todo))):
                metas[f["accessionNumber"]] = m
                if i % 250 == 0:
                    print(f"  text {i}/{len(todo)}  {time.time() - t1:.0f}s  requests {STATS['requests']} retries {STATS['retries']}",
                          flush=True)
        for p in TEXT.glob("*.txt"):          # texts written before strip_header existed
            t = p.read_text(encoding="utf-8")
            c = strip_header(t)
            if c != t:
                p.write_text(c, encoding="utf-8")
        for f in filings:
            m = metas.get(f["accessionNumber"])
            if m:
                f["accepted_et_index"] = m.get("accepted_et_index")
                f["text_status"] = m.get("status")
                f["text_doc"] = m.get("doc")
                f["text_kind"] = m.get("kind")

    conv = calibrate(filings)
    for f in filings:
        c = conv.get(f["cik_filer"])
        if f.get("accepted_et_index"):
            et = datetime.strptime(f["accepted_et_index"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=ET)
            u, src = et.astimezone(UTC), "index"
        else:
            u = raw_to_utc(f["acceptanceDateTime"], c or "shifted")
            et, src = u.astimezone(ET), f"raw_{c or 'unverified'}"
        sess, first = classify(et, cal, calset)
        f.update(accepted_utc=u.strftime("%Y-%m-%dT%H:%M:%SZ"), accepted_et=et.strftime("%Y-%m-%d %H:%M:%S"),
                 time_source=src, raw_convention=c, session=sess, first_session=first)
    filings.sort(key=lambda f: f["accepted_utc"])
    for s, v in companies.items():
        v["raw_convention"] = conv.get(v["cik"])

    cols = ["symbols", "cik", "cik_filer", "accessionNumber", "form", "items", "filingDate", "reportDate", "acceptanceDateTime",
            "accepted_utc", "accepted_et", "accepted_et_index", "time_source", "raw_convention", "session", "first_session",
            "primaryDocument",
            "text_status", "text_kind", "text_doc"]
    with open(OUT / "filings.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(filings)

    earn = {}
    for s in cmap:
        earn[s] = [{"accession": f["accessionNumber"], "form": f["form"], "accepted_utc": f["accepted_utc"],
                    "accepted_et": f["accepted_et"], "session": f["session"], "first_session": f["first_session"],
                    "text": (TEXT / f"{f['accessionNumber']}.txt").exists(), "source": "item 2.02"}
                   for f in filings if s in f["symbols"].split("|") and "2.02" in items(f)]
        have = [datetime.fromisoformat(e["accepted_utc"][:19]) for e in earn[s]]
        extra = [f for f in filings if s in f["symbols"].split("|") and inferred_earnings(f)   # not a re-filing of a 2.02
                 and all(abs((datetime.fromisoformat(f["accepted_utc"][:19]) - h).days) > 10 for h in have)]
        for f in extra:
            earn[s].append({"accession": f["accessionNumber"], "form": f["form"], "accepted_utc": f["accepted_utc"],
                            "accepted_et": f["accepted_et"], "session": f["session"], "first_session": f["first_session"],
                            "text": True, "source": "text, items " + f["items"]})
        earn[s].sort(key=lambda e: e["accepted_utc"])
    (OUT / "earnings_dates.json").write_text(json.dumps({"generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                          "since": SINCE, "symbols": earn}, indent=0), encoding="utf-8")
    (OUT / "companies.json").write_text(json.dumps(companies, indent=1), encoding="utf-8")
    print(f"done: {STATS}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
