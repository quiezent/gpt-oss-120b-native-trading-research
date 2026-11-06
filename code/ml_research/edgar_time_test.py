"""edgar_fetch.fetch_convention / accepted_utc, the live 8-K times of events.live and livenews: on fixed rows, then on every
submissions JSON live cached in data/edgar/live/<day>/ (17 Oct 2026 on), against the filing index pages live fetched when
a model opened an 8-K's article (data/edgar/live/index/, EDGAR's own 'Accepted' time). Those pages are the only truth in
hand, so the replay measures the fetch's vote on the filings models opened. A filing's first look has no such page: the
vote alone decides it, and it can't see a fresh row written differently from the rest of its JSON (printed below).
Offline; the counts grow with the shared live cache.

    python ml_research/edgar_time_test.py
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml_research import edgar_fetch as ef  # noqa: E402

LIVE = ROOT / "data" / "edgar" / "live"
bad = 0


def check(name, ok, detail=""):
    global bad
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    bad += not ok


def et(u):
    return u.astimezone(ef.ET).strftime("%Y-%m-%d %H:%M")


# fixed rows: a fetch whose archived filing (true 16:04 ET = 20:04 UTC) is written as true UTC, and one written shifted
known = ef._known()
saved = dict(known)
known.clear()
known["A-1"] = ef._index_utc("2026-10-20 16:04:00")
for form, raw_old in (("utc", "2026-10-20T20:04:00.000Z"), ("shifted", "2026-10-21T00:04:00.000Z")):
    new = "2026-10-28T20:04:00.000Z" if form == "utc" else "2026-10-29T00:04:00.000Z"
    rc = {"accessionNumber": ["A-2", "A-1"], "acceptanceDateTime": [new, raw_old]}
    conv = ef.fetch_convention(rc)
    got = et(ef.accepted_utc(rc, 0, conv))
    check(f"fetch written {form}", conv == form and got == "2026-10-28 16:04", f"convention {conv}, new filing {got} ET")
check("no known filing: no vote", ef.fetch_convention({"accessionNumber": ["X"], "acceptanceDateTime": ["2026-10-28T20:04:00.000Z"]}) is None)
rc = {"accessionNumber": ["A-1"], "acceptanceDateTime": ["2026-10-21T00:04:00.000Z"]}
check("an index time beats the field", et(ef.accepted_utc(rc, 0, "utc")) == "2026-10-20 16:04", et(ef.accepted_utc(rc, 0, "utc")))
known.clear()
known.update(saved)

# every cached live fetch: the new filings a model opened, under the fetch's vote and under companies.json's per-company
# convention (the code before this fix), each against the index page
comp = json.loads((ROOT / "data" / "edgar" / "companies.json").read_text(encoding="utf-8"))
conv_cik = {int(v["cik"]): v.get("raw_convention") for v in comp.values()}
score, uniq, misses, close, votes = collections.Counter(), collections.defaultdict(set), [], {}, collections.Counter()
for p in sorted(LIVE.glob("2026*/CIK*.json")):
    day = p.parent.name
    j = json.loads(p.read_text(encoding="utf-8"))
    j = j.get("json", j)
    rc = j.get("filings", {}).get("recent", {})
    vote = ef.fetch_convention(rc)
    era = "from 26 Oct" if day >= "20261026" else "before 26 Oct"
    if era == "from 26 Oct":
        votes[vote] += 1
    for k, acc in enumerate(rc.get("accessionNumber", [])):
        if acc in known:
            continue
        u = ef._live_index(acc)
        if u is None:
            continue
        raw = rc["acceptanceDateTime"][k]
        conv = conv_cik.get(int(p.stem[3:])) or "shifted"
        new, old = ef.raw_to_utc(raw, vote or "shifted"), ef.raw_to_utc(raw, conv)
        score[(era, "n")] += 1
        score[(era, "vote")] += new == u
        score[(era, "per-company")] += old == u
        uniq[era].add(acc)
        if new != u:
            misses.append((day, acc, raw, et(u)))
        hhmm = u.astimezone(ef.ET).strftime("%H:%M")
        if era == "from 26 Oct" and conv == "shifted" and "2.02" in (rc["items"][k] or "") and hhmm >= "16:00":
            close[acc] = (et(old), et(new), et(u))
for era in ("before 26 Oct", "from 26 Oct"):
    print(f"     {era}: {score[(era, 'n')]} fetch-filing pairs ({len(uniq[era])} filings) with an index page; right under the "
          f"fetch's vote {score[(era, 'vote')]}, under the per-company convention {score[(era, 'per-company')]}")
check("the vote, from 26 Oct", score[("from 26 Oct", "vote")] == score[("from 26 Oct", "n")] > 0,
      f"{score[('from 26 Oct', 'vote')]} of {score[('from 26 Oct', 'n')]}")
check("every fetch from 26 Oct votes utc", set(votes) == {"utc"}, dict(votes))
check("the vote beats the per-company convention, from 26 Oct", score[("from 26 Oct", "vote")] > score[("from 26 Oct", "per-company")])
print("     vote misses, all before 26 Oct, fresh rows in true UTC inside a shifted JSON (the old code missed them too):", misses)

# after-close earnings 8-Ks of the "shifted" companies with an index page: before the close under the old code, after it now
check("after-close 2.02s with an index page", bool(close) and all(o[11:] < "16:00" and n == u and n[11:] >= "16:00" for o, n, u in close.values()),
      f"{len(close)} filings (old, new, index): {close}")
print("ALL OK" if not bad else f"{bad} FAILED")
