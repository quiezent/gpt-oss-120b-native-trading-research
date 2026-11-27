"""Offline checks of ink_hours.py on made-up look records in a temporary folder ($0; nothing under runs/ is read or written).

    python ink_hours_test.py
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import ink_hours  # noqa: E402
import inkling  # noqa: E402

ok = []


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail and not good else ""))


INK = "thinkingmachines/Inkling-2-Small"
day = Path(tempfile.mkdtemp(prefix="ink_hours_")) / "2026-11-23"
day.mkdir(parents=True)
reply = lambda pt, ca, ct, utc: {"step": 0, "tokens": ct, "prompt_tokens": pt, "cached_tokens": ca, "utc": utc, "response_id": "r"}
recs = {
    "ink2-30k-0930.json": {"time": "2026-11-23T09:30:00-05:00", "sleeve": "ink2-30k", "model": INK, "trace": [
        reply(10000, 8000, 500, "2026-11-23T14:30:05Z"), reply(11000, 9000, 600, "2026-11-23T14:31:00Z"),
        {"step": 2, "error": "request failed (HTTP 503); retried", "utc": "2026-11-23T14:31:40Z"},
        reply(12000, 10000, 700, "2026-11-23T14:59:59Z")]},
    "ink2-90k-ml-0930.json": {"time": "2026-11-23T09:30:00-05:00", "sleeve": "ink2-90k-ml", "model": INK, "trace": [
        reply(20000, 0, 1000, "2026-11-23T15:00:00Z"),
        {"step": 1, "error": "MALFORMED: HTTP 422", "http": 422, "utc": "2026-11-23T15:00:30Z"},
        {"step": 2, "error": "request failed after the re-ask: TimeoutError: x", "utc": "2026-11-23T15:01:00Z"},
        {"error": "look ended: HttpError: HTTP 503", "utc": "2026-11-23T15:01:10Z", "inkling": {"attempts": 4}},
        {"error": "Traceback ..."}]},
    "ink2-30k-ml-1000.json": {"time": "2026-11-23T10:00:00-05:00", "sleeve": "ink2-30k-ml", "model": INK, "trace": [
        {"step": 0, "tokens": 300, "prompt_tokens": 5000}]},           # a record from before 80c31a7c: no utc, no cached
    "base-30k-0930.json": {"time": "2026-11-23T09:30:00-05:00", "sleeve": "base-30k", "model": "base", "trace": [
        {"step": 0, "tokens": 99999, "prompt_tokens": 99999, "utc": "2026-11-23T14:30:00Z"}]},
    "broken-0930.json": None,
}
for n, r in recs.items():
    (day / n).write_text("{not json" if r is None else json.dumps(r), encoding="utf-8")

hours, sleeves, bad = ink_hours.sums(day)
h14, h15 = hours["2026-11-23 14"], hours["2026-11-23 15"]
check("is_inkling as inkling.is_inkling", all(ink_hours.is_inkling(m) == inkling.is_inkling(m) for m in
      (INK, INK + ":peft:131072", "base", None, "openai/gpt-oss-120b", "thinkingmachines/Inkling-3")))
check("14 UTC: three replies, uncached = prompt - cached, cached, sample",
      (h14["replies"], h14["uncached"], h14["cached"], h14["sample"]) == (3, 6000, 27000, 1800), dict(h14))
check("14 UTC: the retried attempt counted as a no-reply entry", h14["no_reply"] == 1 and h14["no_reply_retried"] == 1, dict(h14))
check("15 UTC (14:59:59 stays in 14): two replies (one untimed), three no-reply entries by kind; live's traceback "
      "entry (no utc) not counted",
      h15["replies"] == 2 and h15["uncached"] == 25000 and h15["sample"] == 1300 and h15["no_reply"] == 3 and h15["no_reply_malformed"] == 1
      and h15["no_reply_after_reask"] == 1 and h15["no_reply_look_ended"] == 1, dict(h15))
h_rec = hours["2026-11-23 15"]   # the untimed reply's record is 10:00 ET = 15:00 UTC
check("a reply with no utc goes to its record's UTC hour, counted as untimed",
      h_rec["untimed"] == 1 and sleeves["ink2-30k-ml"]["sample"] == 300 and sleeves["ink2-30k-ml"]["uncached"] == 5000, dict(h_rec))
check("a gpt-oss record is not counted", "base-30k" not in sleeves and all(r["sample"] < 99999 for r in hours.values()))
check("an unreadable record is reported, not counted", bad == ["broken-0930.json: JSONDecodeError"], bad)
check("per sleeve: looks and replies", sleeves["ink2-30k"]["looks"] == 1 and sleeves["ink2-30k"]["replies"] == 3
      and sleeves["ink2-90k-ml"]["no_reply"] == 3, {k: dict(v) for k, v in sleeves.items()})
check("cost at the rate card", abs(ink_hours.cost({"uncached": 1_000_000, "cached": 1_000_000, "sample": 1_000_000}) - 1.33) < 1e-9)
txt = ink_hours.table(hours, sleeves, bad, "2026-11-23")
check("the table has each hour and the day's total", "| 2026-11-23 14:00 | 3 |" in txt and "| 2026-11-23 15:00 | 2 |" in txt
      and "| **day** | 5 |" in txt and "1 reply entries had no utc" in txt, txt)
check("an empty day prints a table with zeros, no error", "| **day** | 0 |" in ink_hours.table(*ink_hours.sums(day.parent / "none"), "x"))

print(f"\nink_hours_test.py: {'ALL OK' if all(ok) else f'{ok.count(False)} FAILED'}")
sys.exit(0 if all(ok) else 1)
