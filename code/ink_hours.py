"""Inkling endpoint usage by UTC hour, from a day's look records: the per-sleeve cost record (every endpoint request bills
to one shared Tinker session with no project, and billing lags; the Tinker Owner, 21 Nov), summed so billing can be
checked hour by hour. Read-only: it reads runs/live/<day>/*.json and prints; nothing is written.

    python ink_hours.py 2026-11-23 [--live runs/live] [--json]

A look record is an Inkling one when its model is (inkling.is_inkling). Its trace's reply entries (those with "tokens")
give uncached = prompt_tokens - cached_tokens, cached and sample = tokens; its no-reply entries (a "utc" and no "tokens":
retried, a 422 or overflow, a failed request after the re-ask, "look ended") are counted by kind, since they may bill
with no usage. Each entry goes to the UTC hour of its own "utc"; a reply entry without one (a record from before week7
80c31a7c) goes to its record's hour, and is counted as untimed. Cost is at the rate card (prompt $0.30, cached $0.03,
completion $1.00 a million tokens).
"""
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RATE = {"uncached": 0.30, "cached": 0.03, "sample": 1.00}        # dollars a million tokens (the Tinker Owner, 20 Nov)


def is_inkling(model):                    # inkling.is_inkling, without importing the loop's modules
    return str(model or "").split(":", 1)[0].startswith("thinkingmachines/Inkling")


def kind_of(e):
    err = str(e.get("error") or "")
    if err.startswith("look ended"):
        return "look_ended"
    if "after the re-ask" in err:
        return "after_reask"
    if err.startswith("MALFORMED"):
        return "malformed"
    if "retried" in err:
        return "retried"
    return "other"


def hour_of(stamp):
    """'2026-11-23T14:30:05Z' or an ISO time with an offset -> '2026-11-23 14' (UTC), or None."""
    try:
        t = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        return None
    return t.astimezone(timezone.utc).strftime("%Y-%m-%d %H")


def sums(day_dir):
    hours = defaultdict(lambda: defaultdict(int))
    sleeves = defaultdict(lambda: defaultdict(int))
    bad = []
    for p in sorted(Path(day_dir).glob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            bad.append(f"{p.name}: {type(e).__name__}")
            continue
        if not isinstance(rec, dict) or not is_inkling(rec.get("model")):
            continue
        rec_hour = hour_of(rec.get("time"))
        sl = rec.get("sleeve") or p.stem
        sleeves[sl]["looks"] += 1
        for e in rec.get("trace") or []:
            if not isinstance(e, dict):
                continue
            if "tokens" in e:
                h = hour_of(e.get("utc"))
                if h is None:
                    h = rec_hour
                    hours[h]["untimed"] += 1
                pt, ct, ca = int(e.get("prompt_tokens") or 0), int(e.get("tokens") or 0), int(e.get("cached_tokens") or 0)
                for tgt in (hours[h], sleeves[sl]):
                    tgt["replies"] += 1
                    tgt["uncached"] += pt - ca
                    tgt["cached"] += ca
                    tgt["sample"] += ct
            elif "utc" in e:
                h = hour_of(e["utc"]) or rec_hour
                k = kind_of(e)
                for tgt in (hours[h], sleeves[sl]):
                    tgt["no_reply"] += 1
                    tgt[f"no_reply_{k}"] += 1
    return hours, sleeves, bad


def cost(r):
    return sum(r.get(k, 0) * RATE[k] for k in RATE) / 1e6


def table(hours, sleeves, bad, day):
    out = [f"Inkling endpoint usage on {day} by UTC hour, from the look records (rate-card cost)", "",
           "| UTC hour | replies | uncached | cached | sample | no-reply | of which retried / 422-overflow / after re-ask / look ended | cost |",
           "|---|---|---|---|---|---|---|---|"]
    tot = defaultdict(int)
    for h in sorted(k for k in hours if k):
        r = hours[h]
        for k, v in r.items():
            tot[k] += v
        out.append(f"| {h}:00 | {r['replies']} | {r['uncached']:,} | {r['cached']:,} | {r['sample']:,} | {r['no_reply']} | "
                   f"{r['no_reply_retried']} / {r['no_reply_malformed']} / {r['no_reply_after_reask']} / {r['no_reply_look_ended']} | "
                   f"${cost(r):.4f} |")
    if None in hours:
        out.append(f"| (no time) | {hours[None]['replies']} | | | | {hours[None]['no_reply']} | | |")
        for k, v in hours[None].items():
            tot[k] += v
    out.append(f"| **day** | {tot['replies']} | {tot['uncached']:,} | {tot['cached']:,} | {tot['sample']:,} | {tot['no_reply']} | "
               f"{tot['no_reply_retried']} / {tot['no_reply_malformed']} / {tot['no_reply_after_reask']} / {tot['no_reply_look_ended']} | "
               f"${cost(tot):.4f} |")
    out += ["", "| sleeve | looks | replies | uncached | cached | sample | no-reply | cost |", "|---|---|---|---|---|---|---|---|"]
    for s in sorted(sleeves):
        r = sleeves[s]
        out.append(f"| {s} | {r['looks']} | {r['replies']} | {r['uncached']:,} | {r['cached']:,} | {r['sample']:,} | {r['no_reply']} | ${cost(r):.4f} |")
    if tot["untimed"]:
        out.append(f"\n{tot['untimed']} reply entries had no utc and were put in their record's hour.")
    if bad:
        out.append(f"\nUnreadable records: {', '.join(bad)}")
    return "\n".join(out)


def main(argv):
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    day = argv[0]
    live = Path(argv[argv.index("--live") + 1]) if "--live" in argv else ROOT / "runs" / "live"
    hours, sleeves, bad = sums(live / day)
    if "--json" in argv:
        print(json.dumps({"day": day, "hours": {h: dict(r) for h, r in hours.items() if h},
                          "sleeves": {s: dict(r) for s, r in sleeves.items()}, "unreadable": bad}, indent=1))
    else:
        print(table(hours, sleeves, bad, day))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
