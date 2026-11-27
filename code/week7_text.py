"""The Week 7 text as the sleeves would read it, for the Trading Model Researcher (20 Nov 2026): the system text of a
30- and a 15-minute text-B sleeve on a full day (Mon 23 Nov) and a half day (Fri 27 Nov), with and without the
NEXT_LOOK sentence, and recorded states of 19 Nov with the range line where live.py puts it (after the open orders).
No Gateway, no model, $0.

    python week7_text.py                 # writes runs/week7_text/*.txt and prints a summary
"""

import json
from datetime import date
from pathlib import Path

import live
import trader

ROOT = Path(__file__).resolve().parent
MAIN = Path(r"C:\Claude\Trader")
OUT = ROOT / "runs" / "week7_text"
DAYS = {"full": date(2026, 11, 23), "half": date(2026, 11, 27)}
STATES = ["runs/live/2026-11-19/base-90k-b-1530.json", "runs/live/2026-11-19/base-90k-ml-1200.json",
          "runs/live/2026-11-19/cand2-1530.json", "runs/live/2026-11-19/base-ml-1000.json"]


def system_for(meta, day, cadence, next_look):
    return trader.live_system(meta["start_cash"], day, ml=bool(meta.get("ml")), ml_tool=meta.get("ml_tool"),
                              kelly=bool(meta.get("kelly_prompt")), max_gross=trader.max_gross_of(meta),
                              own_limit=trader.own_limit_of(meta), cadence=cadence,
                              close=live.close_time(day) if trader.text_of(meta) == "B" else None, next_look=next_look)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"start_cash": 30000.0, "model": "base", "ml": False, "max_gross": None, "own_limit": 3.0}
    today = trader.live_system(meta["start_cash"], DAYS["full"], max_gross=None, own_limit=3.0)   # the text as live sends it now
    for name, day in DAYS.items():
        for cadence in (30, 15):
            for nl in (False, True):
                text = system_for(meta, day, cadence, nl)
                f = OUT / f"system_{cadence}min_{name}_day{'_next_look' if nl else ''}.txt"
                f.write_text(text, encoding="utf-8")
                print(f"{f.name}: {len(text)} chars" + ("; the same as today's text" if text == today else ""))
    for rel in STATES:
        p = MAIN / rel
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        st = rec["state"]
        line = trader.book_range(st, rec["time"][:10].replace("-", ""))
        text = rec["state_text"]
        cut = text.find("\nSince your last decision")
        out = text[:cut] + ("\n" + line if line else "") + text[cut:] if cut >= 0 else text
        f = OUT / f"state_{Path(rel).stem}_range_line.txt"
        f.write_text(out, encoding="utf-8")
        print(f"{f.name}: {len(st['positions'])} positions; range line: {line or '(none: nothing held, or nothing priced)'}")


if __name__ == "__main__":
    main()
