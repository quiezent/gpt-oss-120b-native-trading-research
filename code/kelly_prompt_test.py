"""Week 5 (Clayton, 9 Nov): the Kelly sizing paragraph and ML3's registration. Offline.
- Off (no "kelly_prompt"), live_system is byte-identical to the committed code before the change, for every ml/news case.
- On, the paragraph appears once, after the target pace and before the order types, in plain ASCII (the Claude CLI gets
  the system prompt as an argument).
- ML3 is a registered version, so its sleeves get looks and their forecasts come from forecast3."""
import importlib.util
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import ml_tools
import trader

ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(("ok   " if cond else "FAIL ") + name)


old_src = subprocess.run(["git", "show", "b1ecd35:trader.py"], capture_output=True,   # the last commit before the paragraph
                         text=True, encoding="utf-8", cwd=Path(__file__).resolve().parent).stdout
assert "def live_system" in old_src
tmp = Path(tempfile.mkdtemp()) / "trader_before.py"
tmp.write_text(old_src, encoding="utf-8")
spec = importlib.util.spec_from_file_location("trader_before", tmp)
before = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(Path(__file__).resolve().parent))
spec.loader.exec_module(before)

day = date(2026, 11, 9)
same = all(trader.live_system(size, day, ml=ml, news=news) == before.live_system(size, day, ml=ml, news=news)
           for size in (30000, 100000) for ml in (False, True) for news in (False, True))
check("off: byte-identical to the code before the change (8 cases)", same)

on = trader.live_system(30000, day, ml=True, kelly=True)
off = trader.live_system(30000, day, ml=True)
check("on: the paragraph once", on.count(trader.KELLY) == 1 and trader.KELLY not in off)
check("on: after the pace, before the order types",
      on.index("Missing an opportunity is as much an error") < on.index("Sizing by the Kelly rule") < on.index("Order types:"))
check("on: removing the paragraph gives the off text", on.replace("\n\n" + trader.KELLY, "") == off)
check("on: plain ASCII", all(ord(c) < 128 for c in on))
check("on, no-ML twin: same paragraph", trader.KELLY in trader.live_system(30000, day, ml=False, kelly=True))

import forecast3  # noqa: E402 - the pin follows the module's own version (ML3.0, then ML3.1 on 15 Nov 2026)
check("ML3 registered: forecast3", ml_tools.module("ML3").__name__ == "forecast3"
      and ml_tools.module("ML3").VERSION == forecast3.VERSION and forecast3.VERSION.startswith("ML3."))
check("ML3 sleeves aren't 'unknown' (they get looks)", ml_tools.unknown({"x": {"ml": True, "ml_tool": "ML3"}}) == [])
check("ML2 still the default", ml_tools.module(None).__name__ == "forecast" and ml_tools.DEFAULT == "ML2")
check("in_use: ML2 first, then ML3", ml_tools.in_use({"a": {"ml": True}, "b": {"ml": True, "ml_tool": "ML3"}}) == ["ML2", "ML3"])
check("ML2's guarded files unchanged", ml_tools.changed_files() == [])

print("ALL OK" if ok else "FAILURES")
