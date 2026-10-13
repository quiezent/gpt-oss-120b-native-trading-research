"""Check p1.py against Codex's saved controlled comparison (local fixtures in data/codex).

1. Initial prompt tokens: my rendering must equal Codex's saved tokens for all 12 cases.
2. Tool results: replaying all 36 trajectories' saved actions through my Host must reproduce every
   saved tool result byte for byte (with sell_cap_bug=True, Codex's behaviour) and the saved
   terminal NAV.
3. Continuations: prompt(n+1) = prompt(n) + sample + rendered tool result, checked on a NAV4
   training rollout.
"""

import json
import sys
from pathlib import Path

import p1
from nav_runner import render_prompt, tool_message_tokens, get_tok

FIX = Path(__file__).resolve().parent / "data" / "codex"


def main():
    bars = p1.load_bars("codex")
    tok = get_tok()
    bad = 0
    # 1. prompts
    for case in sorted((FIX / "controlled_prepared").glob("2026-*")):
        start, h = case.name.rsplit("-h", 1)
        host = p1.Host(bars, start, int(h))
        mine = render_prompt(host, start)
        theirs = json.loads((case / "initial_prompt_tokens.json").read_text())
        same = mine == theirs
        bad += not same
        if not same:
            k = next((i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b), min(len(mine), len(theirs)))
            print(f"PROMPT MISMATCH {case.name}: len {len(mine)} vs {len(theirs)}, first diff at {k}:",
                  repr(tok.decode(mine[max(0, k - 20):k + 20])), "||", repr(tok.decode(theirs[max(0, k - 20):k + 20])))
    print("prompts checked")
    # 2. tool results
    n = 0
    for f in sorted((FIX / "controlled_live").glob("*/*/comparison_outcome.json")):
        o = json.loads(f.read_text())
        ep = o["episode"]
        host = p1.Host(bars, ep["start_session"], ep["horizon_sessions"], sell_cap_bug=True)
        for a in o["public_actions"]:
            got = host.execute(a["action"]["arguments"])
            n += 1
            if p1.canonical(got) != p1.canonical(a["tool_result"]):
                bad += 1
                print("RESULT MISMATCH", f.parent.parent.name, f.parent.name, "step", a["step"])
                print("  mine  ", p1.canonical(got)[:600])
                print("  theirs", p1.canonical(a["tool_result"])[:600])
        host.terminal()
        nav = o["economic_outcome"].get("terminal_nav_usd") or o["economic_outcome"].get("terminal_marked_nav_usd")
        ret = (float(host.nav()) - 10000) / 100
        want = o["economic_outcome"].get("reward")
        if want is not None and abs(ret / 100 - float(want)) > 1e-9:
            bad += 1
            print("NAV MISMATCH", f.parent.parent.name, f.parent.name, ret, want)
    print(f"{n} tool results checked")
    # 3. continuation
    d = FIX / "nav4_training" / "b0g0r0"
    s1 = json.loads((d / "step-0001-model.raw").read_text())
    s2 = json.loads((d / "step-0002-model.raw").read_text())
    rec = json.loads((d / "step-0001-receipt.json").read_text())
    result = rec.get("tool_result") or rec.get("result")
    cont = s1["prompt_tokens"] + s1["sequences"][0]["tokens"] + tool_message_tokens(result)
    same = cont == s2["prompt_tokens"]
    bad += not same
    print("continuation", "OK" if same else "MISMATCH")
    print("ALL OK" if not bad else f"{bad} problems")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
