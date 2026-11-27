"""Smoke draws of the Inkling look on recorded live states (20 Nov 2026): no Gateway, no orders.

Each record (runs/live/<date>/<sleeve>-HHMM.json) gives the sleeve's state text as the model read it; the system text is
rebuilt with trader.live_system from runs/sleeves.json and the record's settings, as live.py builds it. Research calls are
answered from the record's own trace: the stored result of the same tool with the same arguments, else the next unused
stored result of that tool under a first line naming the arguments it answers, else one line saying none was recorded.
Stored framework lines (limit lines, "context nearly full" lines) are never served as data, and a stored result's own
counter line ("(research call k of N)", "(order check k of 3)") and cut line are taken off, so the loop's own counter is
the only one the model reads (the Researcher's read, 20 Nov). A check_orders call gets the record's stored check result
the same way, else a line that no check ran (not counted). The results serve form and size only: the decision is not
scored, though each output file carries the record's state and state text for a check walk.

    python inkling_smoke.py --dry                                   # $0: the first request's size, nothing sent
    python inkling_smoke.py --records runs/live/2026-11-19/base-90k-b-0900.json --seed 1
    python inkling_smoke.py --model thinkingmachines/Inkling-2-Small:peft:262144 --tag peft

The default records are base-90k-b and base-90k-ml at 09:00 and 12:00 on Thu 19 Nov. Output: runs/inkling_smoke/<tag>/
<record name>.json (record path, state, state text, call, trace, cost, any error) and a summary line per look. A tag is a
plain name (letters, digits, - and _); a tag whose folder already holds output is refused unless --overwrite is given. A
look that raises is written with its trace and cost so far (paid tokens are never lost), and the run goes on.
"""

import argparse
import json
import re
import sys
import traceback
from datetime import date
from pathlib import Path

import agent
import inkling
import ml_tools
import tools
import trader

ROOT = Path(__file__).resolve().parent
MAIN = Path(r"C:\Claude\Trader")          # the live ledgers and records (a worktree has none of its own)
DEFAULT = [f"runs/live/2026-11-19/{s}-{t}.json" for s in ("base-90k-b", "base-90k-ml") for t in ("0900", "1200")]
NONE_LINE = "(smoke replay: no recorded result for this call)"
FRAMEWORK = ("research limit reached", "context nearly full (", "check limit reached")
COUNTER = re.compile(r"^\((research call \d+ of \d+|order check \d+ of \d+)\)$")


def find(rel):
    for base in (Path.cwd(), ROOT, MAIN):
        p = base / rel
        if p.exists():
            return p
    raise FileNotFoundError(rel)


def clean(result):
    """A stored result as the tool wrote it: its trailing counter line and the loop's cut line taken off."""
    lines = str(result).split("\n")
    while lines and (COUNTER.match(lines[-1].strip()) or lines[-1].strip() == agent.CUT_LINE):
        lines.pop()
    return "\n".join(lines)


def is_framework(result):
    r = str(result)
    return r.startswith(FRAMEWORK) or r == getattr(__import__("checks"), "CAP_TEXT", None)


class Replay:
    """Recorded results of one look, served by tool and arguments."""

    def __init__(self, trace):
        self.by = {}
        for e in trace:
            if e.get("tool") and e["tool"] != "submit" and "result" in e and not is_framework(e["result"]):
                self.by.setdefault(e["tool"], []).append([e.get("args"), clean(e["result"]), False])

    def answer(self, name, args):
        rows = self.by.get(name) or []
        for r in rows:
            if r[0] == args and not r[2]:
                r[2] = True
                return r[1], True
        for r in rows:
            if not r[2]:
                r[2] = True
                return f"(smoke replay: the recorded result for {json.dumps(r[0], sort_keys=True)})\n{r[1]}", True
        return NONE_LINE, False


def setup(rec):
    sleeves = json.loads(find("runs/sleeves.json").read_text(encoding="utf-8"))
    meta = dict(sleeves[rec["sleeve"]])
    for k in ("max_gross", "own_limit", "kelly_prompt", "max_calls"):   # ml_tool: sleeves.json's key (the record holds its label)
        if k in rec:
            meta[k] = rec[k]
    ml = bool(meta.get("ml"))
    today = date.fromisoformat(rec["time"][:10])
    system = trader.live_system(meta["start_cash"], today, ml=ml, ml_tool=meta.get("ml_tool"),
                                kelly=bool(meta.get("kelly_prompt")), max_gross=trader.max_gross_of(meta),
                                own_limit=trader.own_limit_of(meta))
    specs = list(tools.SPECS) + ([ml_tools.module(meta.get("ml_tool")).SPEC] if ml else [])
    return meta, system, specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="*", default=DEFAULT)
    ap.add_argument("--model", default=inkling.MODEL)
    ap.add_argument("--effort", default=inkling.EFFORT)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--context", type=int, default=inkling.CONTEXT)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--overwrite", action="store_true", help="write into a tag folder that already holds output")
    ap.add_argument("--dry", action="store_true", help="$0: build the first request and print its size; send nothing")
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", a.tag):
        raise SystemExit(f"--tag {a.tag!r}: a plain name (letters, digits, - and _)")
    eff = float(a.effort) if str(a.effort).replace(".", "", 1).isdigit() else a.effort
    out_dir = ROOT / "runs" / "inkling_smoke" / a.tag
    if not a.dry and out_dir.exists() and any(out_dir.iterdir()) and not a.overwrite:
        raise SystemExit(f"{out_dir} already holds output; pick another --tag or pass --overwrite")
    for rel in a.records:
        path = find(rel)
        rec = json.loads(path.read_text(encoding="utf-8"))
        meta, system, specs = setup(rec)
        rp = Replay(rec.get("trace") or [])

        def run_tool(gw, name, args, **_):
            return rp.answer(name, args)[0]

        def checker(args):
            text, found = rp.answer("check_orders", args)
            return (text, True, None) if found else ("(smoke replay: no order check ran)", False, None)
        checker.own_limit = trader.own_limit_of(meta)
        submit_tool = trader.TOOL_LIVE_PROBS if trader.text_of(meta) == "A" else trader.TOOL_LIVE
        kw = dict(effort=eff, seed=a.seed, ml_tool=meta.get("ml_tool"), submit_tool=submit_tool,
                  sleeve_value=(rec.get("state") or {}).get("value"), look_key=(rec["sleeve"], rec["time"]),
                  checker=checker, final_line=meta.get("final_line"), max_calls=trader.max_calls_of(meta),
                  model=a.model, context=a.context, run_tool=run_tool)
        if a.dry:
            sent = []

            def send(body):
                sent.append(body)
                raise inkling.HttpError(400, "dry run: not sent")
            try:
                inkling.look(None, system, rec["state_text"], specs, send=send, **kw)
            except inkling.HttpError:
                pass
            b = sent[0]
            n_sys, n_state = inkling.est(system), inkling.est(rec["state_text"])
            n_tools = inkling.est(json.dumps(b["tools"], separators=(",", ":")))
            print(f"{rel}: model {b['model']}, effort {b['reasoning_effort']}, {len(b['tools'])} tools, tool_choice "
                  f"{b['tool_choice']}, max_completion_tokens {b['max_completion_tokens']}; estimated first prompt "
                  f"{n_sys + n_state + n_tools + inkling.FIRST_PAD} tokens (system {n_sys}, state {n_state}, tools {n_tools}, "
                  f"framing {inkling.FIRST_PAD}; o200k x {inkling.SLACK} + {inkling.FRAME} a message) of {a.context}; "
                  f"{sum(len(v) for v in rp.by.values())} recorded results to serve")
            continue
        error = None
        try:
            call, trace, prefill, sampled = inkling.look(None, system, rec["state_text"], specs, **kw)
        except Exception as e:            # noqa: BLE001 - the paid trace and cost so far are written, and the run goes on
            call, trace = None, list(getattr(e, "inkling_trace", []))
            prefill, sampled = getattr(e, "inkling_cost", (0, 0))
            error = traceback.format_exc()[-1500:]
        out_dir.mkdir(parents=True, exist_ok=True)
        name = path.stem
        (out_dir / f"{name}.json").write_text(json.dumps({
            "record": str(path), "sleeve": rec["sleeve"], "time": rec["time"], "state": rec.get("state"),
            "state_text": rec["state_text"], "model": a.model, "effort": eff, "seed": a.seed, "context": a.context,
            "system_sha": __import__("hashlib").sha256(system.encode()).hexdigest()[:16],
            "tools_sha": inkling.words_sha(specs + [__import__("checks").CHECK_SPEC, submit_tool]),
            "call": call, "cost": {"prefill_tokens": prefill, "sampled_tokens": sampled}, "error": error, "trace": trace},
            indent=1, default=float), encoding="utf-8")
        st = next((e["inkling"] for e in reversed(trace) if isinstance(e, dict) and isinstance(e.get("inkling"), dict)), {})
        replies = [e for e in trace if "prompt_tokens" in e]
        gaps = [e["prompt_tokens"] - e["est_prompt"] for e in replies if e["prompt_tokens"]]
        print(f"{name}: {'ERROR (written) ' if error else ''}{'decision' if call else 'NO DECISION'}"
              f" ({len((call or {}).get('place') or [])} orders, {len((call or {}).get('cancel') or [])} cancels);"
              f" {st.get('requests')} requests ({st.get('attempts')} attempts, {st.get('lost_attempts')} lost),"
              f" {sum(1 for e in trace if e.get('tool') and e['tool'] != 'submit')} tool calls,"
              f" forced {st.get('forced_turns')} (plain {st.get('forced_plain')}, reasoned {st.get('forced_reasoned')}),"
              f" 422s {st.get('http_422')}; prefill {prefill}, cached {st.get('cached_tokens')}, sampled {sampled},"
              f" longest reply {max([e.get('tokens') or 0 for e in replies] or [0])};"
              f" server minus estimate {min(gaps) if gaps else '-'} to {max(gaps) if gaps else '-'} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
