"""agent.parse_any and trader.SEG on fixed Harmony turns, then on every gpt-oss turn stored in runs/live/2026-10-*/:
a turn that parsed before the 2 Nov fix parses the same, and the turns it rescues are the plain " json" header with a
valid body. Offline (the gpt-oss tokenizer from the local cache).

    python parse_test.py
"""
import json
import re
from pathlib import Path

import agent
import trader

ROOT = Path(__file__).resolve().parent
NAMES = {"submit", "quote", "news", "article", "bars", "stats", "scanner", "tape", "forecast"}
# trader.SEG before 2 Nov: the content type only as <|constrain|>json
OLD = re.compile(r"^(?: to=(?P<r1>[^\s<]+))?<\|channel\|>(?P<ch>analysis|commentary|final)(?: to=(?P<r2>[^\s<]+))?"
                 r"\s*(?:<\|constrain\|>(?P<con>\w+))?<\|message\|>(?P<body>.*?)(?P<term><\|end\|>|<\|call\|>|<\|return\|>)$", re.S)
THINK = "<|channel|>analysis<|message|>Check BA's filing first.<|end|><|start|>assistant"


def parse(text, seg=None):
    agent.SEG = seg or trader.SEG
    try:
        toks = list(trader.tokenizer().encode(text, add_special_tokens=False))
        return agent.parse_any(toks, "stop", NAMES)
    finally:
        agent.SEG = trader.SEG


def check(name, text, want):
    got = parse(text)
    ok = got[:2] == want[:2] and (len(want) < 3 or got[2] == want[2])
    print(("ok   " if ok else "FAIL ") + f"{name}: {got[0]} {got[1]} {str(got[2])[:60]}")
    return ok


ok = [check("constrain json", THINK + '<|channel|>commentary to=functions.news <|constrain|>json<|message|>{"symbol": "BA"}<|call|>',
            ("TOOL", "news", {"symbol": "BA"})),
      check("plain json (the fix)", THINK + '<|channel|>commentary to=functions.news json<|message|>{"symbol": "BA"}<|call|>',
            ("TOOL", "news", {"symbol": "BA"})),
      check("no content type", THINK + '<|channel|>commentary to=functions.quote<|message|>{"symbols": ["BA"]}<|call|>',
            ("TOOL", "quote", {"symbols": ["BA"]})),
      check("plain json, bad body", THINK + '<|channel|>commentary to=functions.news json<|message|>{symbol: BA}<|call|>',
            ("MALFORMED", None)),
      check("plain code (base-ml's dry look, 1 Nov)", THINK + '<|channel|>commentary to=functions.tape code<|message|>{}<|call|>',
            ("TOOL", "tape", {})),
      check("plain code, bad body", THINK + '<|channel|>commentary to=functions.news code<|message|>print(1)<|call|>',
            ("MALFORMED", None)),
      check("another plain word", THINK + '<|channel|>commentary to=functions.news xml<|message|>{"symbol": "BA"}<|call|>',
            ("MALFORMED", None)),
      check("json glued to the tool name", THINK + '<|channel|>commentary to=functions.newsjson<|message|>{"symbol": "BA"}<|call|>',
            ("MALFORMED", None)),
      check("forced submit (agent.SUBMIT_OPEN)", "<|channel|>commentary to=functions.submit <|constrain|>json<|message|>"
            '{"place": [], "cancel": [], "notes": "hold"}<|call|>', ("TOOL", "submit")),
      check("prose answer", THINK + "<|channel|>final<|message|>I would hold.<|return|>", ("FINAL", None)),
      check("a submit ending in <|return|>", THINK + '<|channel|>commentary to=functions.submit <|constrain|>json<|message|>'
            '{"place": [], "cancel": ["x"]}<|return|>', ("TOOL", "submit", {"place": [], "cancel": ["x"]})),
      check("a research call ending in <|return|>", THINK + '<|channel|>commentary to=functions.news json<|message|>{"symbol": "BA"}<|return|>',
            ("TOOL", "news", {"symbol": "BA"})),
      check("a bad body ending in <|return|>", THINK + '<|channel|>commentary to=functions.submit json<|message|>{bad<|return|>', ("MALFORMED", None)),
      check("an undeclared tool ending in <|return|>", THINK + '<|channel|>commentary to=functions.nope json<|message|>{}<|return|>', ("MALFORMED", None)),
      check("an answer that is the submit's JSON", THINK + '<|channel|>final<|message|>{"place": [{"symbol": "A"}], "cancel": []}<|return|>',
            ("TOOL", "submit", {"place": [{"symbol": "A"}], "cancel": []})),
      check("a short label, then one object in a fence", THINK + '<|channel|>final<|message|>**Submit**\n```json\n{"cancel": ["x"], "place": []}\n```<|return|>',
            ("TOOL", "submit", {"cancel": ["x"], "place": []})),
      check("words before the object stay prose", THINK + '<|channel|>final<|message|>If NVDA breaks 200, submit:\n{"place": [], "cancel": []}<|return|>', ("FINAL", None)),
      check("a template in an answer stays prose", THINK + '<|channel|>final<|message|>{"place": [{"symbol": "...", "quantity": "..."}], "cancel": []}<|return|>', ("FINAL", None)),
      check("a template in a submit call is malformed", THINK + '<|channel|>commentary to=functions.submit json<|message|>{"place": [], "cancel": ["tt-x", "..."], "notes": "..."}<|call|>', ("MALFORMED", None)),
      check("dots inside real text are fine", THINK + '<|channel|>commentary to=functions.submit json<|message|>{"place": [], "cancel": [], "notes": "wait... then sell"}<|call|>', ("TOOL", "submit")),
      check("two objects in an answer stay prose", THINK + '<|channel|>final<|message|>A: {"place": [], "cancel": []} B: {"place": [], "cancel": ["x"]}<|return|>', ("FINAL", None)),
      check("words after the object stay prose", THINK + '<|channel|>final<|message|>{"place": [], "cancel": []}\nOn reflection, no.<|return|>', ("FINAL", None)),
      check("a revision cut off after a draft stays prose", THINK + '<|channel|>final<|message|>{"place": [], "cancel": []}\nBetter: {"place": [{"sym<|return|>', ("FINAL", None)),
      check("an answer with other JSON stays prose", THINK + '<|channel|>commentary<|message|>{"name": "functions.tape", "arguments": {}}<|return|>', ("FINAL", None)),
      check("a place that isn't a list stays prose", THINK + '<|channel|>final<|message|>{"place": "AAPL"}<|return|>', ("FINAL", None)),
      check("a cancel that isn't a list stays prose", THINK + '<|channel|>final<|message|>{"place": [], "cancel": null}<|return|>', ("FINAL", None)),
      check("no place in an answer stays prose", THINK + '<|channel|>final<|message|>{"new orders": [{"symbol": "A"}], "cancel": []}<|return|>', ("FINAL", None)),
      check("deep nesting in an answer stays prose", THINK + '<|channel|>final<|message|>{"place":' + "[" * 2000 + "<|return|>", ("FINAL", None)),
      check("deep nesting in a call is malformed", THINK + '<|channel|>commentary to=functions.submit json<|message|>{"place":' + "[" * 2000 + "<|call|>", ("MALFORMED", None)),
      check("orders under another key in a submit call", THINK + '<|channel|>commentary to=functions.submit json<|message|>{"new_orders": [], "cancel": []}<|call|>', ("MALFORMED", None)),
      check("a submit call that isn't an object", THINK + '<|channel|>commentary to=functions.submit json<|message|>[1, 2]<|return|>', ("MALFORMED", None)),
      check("JSON in analysis is not a decision", '<|channel|>analysis<|message|>Draft {"place": [], "cancel": []}<|return|>', ("FINAL", None)),
      check("call before the end", '<|channel|>commentary to=functions.news json<|message|>{"symbol": "BA"}<|end|>'
            '<|start|>assistant<|channel|>final<|message|>x<|return|>', ("MALFORMED", None))]

# Every stored gpt-oss turn (look files keep the decoded turn as "said" since week 3, and malformed ones since week 2)
same = changed = rescued = still_bad = cut = read = prose = 0
bad_rescues, moved = [], []
for f in sorted(ROOT.glob("runs/live/2026-1[01]-*/*-[0-9][0-9][0-9][0-9].json")):
    if f.parent.name > "2026-11-02":                        # a fixed set of turns: all before this rule went live
        continue
    lk = json.loads(f.read_text(encoding="utf-8"))
    if lk.get("dry_run") or str(lk.get("model", "")).startswith("claude"):
        continue
    for t in lk.get("trace") or []:
        if t.get("said") is None:
            continue
        if not t["said"].endswith(("<|call|>", "<|return|>")):     # week 2 kept 4,000 characters of a turn
            cut += 1
            continue
        old, new = parse(t["said"], OLD), parse(t["said"])
        if t.get("tool") and new[:2] != ("TOOL", t["tool"]):
            moved.append(f"{f.parent.name}/{f.stem}")
        if str(t.get("error", "")).startswith("FINAL"):     # stored as prose, and rewritten; a submit now if it holds one
            read += new[:2] == ("TOOL", "submit")
            prose += new[0] == "FINAL"
        elif old[0] == "TOOL" or old[0] == "FINAL":
            same += old == new
            changed += old != new
        elif new[0] == "TOOL":
            rescued += 1
            last = t["said"].split("<|start|>assistant")[-1]
            if not re.search(r" to=functions\.\w+ (?:json|code)<\|message\|>", last):
                bad_rescues.append(f"{f.parent.name}/{f.name}")
        else:
            still_bad += 1
print(f"stored turns: {same} parse as before, {changed} changed, {rescued} rescued, {still_bad} still malformed, "
      f"{cut} stored cut (not tested); prose turns: {read} now read as the submit, {prose} still prose; "
      f"stored calls no longer accepted: {moved}")
# the 7 prose turns that are the decision; the tape call in commentary and the answer with a closing line stay prose;
# the only stored calls refused are rl3's two with orders under another key, and base-ml's template of 28 Oct
ok.append(changed == 0 and not bad_rescues and rescued > 0 and read == 7 and prose == 2
          and moved == ["2026-10-20/rl3-ml-1230", "2026-10-21/rl3-ml-1100", "2026-10-28/base-ml-0930"])
if bad_rescues:
    print("FAIL rescued without the plain json header:", bad_rescues[:5])
print("ALL OK" if all(ok) else "FAILED")
