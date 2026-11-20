"""Two parser fixes of 15 Nov 2026, replayed on week 5's recorded looks (runs/live/2026-11-09 to -13, read only):
agent.unwrap_args (a call written as {"name": <tool>, "arguments": {...}} runs with the inner arguments, research calls and
submit alike) and tools.article_ref (an article id with or without its provider prefix asks for the same article).
Offline: the gpt-oss tokenizer from the local cache, a fake gateway and a fake livenews; nothing is written.

    python parser_fixes_test.py
"""
import glob
import json
import sys
import types
from pathlib import Path

import agent
import forecast
import tools
import trader

ROOT = Path(__file__).resolve().parent
DAYS = ["2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13"]
NAMES = {"submit", "quote", "news", "article", "bars", "stats", "scanner", "tape", "forecast"}
ok = []


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail else ""))


def is_env(a):
    return isinstance(a, dict) and set(a) == {"name", "arguments"}


def parse_said(said):
    return agent.parse_any(list(trader.tokenizer().encode(said, add_special_tokens=False)), "stop", NAMES)


looks = []
for d in DAYS:
    for f in sorted(glob.glob(str(ROOT / "runs" / "live" / d / "*.json"))):
        try:
            looks.append((Path(f), json.loads(Path(f).read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            pass
gpt = [(f, L) for f, L in looks if not L.get("dry_run") and (str(L.get("model")) == "base" or str(L.get("model")).startswith("tinker://"))]
print(f"{len(looks)} looks read, {len(gpt)} non-dry gpt-oss looks")

# ---- no tool's parameters are an envelope's keys, so a real argument object is never unwrapped
props = {s["name"]: set(s["parameters"]["properties"]) for s in tools.SPECS + [trader.tool_live(), trader.tool_live(True), forecast.SPEC]}
check("no tool has both parameters 'name' and 'arguments'", not any({"name", "arguments"} <= p for p in props.values()),
      ", ".join(f"{k}: {sorted(v)}" for k, v in props.items() if "name" in v or "arguments" in v) or "none has either")

# ---- every recorded tool call through unwrap_args: envelopes give their inner arguments, everything else is unchanged
n_calls, n_env, same, inner = 0, 0, 0, 0
for f, L in looks:
    for t in L.get("trace") or []:
        if "tool" not in t or "args" not in t:
            continue
        n_calls += 1
        a, got = t["args"], agent.unwrap_args(t["tool"], t["args"])
        if is_env(a):
            n_env += 1
            inner += got == a["arguments"]
        else:
            same += got is a
check(f"all {n_calls} recorded tool calls (all sleeves, dry included): the {n_env} envelopes give their inner arguments",
      n_env == 13 and inner == n_env, f"{inner} of {n_env}")
check(f"the other {n_calls - n_env} recorded calls are unchanged", same == n_calls - n_env, f"{same} unchanged")

# ---- the same calls as the model wrote them: every gpt-oss turn re-parsed from its "said" with the new parse_any
n_said, rep_same, rep_inner, rep_env, newly, newly_ok, still_bad = 0, 0, 0, 0, [], 0, 0
for f, L in gpt:
    for t in L.get("trace") or []:
        said = t.get("said")
        if not said:
            continue
        kind, name, val = parse_said(said)
        if "tool" in t and t["tool"] != "submit":
            n_said += 1
            if is_env(t["args"]):
                rep_env += 1
                rep_inner += (kind, name, val) == ("TOOL", t["tool"], t["args"]["arguments"])
            else:
                rep_same += (kind, name, val) == ("TOOL", t["tool"], t["args"])
        elif "tool" in t:                                   # the look's decision: the file's "call", with cancel defaulted
            n_said += 1
            val = dict(val or {})
            val.setdefault("cancel", [])
            rep_same += kind == "TOOL" and name == "submit" and val == L.get("call")
        elif str(t.get("error", "")).startswith("MALFORMED"):
            if kind == "TOOL":
                newly.append((f.parent.name + "/" + f.stem, name, val))
                newly_ok += name == "submit" and agent.is_submit(val)
            else:
                still_bad += 1
print(f"{n_said} recorded gpt-oss calls re-parsed from what the model wrote")
check("re-parsed: the 13 envelopes give their inner arguments", rep_env == 13 and rep_inner == 13, f"{rep_inner} of {rep_env}")
check(f"re-parsed: the other {n_said - rep_env} calls and decisions parse as recorded", rep_same == n_said - rep_env,
      f"{rep_same} the same")
check("of the recorded malformed turns, only h-ml's 3 enveloped submits (9 Nov 11:00) now parse, each as a decision",
      len(newly) == 3 and newly_ok == 3 and all(k == "2026-11-09/h-ml-1100" for k, _, _ in newly),
      "; ".join(f"{k} {n} place={len((v or {}).get('place', []))}" for k, n, v in newly) + f"; {still_bad} stay malformed")

# ---- the shapes left alone
q = {"name": "AMAT", "symbols": ["AMAT"]}
check("a real argument dict with a 'name' key and no 'arguments' is unchanged", agent.unwrap_args("quote", q) is q)
q = {"name": "AMAT"}
check("{'name': ...} alone is unchanged", agent.unwrap_args("quote", q) is q)
e = {"name": "functions.news", "arguments": {"symbol": "AMAT"}}
check("an envelope naming another tool is left alone", agent.unwrap_args("quote", e) is e)
e = {"name": "functions.quote", "arguments": "AMAT"}
check("an envelope whose arguments are not an object is left alone", agent.unwrap_args("quote", e) is e)
e = {"name": "functions.quote", "arguments": {"symbols": ["A"]}, "x": 1}
check("an envelope with a third key is left alone", agent.unwrap_args("quote", e) is e)
check("an envelope with or without 'functions.' unwraps", agent.unwrap_args("quote", {"name": "quote", "arguments": {"symbols": ["A"]}})
      == agent.unwrap_args("quote", {"name": "functions.quote", "arguments": {"symbols": ["A"]}}) == {"symbols": ["A"]})
THINK = "<|channel|>analysis<|message|>Reasoning.<|end|><|start|>assistant"
k = parse_said(THINK + '<|channel|>commentary to=functions.submit <|constrain|>json<|message|>'
               '{"name": "functions.submit", "arguments": {"place": [], "cancel": [], "notes": "hold"}}<|call|>')
check("an enveloped submit is a decision", k == ("TOOL", "submit", {"place": [], "cancel": [], "notes": "hold"}), str(k))
k = parse_said(THINK + '<|channel|>commentary to=functions.submit <|constrain|>json<|message|>'
               '{"name": "functions.quote", "arguments": {"place": [], "cancel": []}}<|call|>')
check("a submit enveloped under another tool's name stays malformed", k[0] == "MALFORMED", str(k))


# ---- article ids: with or without the prefix, the same request (fake gateway, fake livenews)
class FakeGW:
    def __init__(self, sim=False):
        self.asked = []
        if sim:
            self.news_archive = object()

    def article(self, provider, article_id):
        self.asked.append((provider, article_id))
        return {"text": "body"}


fake_ln = types.ModuleType("ml_research.livenews")
fake_ln.asked = []


def fake_article(gw, provider, article_id):
    fake_ln.asked.append((provider, article_id))
    return {"text": "body"}


fake_ln.article = fake_article
import ml_research                                          # noqa: E402
real_ln = sys.modules.get("ml_research.livenews")
sys.modules["ml_research.livenews"], had = fake_ln, hasattr(ml_research, "livenews")
old_attr = getattr(ml_research, "livenews", None)
ml_research.livenews = fake_ln
try:
    for prov, ids, want in [("BRFG", ["1fc7c7a5", "BRFG$1fc7c7a5", " 1fc7c7a5 "], ("BRFG", "BRFG$1fc7c7a5")),
                            ("brfg", ["1fc7c7a5", "BRFG$1fc7c7a5"], ("BRFG", "BRFG$1fc7c7a5")),
                            ("BRFUPDN", ["1fc3c17d", "BRFUPDN$1fc3c17d"], ("BRFUPDN", "BRFUPDN$1fc3c17d")),
                            ("DJNL", ["abc123", "DJNL$abc123"], ("DJNL", "DJNL$abc123")),
                            ("MACRO", ["cal-2026-11-10", "MACRO$cal-2026-11-10"], ("MACRO", "MACRO$cal-2026-11-10")),
                            ("EDGAR", ["0001193125-26-452530", "EDGAR$0001193125-26-452530"], ("EDGAR", "0001193125-26-452530"))]:
        fake_ln.asked = []
        outs = [tools.run(FakeGW(), "article", {"provider": prov, "article_id": i}) for i in ids]
        check(f"live: {prov} ids {ids} all ask livenews for {want}", fake_ln.asked == [want] * len(ids) and outs == ["body"] * len(ids),
              str(fake_ln.asked))
        g = FakeGW(sim=True)
        outs = [tools.run(g, "article", {"provider": prov, "article_id": i}) for i in ids]
        check(f"simulator: {prov} ids all ask the gateway for {want}", g.asked == [want] * len(ids), str(g.asked))
    fake_ln.asked = []
    tools.run(FakeGW(), "article", {"provider": "BRFG", "article_id": "BRFUPDN$1fc3c17d"})
    check("an id carrying another provider's prefix is passed as written", fake_ln.asked == [("BRFG", "BRFUPDN$1fc3c17d")], str(fake_ln.asked))
finally:
    if real_ln is not None:
        sys.modules["ml_research.livenews"] = real_ln
    else:
        sys.modules.pop("ml_research.livenews", None)
    if had:
        ml_research.livenews = old_attr
    else:
        del ml_research.livenews

# the real livenews hands a BRFG id to the gateway as given (EDGAR and MACRO it serves itself), so the prefix reaches IBKR
from ml_research import livenews                            # noqa: E402
g = FakeGW()
for i in ["1fc7c7a5", "BRFG$1fc7c7a5"]:
    tools.run(g, "article", {"provider": "BRFG", "article_id": i})
check("live, the real livenews: both ids reach the gateway as BRFG$1fc7c7a5", g.asked == [("BRFG", "BRFG$1fc7c7a5")] * 2, str(g.asked))

# every recorded week-5 gpt-oss article call: the bare BRFG ids that failed now ask for the prefixed form, which worked elsewhere
worked, bare, fixed, changed_ok = set(), [], 0, 0
for f, L in gpt:
    for t in L.get("trace") or []:
        if t.get("tool") == "article" and isinstance(t.get("args"), dict):
            a, err = t["args"], str(t.get("result", "")).startswith("error")
            if not err:
                worked.add((str(a.get("provider")), str(a.get("article_id"))))
                changed_ok += tools.article_ref(a.get("provider"), a.get("article_id")) != (str(a.get("provider")), str(a.get("article_id")))
            elif str(a.get("provider")).upper() not in ("EDGAR",) and "$" not in str(a.get("article_id")):
                bare.append(a)
for a in bare:
    fixed += tools.article_ref(a.get("provider"), a.get("article_id")) in worked
check("no recorded article call that worked asks for anything else now", changed_ok == 0, f"{len(worked)} distinct ids that worked")
check("the recorded bare-id failures (BRFG) now ask for an id that worked in another look", len(bare) == 15 and fixed == 14,
      f"{fixed} of {len(bare)} (the 15th names provider 'Briefing.com', which IBKR does not take either way)")

print("ALL OK" if all(ok) else "FAILED")
sys.exit(0 if all(ok) else 1)
