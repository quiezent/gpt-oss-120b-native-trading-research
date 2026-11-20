"""ml_tools: each sleeve gets its own forecast tool version on both paths (gpt-oss and Claude), an unknown version gets
none, and ML2's model files and text are guarded. A fake ML3 stands in for a future version. Offline (the gpt-oss
tokenizer from the local cache; no gateway, no Tinker).

    python ml_tools_test.py
"""
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import tinker  # noqa: F401 - agent's dependency, first so a failure here reads plainly

import agent
import forecast
import live
import ml_tools
import tools
import trader

ROOT = Path(__file__).resolve().parent
results = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))
    results.append(bool(ok))


# ---------------------------------------------------------------- a fake ML3, registered as a new version would be
ML3_SPEC = {"name": "forecast", "description": "the ML3 tool's own text",
            "parameters": {"type": "object", "properties": {k: {"description": k} for k in ("symbols", "notional_usd", "sleeve_usd")}}}
fake = types.ModuleType("forecast_ml3_fake")
fake.VERSION, fake.SPEC, fake.calls = "ML3.0", ML3_SPEC, []
fake.run = lambda gw, args, **kw: (fake.calls.append(args), "ML3 ROW")[1]    # **kw: ML2.3's sleeve_value and look
fake.compute = lambda: ("20261030", {})
sys.modules["forecast_ml3_fake"] = fake
ml_tools.VERSIONS["ML3"] = "forecast_ml3_fake"

ml2_calls, real_run = [], forecast.run
forecast.run = lambda gw, args, **kw: (ml2_calls.append(args), "ML2 ROW")[1]

# ---------------------------------------------------------------- the registry
check("no version means ML2", ml_tools.module() is forecast and ml_tools.module("ML2") is forecast)
check("ML3 is its own module", ml_tools.module("ML3") is fake)
try:
    ml_tools.module("ML9")
    check("an unknown version raises", False)
except KeyError as e:
    check("an unknown version raises", "ML9" in str(e), str(e))
S = {"a": {"ml": True}, "b": {"ml": True, "ml_tool": "ML2"}, "c": {"ml": True, "ml_tool": "ML3"},
     "d": {"ml": False}, "e": {"ml": True, "ml_tool": "ML9"}}
check("unknown(): only the sleeve with a version not registered", ml_tools.unknown(S) == ["e"], str(ml_tools.unknown(S)))
check("in_use(): the default first", ml_tools.in_use({k: S[k] for k in "abcd"}) == ["ML2", "ML3"])
check("version_of(): the exact revision, None without the tool, the name if unknown",
      ml_tools.version_of(S["a"]) == forecast.VERSION and ml_tools.version_of(S["c"]) == "ML3.0" and ml_tools.version_of(S["d"]) is None
      and ml_tools.version_of(S["e"]) == "ML9")

# ---------------------------------------------------------------- the refresh: one version's trouble stays its own
real_sleeves, real_log, real_compute = live.paper.sleeves, live.log, forecast.compute
said = []
live.log = said.append
forecast.compute = lambda: ("20261030", {"A": {}})


def boom3():
    raise RuntimeError("ML3 broke")


live.paper.sleeves = lambda: {"a": {"ml": True}, "c": {"ml": True, "ml_tool": "ML3"}, "e": {"ml": True, "ml_tool": "ML9"}}
fake.compute = boom3
live.compute_versions()
check("compute_versions: ML2 runs; a broken ML3 and an unknown ML9 are logged, not raised",
      any("ML2" in x and "1 symbols" in x for x in said) and any("ML3 forecasts failed" in x for x in said) and any("no forecasts for ML9" in x for x in said),
      str(len(said)))


def boom2():
    raise RuntimeError("ML2 broke")


forecast.compute = boom2
try:
    live.compute_versions()
    check("compute_versions: ML2's failure raises, so the refresh retries", False)
except RuntimeError:
    check("compute_versions: ML2's failure raises, so the refresh retries", True)

# ML2 is computed whenever any version is in use, and a failed other version is tried again alone (16 Nov 2026: from Week 6
# no sleeve is on ML2, whose file ML3 and ML4 read; with every sleeve on one version its failed compute had no second try)
from datetime import datetime as _dt, timedelta as _td      # noqa: E402
ran = []
forecast.compute = lambda: (ran.append("ML2"), ("20261030", {"A": {}}))[1]
fake.compute = lambda: (ran.append("ML3"), ("20261030", {}))[1]
live.paper.sleeves = lambda: {"c": {"ml": True, "ml_tool": "ML3"}, "d": {"ml": False}}
del said[:]
check("compute_versions: with no sleeve on ML2, ML2 is still computed, first; nothing failed",
      live.compute_versions() == [] and ran == ["ML2", "ML3"] and live.COMPUTE_RETRY["versions"] == [], str(ran))
live.paper.sleeves = lambda: {"d": {"ml": False}}
del ran[:]
check("compute_versions: no sleeve with the tool, nothing is computed", live.compute_versions() == [] and ran == [], str(ran))
live.paper.sleeves = lambda: {"c": {"ml": True, "ml_tool": "ML3"}, "e": {"ml": True, "ml_tool": "ML9"}}
state = {"ok": False}


def flaky3():
    ran.append("ML3")
    if not state["ok"]:
        raise RuntimeError("ML3 broke")
    return "20261030", {}


fake.compute = flaky3
del ran[:], said[:]
failed = live.compute_versions()
t0 = live.COMPUTE_RETRY["last"]
check("compute_versions: a failed ML3 is returned and kept for the retry; ML2 ran and did not raise; the unknown ML9 is "
      "not a retry", failed == ["ML3"] and ran == ["ML2", "ML3"] and live.COMPUTE_RETRY["versions"] == ["ML3"]
      and live.COMPUTE_RETRY["day"] == t0.date(), str((failed, ran, live.COMPUTE_RETRY)))
day = t0.replace(hour=10, minute=0, second=0, microsecond=0)
live.COMPUTE_RETRY["last"] = day
del ran[:]
live.compute_retry(day + _td(minutes=4))
check("compute_retry: nothing within 5 minutes of the last try", ran == [] and live.COMPUTE_RETRY["versions"] == ["ML3"])
live.compute_retry(day + _td(minutes=5))
check("compute_retry: after 5 minutes the failed version is tried again, alone (ML2 is not recomputed); still failing, kept",
      ran == ["ML3"] and live.COMPUTE_RETRY["versions"] == ["ML3"] and live.COMPUTE_RETRY["last"] == day + _td(minutes=5)
      and any("trying ML3 again" in x for x in said) and any("ML3 still failing" in x for x in said), str((ran, said[-3:])))
state["ok"] = True
del ran[:]
live.compute_retry(day + _td(minutes=10))
check("compute_retry: once it succeeds the version is off the list and it is logged",
      ran == ["ML3"] and live.COMPUTE_RETRY["versions"] == [] and any("the retry succeeded" in x for x in said), str(said[-2:]))
del ran[:]
live.compute_retry(day + _td(minutes=30))
check("compute_retry: nothing left to retry, nothing runs", ran == [])
state["ok"] = False
live.COMPUTE_RETRY.update(day=day.date(), versions=["ML3"], last=day)
live.compute_retry(day.replace(hour=15, minute=30))
live.compute_retry(day + _td(days=1, minutes=30))
check("compute_retry: not at or after 15:30, and not on another day", ran == [] and live.COMPUTE_RETRY["versions"] == ["ML3"])


def no_sleeves():
    raise OSError("sleeves.json unreadable")


live.paper.sleeves = no_sleeves
try:
    live.compute_retry(day + _td(minutes=20))
    check("compute_retry: never raises (the ledger unreadable: logged)", any("the retry failed" in x for x in said), str(said[-1:]))
except Exception as e:                                       # noqa: BLE001
    check("compute_retry: never raises (the ledger unreadable: logged)", False, repr(e))
live.COMPUTE_RETRY.update(day=None, versions=[], last=None)
src = open(live.__file__, encoding="utf-8").read()
check("the loop calls compute_retry between jobs, and refresh still calls compute_versions before the warm-up",
      "            compute_retry()" in src and src.index("        compute_versions()\n        warm_outside()") > 0)
live.paper.sleeves, live.log, forecast.compute = real_sleeves, real_log, real_compute
fake.compute = lambda: ("20261030", {})

# ---------------------------------------------------------------- tools.run
gw = types.SimpleNamespace()
check("tools.run, no version: ML2", tools.run(gw, "forecast", {"symbols": ["A"]}) == "ML2 ROW")
check("tools.run, ML2", tools.run(gw, "forecast", {"symbols": ["A"]}, ml_tool="ML2") == "ML2 ROW")
check("tools.run, ML3", tools.run(gw, "forecast", {"symbols": ["A"]}, ml_tool="ML3") == "ML3 ROW")
check("tools.run, unknown: an error as text, not another version's rows",
      "ROW" not in tools.run(gw, "forecast", {"symbols": ["A"]}, ml_tool="ML9"), tools.run(gw, "forecast", {"symbols": ["A"]}, ml_tool="ML9"))


# ---------------------------------------------------------------- the gpt-oss path, through agent.gptoss_look
def fake_sampler(turns):
    """agent.sample_once's stand-in: one scripted turn per call."""
    it = iter(turns)

    def sample_once(sc, renew, model_input, params, trace, step):
        toks = list(trader.tokenizer().encode(next(it), add_special_tokens=False))
        return types.SimpleNamespace(sequences=[types.SimpleNamespace(tokens=toks, stop_reason="stop")]), sc
    return sample_once


CALL = '<|channel|>analysis<|message|>Check the risk.<|end|><|start|>assistant<|channel|>commentary to=functions.forecast json<|message|>{"symbols": ["NVDA"]}<|call|>'
SUBMIT = '<|channel|>commentary to=functions.submit <|constrain|>json<|message|>{"place": [], "cancel": [], "notes": "hold"}<|call|>'
real_sample = agent.sample_once
for version, want, spec in ((None, "ML2 ROW", forecast.SPEC), ("ML2", "ML2 ROW", forecast.SPEC), ("ML3", "ML3 ROW", ML3_SPEC)):
    agent.sample_once = fake_sampler([CALL, SUBMIT])
    call, trace, _, _ = agent.gptoss_look(None, gw, "system", "state", "2026-11-02", [spec], "high", ml_tool=version)
    got = [t.get("result", "") for t in trace if t.get("tool") == "forecast"]
    check(f"gptoss_look on {version}: its tool's rows reach the model", call is not None and len(got) == 1 and got[0].startswith(want), got[0][:40] if got else "no call")
agent.sample_once = real_sample


# ---------------------------------------------------------------- the Claude path: mcp.json, then the tool server
class Stop(Exception):
    pass


def boom(*a, **k):
    raise Stop


with tempfile.TemporaryDirectory() as d:
    real = agent.subprocess
    agent.subprocess = types.SimpleNamespace(run=boom, Popen=boom, PIPE=real.PIPE, STDOUT=real.STDOUT, DEVNULL=real.DEVNULL,
                                             TimeoutExpired=real.TimeoutExpired)
    try:
        agent.claude_look("claude-haiku-5-5", "system", "state", [ML3_SPEC], Path(d) / "w", "max", client="trader-mcp-c", ml_tool="ML3")
    except Stop:
        pass
    finally:
        agent.subprocess = real
    env = json.loads((Path(d) / "w" / "mcp.json").read_text(encoding="utf-8"))["mcpServers"]["trader"]["env"]
    check("claude_look: mcp.json carries the version and the client", env.get("TRADER_ML_TOOL") == "ML3" and env.get("TRADER_MCP_CLIENT") == "trader-mcp-c",
          f"{env.get('TRADER_ML_TOOL')}, {env.get('TRADER_MCP_CLIENT')}")

    PROBE = """
import sys, types
sys.path.insert(0, sys.argv[1])
import ml_tools
fake = types.ModuleType("forecast_ml3_fake")
fake.VERSION = "ML3.0"
fake.SPEC = {"name": "forecast", "description": "the ML3 tool's own text",
             "parameters": {"type": "object", "properties": {k: {"description": k} for k in ("symbols", "notional_usd", "sleeve_usd")}}}
sys.modules["forecast_ml3_fake"] = fake
ml_tools.VERSIONS["ML3"] = "forecast_ml3_fake"
import mcp_server
print(mcp_server.ML_TOOL, mcp_server._forecast.VERSION, mcp_server._forecast.SPEC["description"][:18])
"""

    def server(version):
        e = {**__import__("os").environ, "TRADER_SUBMIT_FILE": str(Path(d) / "s.json"), "TRADER_LOG_FILE": str(Path(d) / "l.jsonl"),
             "TRADER_HAS_FORECAST": "1", "TRADER_MCP_CLIENT": "trader-mcp-a", "TRADER_ML_TOOL": version}
        r = subprocess.run([sys.executable, "-c", PROBE, str(ROOT)], capture_output=True, text=True, encoding="utf-8", errors="replace", env=e, cwd=d, timeout=180)
        return (r.stdout.strip().splitlines() or [""])[-1] if r.returncode == 0 else f"rc {r.returncode}: {r.stderr[-200:]}"

    got = server("ML3")
    check("mcp_server with ML3: ML3's tool and text", got.startswith("ML3 ML3.0 the ML3 tool"), got)
    got = server("")
    check("mcp_server with none: ML2's", got.startswith(f"None {forecast.VERSION} "), got)

    # ML4 (forecast4.py, real modules, no fake): the Claude path declares track_record only for ML4; ML2's and ML3's tool is
    # declared exactly as before; a record call through the server returns the record and counts as one research call
    PROBE4 = """
import asyncio, json, sys
sys.path.insert(0, sys.argv[1])
import mcp_server
async def go():
    t = next(x for x in await mcp_server.server.list_tools() if x.name == "forecast")
    sch = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None)
    out = {"version": mcp_server._forecast.VERSION, "props": sorted(sch["properties"]),
           "required": sch.get("required"), "desc_same": t.description == mcp_server._forecast.SPEC["description"]}
    if "track_record" in out["props"]:
        mcp_server.gw = lambda: object()            # no gateway: a record-only call does not use it
        r = await mcp_server.server.call_tool("forecast", {"symbols": [], "track_record": True})
        text = "".join(getattr(c, "text", "") for c in (r.content if hasattr(r, "content") else r))
        out["record"] = text == mcp_server._forecast.TRACK_RECORD + "\\n(research call 1 of 12)"
        out["calls"] = mcp_server._calls
    print(json.dumps(out))
asyncio.run(go())
"""

    def server4(version):
        e = {**__import__("os").environ, "TRADER_SUBMIT_FILE": str(Path(d) / "s4.json"), "TRADER_LOG_FILE": str(Path(d) / "l4.jsonl"),
             "TRADER_HAS_FORECAST": "1", "TRADER_MCP_CLIENT": "trader-mcp-a", "TRADER_ML_TOOL": version}
        r = subprocess.run([sys.executable, "-c", PROBE4, str(ROOT)], capture_output=True, text=True, encoding="utf-8", errors="replace", env=e, cwd=d, timeout=180)
        try:
            return json.loads((r.stdout.strip().splitlines() or ["{}"])[-1])
        except ValueError:
            return {"rc": r.returncode, "err": r.stderr[-300:]}

    three = ["notional_usd", "sleeve_usd", "symbols"]
    for v, want in (("", "ML2.4"), ("ML2", "ML2.4"), ("ML3", "ML3.2")):
        got = server4(v)
        check(f"mcp_server, real {v or 'default'}: {want}'s text, the three parameters as before",
              got.get("version") == want and got.get("props") == three and got.get("required") == ["symbols"] and got.get("desc_same"), str(got))
    got = server4("ML4")
    check("mcp_server, ML4: ML4.0's text, track_record declared; a record call returns the record and is research call 1 of 12",
          got.get("version") == "ML4.1" and got.get("props") == sorted(three + ["track_record"]) and got.get("required") == ["symbols"]
          and got.get("desc_same") and got.get("record") and got.get("calls") == 1, str(got))
    logged = [json.loads(x) for x in (Path(d) / "l4.jsonl").read_text(encoding="utf-8").splitlines()] if (Path(d) / "l4.jsonl").exists() else []
    check("mcp_server, ML4: the record call is logged with its track_record argument",
          len(logged) == 1 and logged[0]["args"].get("track_record") is True, str([x.get("args") for x in logged]))

# ---------------------------------------------------------------- ML4 on the gpt-oss path (forecast4.py; ML2's run is faked above)
import forecast4  # noqa: E402
check("ML4 is forecast4.py, with its own SPEC", ml_tools.module("ML4") is forecast4 and ml_tools.module("ML4").SPEC is forecast4.SPEC
      and ml_tools.version_of({"ml": True, "ml_tool": "ML4"}) == "ML4.1")
check("tools.run, ML4: ML2's text comes through forecast4 (no ML2 header here, so nothing is added)",
      tools.run(gw, "forecast", {"symbols": ["A"]}, ml_tool="ML4") == "ML2 ROW")
REC = '<|channel|>analysis<|message|>Read the record.<|end|><|start|>assistant<|channel|>commentary to=functions.forecast json<|message|>{"symbols": [], "track_record": true}<|call|>'
agent.sample_once = fake_sampler([CALL, REC, SUBMIT])
call, trace, _, _ = agent.gptoss_look(None, gw, "system", "state", "2026-11-16", [forecast4.SPEC], "high", ml_tool="ML4")
agent.sample_once = real_sample
got = [t.get("result", "") for t in trace if t.get("tool") == "forecast"]
check("gptoss_look on ML4: rows, then a record call that is research call 2 of 12",
      call is not None and len(got) == 2 and got[0].startswith("ML2 ROW") and got[0].endswith("(research call 1 of 12)")
      and got[1] == forecast4.TRACK_RECORD + f"\n(research call 2 of {tools.MAX_CALLS})", str([g[:40] + " ... " + g[-30:] for g in got]))

# ---------------------------------------------------------------- four Claude lanes
ids = json.loads(Path(r"C:\Claude\TWS-Gateway\clients.json").read_text(encoding="utf-8"))
check("each Claude lane has its own research client", len(set(live.MCP_CLIENTS)) == len(live.MCP_CLIENTS) and all(c in ids for c in live.MCP_CLIENTS),
      str({c: ids.get(c) for c in live.MCP_CLIENTS}))
jobs = list("abcd")
check("four sleeves, four lanes, one each", [jobs[k::len(live.MCP_CLIENTS)] for k in range(len(live.MCP_CLIENTS))] == [["a"], ["b"], ["c"], ["d"]])

# ---------------------------------------------------------------- ML2's guard
check("ML2's files and text match the manifest", ml_tools.changed_files() == [], str(ml_tools.changed_files()))
check("the guard covers code, models, EDGAR data and text",
      all(f in ml_tools.ML2_FILES for f in ("forecast.py", "ml_research/vol2.py", "runs/ml/research/events_model.pkl", "data/edgar/filings.csv"))
      and len(ml_tools._texts()) == 4)
with tempfile.TemporaryDirectory() as d:
    real_manifest = ml_tools.MANIFEST
    doc = json.loads(real_manifest.read_text(encoding="utf-8"))
    name = sorted(doc["files"])[0]
    ml_tools.MANIFEST = Path(d) / "m.json"
    ml_tools.MANIFEST.write_text(json.dumps({**doc, "files": {**doc["files"], name: "0" * 16}}), encoding="utf-8")
    check("a changed model file is named", ml_tools.changed_files() == [name], str(ml_tools.changed_files()))
    ml_tools.MANIFEST.write_text(json.dumps({**doc, "texts": {**doc["texts"], "trader.S3_ML": "0" * 16}}), encoding="utf-8")
    check("changed instruction text is named", ml_tools.changed_files() == ["trader.S3_ML"], str(ml_tools.changed_files()))
    ml_tools.MANIFEST.write_text(json.dumps({**doc, "version": "ML2.0"}), encoding="utf-8")
    check("a revision with no new manifest is named", any("VERSION" in x for x in ml_tools.changed_files()), str(ml_tools.changed_files()))
    ml_tools.MANIFEST = real_manifest

forecast.run = real_run
print("ALL OK" if all(results) else "FAILED")
sys.exit(0 if all(results) else 1)
