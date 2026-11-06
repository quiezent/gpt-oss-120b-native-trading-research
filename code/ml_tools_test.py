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
fake.run = lambda gw, args: (fake.calls.append(args), "ML3 ROW")[1]
fake.compute = lambda: ("20261030", {})
sys.modules["forecast_ml3_fake"] = fake
ml_tools.VERSIONS["ML3"] = "forecast_ml3_fake"

ml2_calls, real_run = [], forecast.run
forecast.run = lambda gw, args: (ml2_calls.append(args), "ML2 ROW")[1]

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
