"""One look: a model reads its sleeve and the market, may research with tools.py, then submits orders.

gptoss_look  runs base / trained gpt-oss checkpoints on Tinker with native (Harmony) tool calls.
claude_look  runs a Claude model through the Claude CLI, headless, with every built-in tool disabled and
             only mcp_server.py's tools (the same research tools and submit).
Both return (call or None, trace), where trace records every research call for the research log.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import tinker

import tools
import trader

ROOT = Path(__file__).resolve().parent
# openai/gpt-oss-120b (rl2's base) has a 32,768-token context on Tinker (server capabilities). At effort "high"
# rl2-step010 wrote turns up to 7,174 tokens and whole looks up to 19,313 (18 Oct, 12 simulated looks), so each turn
# may use up to MAX_OUT, less whatever the context has no room for.
CONTEXT, MAX_OUT, MIN_ROOM = 32768, 16384, 1024
CLAUDE_TIMEOUT = 1200


# ---------------------------------------------------------------- gpt-oss (Tinker)
SEG = trader.SEG


def parse_any(tokens, stop_reason, names):
    """('TOOL', name, args) | ('FINAL', None, text) | ('MALFORMED', None, reason)."""
    tok = trader.tokenizer()
    if str(stop_reason).split(".")[-1].lower() != "stop" or not tokens or tokens[-1] not in (trader.CALL, trader.RET):
        return "MALFORMED", None, "output ended without a tool call"
    segs = tok.decode(tokens).split("<|start|>assistant")
    for k, seg in enumerate(segs):
        m = SEG.match(seg)
        if not m or "<|" in m["body"]:
            return "MALFORMED", None, f"unparseable segment {k}"
        rc = m["r1"] or m["r2"]
        if k < len(segs) - 1:
            if m["term"] != "<|end|>" or rc:
                return "MALFORMED", None, "a call before the end"
            continue
        if m["term"] == "<|call|>":
            name = (rc or "").removeprefix("functions.")
            if name not in names:
                return "MALFORMED", None, f"call to undeclared tool {rc!r}"
            try:
                args = json.loads(m["body"])
            except ValueError as e:
                return "MALFORMED", None, f"bad JSON: {e}"
            return "TOOL", name, args
        return "FINAL", None, m["body"]
    return "MALFORMED", None, "empty"


def SUBMIT_OPEN():
    return list(trader.tokenizer().encode("<|channel|>commentary to=functions.submit <|constrain|>json<|message|>",
                                          add_special_tokens=False))


def tool_result_tokens(name, text):
    s = f"<|start|>functions.{name} to=assistant<|channel|>commentary<|message|>{text}<|end|><|start|>assistant"
    return list(trader.tokenizer().encode(s, add_special_tokens=False))


def harmony_specs(specs):
    """Tool specs for the gpt-oss renderer, which writes a description as one "// " comment line: a multi-line
    description (forecast's) gets "// " on every line, as the Harmony format writes it."""
    return [dict(s, description=s["description"].replace("\n", "\n// ")) if "\n" in s.get("description", "") else s
            for s in specs]


SAMPLE_TIMEOUT = 600   # s a call; a 16,384-token answer takes about 5 minutes


def sample_once(sc, renew, model_input, params, trace, step):
    """One sample with a deadline. tinker-feedback #151 (open; Tinker Owner, 25 Oct): a dead pooled connection can hang a
    sample forever, which in live.py would stall every later sleeve's look. On a timeout, retry once on a new client from
    renew(); a second timeout raises, ending only this look."""
    try:
        return sc.sample(model_input, 1, params).result(timeout=SAMPLE_TIMEOUT), sc
    except TimeoutError:
        if renew is None:
            raise
        trace.append({"step": step, "error": f"sample timed out after {SAMPLE_TIMEOUT} s; retried on a new client"})
        sc = renew()
        return sc.sample(model_input, 1, params).result(timeout=SAMPLE_TIMEOUT), sc


def gptoss_look(sc, gw, system, state_text, current_date, research_specs, effort, seed=None, renew=None):
    from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
    r = GptOssRenderer(trader.tokenizer(), use_system_prompt=True, reasoning_effort=effort, current_date=current_date)
    specs = research_specs + [trader.TOOL_LIVE]
    names = {s["name"] for s in specs}
    msgs = r.create_conversation_prefix_with_tools(harmony_specs(specs), system)
    msgs.append({"role": "user", "content": state_text})
    prompt = list(r.build_generation_prompt(msgs).to_ints())
    trace, calls, sampled, prefill, bad, force = [], 0, 0, 0, 0, []
    for step in range(tools.MAX_CALLS + 4):
        room = CONTEXT - len(prompt) - len(force)
        if room < MIN_ROOM:
            trace.append({"step": step, "error": f"context full ({len(prompt)} tokens)"})
            return None, trace, prefill, sampled
        params = tinker.SamplingParams(temperature=1.0, max_tokens=min(MAX_OUT, room), stop=[trader.RET, trader.CALL],
                                       **({"seed": seed * 100 + step} if seed is not None else {}))
        res, sc = sample_once(sc, renew, tinker.ModelInput.from_ints(prompt + force), params, trace, step)
        seq = res.sequences[0]
        toks = force + list(seq.tokens)
        prefill += len(prompt) + len(force)
        sampled += len(seq.tokens)
        kind, name, val = parse_any(toks, seq.stop_reason, names)
        said = trader.tokenizer().decode(toks)            # the reasoning and call, in full: week 3 is the next replay set
        if kind != "TOOL":
            bad += 1
            trace.append({"step": step, "error": f"{kind}: {str(val)[:200]}", "tokens": len(seq.tokens), "said": said})
            if bad >= 3:
                return None, trace, prefill, sampled
            if kind == "FINAL" or bad == 2:        # prose, or a second malformed try: open the submit call; the model writes its orders
                force = SUBMIT_OPEN()
            continue                               # re-ask at the same point (as the daily loop re-asks malformed output)
        if name == "submit":
            if not isinstance(val, dict) or not isinstance(val.get("place", []), list) or not isinstance(val.get("cancel", []), list):
                return None, trace + [{"step": step, "error": "submit arguments malformed"}], prefill, sampled
            val.setdefault("place", [])
            val.setdefault("cancel", [])
            trace.append({"step": step, "tool": "submit", "tokens": len(seq.tokens), "said": said})
            return val, trace, prefill, sampled
        if calls >= tools.MAX_CALLS:      # open the submit call, as for prose: at high effort it kept researching
            out = f"research limit reached ({tools.MAX_CALLS} calls): call submit now"
            force = SUBMIT_OPEN()
        else:                             # the count: week 2, 135 of 276 gpt-oss looks reached the cap, never told it
            calls += 1
            out = tools.run(gw, name, val) + f"\n(research call {calls} of {tools.MAX_CALLS})"
        trace.append({"step": step, "tool": name, "args": val, "result": out, "tokens": len(seq.tokens), "said": said})
        prompt = prompt + toks + tool_result_tokens(name, out)
    return None, trace + [{"error": "no submit within the step limit"}], prefill, sampled


# ---------------------------------------------------------------- Claude (CLI + MCP)
def claude_exe():
    """The newest Claude CLI bundled with the desktop app. The app is a packaged Windows app, so its
    %APPDATA% is virtualized: processes inside the package see it under %APPDATA%, others only under
    %LOCALAPPDATA%\\Packages\\Claude_*\\LocalCache\\Roaming. Since the app update of 20 Oct 2026 it lives
    in %LOCALAPPDATA%\\Claude-Data\\claude_*\\claude-code."""
    local = Path(os.environ["LOCALAPPDATA"])
    bases = [Path(os.environ["APPDATA"]) / "Claude" / "claude-code"]
    bases += [p / "LocalCache" / "Roaming" / "Claude" / "claude-code" for p in (local / "Packages").glob("Claude_*")]
    bases += [p / "claude-code" for p in (local / "Claude-Data").glob("claude_*")]
    exes = sorted((e for b in bases for e in b.glob("*/*/claude.exe")), key=lambda p: p.stat().st_mtime)
    if not exes:
        raise FileNotFoundError("Claude CLI not found (desktop app bundle)")
    return str(exes[-1])


def claude_look(model, system, state_text, research_specs, workdir, effort, timeout=CLAUDE_TIMEOUT, client="trader-mcp-a"):
    """Run one look with the Claude CLI at reasoning effort low|medium|high|xhigh|max. The MCP server researches on
    the read-only client named by `client` (trader-mcp-a = 12, trader-mcp-b = 13), so the caller may hold client 10. A submit made
    before the CLI times out still counts."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    submit_file, log_file = workdir / "submit.json", workdir / "research.jsonl"
    for f in (submit_file, log_file):
        if f.exists():
            f.unlink()
    have = {s["name"] for s in research_specs}
    env = dict(os.environ, TRADER_SUBMIT_FILE=str(submit_file), TRADER_LOG_FILE=str(log_file),
               TRADER_HAS_NEWS="1" if "news" in have else "0", TRADER_HAS_SCANNER="1" if "scanner" in have else "0",
               TRADER_HAS_STATS="1" if "stats" in have else "0", TRADER_HAS_FORECAST="1" if "forecast" in have else "0",
               TRADER_MCP_CLIENT=client, CLAUDE_CODE_DISABLE_AUTO_MEMORY="1")
    cfg = {"mcpServers": {"trader": {"command": sys.executable,
                                     "args": [str(ROOT / "mcp_server.py")], "env": {k: env[k] for k in
                                     ("TRADER_SUBMIT_FILE", "TRADER_LOG_FILE", "TRADER_HAS_NEWS", "TRADER_HAS_SCANNER",
                                      "TRADER_HAS_STATS", "TRADER_HAS_FORECAST", "TRADER_MCP_CLIENT")}}}}
    cfg_file = workdir / "mcp.json"
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
    allowed = ",".join(f"mcp__trader__{n}" for n in sorted(have | {"submit"}))
    instructions = (system + "\n\nUse the tools named " + ", ".join(sorted(have)) + f" for research (at most {tools.MAX_CALLS} calls), "
                    "then call submit exactly once. Your decision is what you pass to submit; any other text is ignored.")
    # The look sees only its system prompt, state and tools: --setting-sources "" keeps out the user's and the repo's
    # CLAUDE.md, and the env var above the session memory index, which the CLI otherwise loads from the workdir (25 Oct).
    exe = claude_exe()
    cmd = [exe, "-p", state_text, "--model", model, "--effort", effort, "--system-prompt", instructions, "--tools", "",
           "--mcp-config", str(cfg_file), "--strict-mcp-config", "--allowedTools", allowed, "--setting-sources", "",
           "--output-format", "stream-json", "--verbose", "--no-session-persistence"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
                           timeout=timeout, cwd=str(workdir))
        rc, stdout, stderr = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        txt = lambda x: x.decode("utf-8", "replace") if isinstance(x, bytes) else (x or "")
        rc, stdout, stderr = f"timeout after {timeout:.0f}s", txt(e.stdout), txt(e.stderr)
    trace = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()] if log_file.exists() else []
    events = []
    for l in stdout.splitlines():
        try:
            events.append(json.loads(l))
        except ValueError:
            pass
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), {})
    out = next((e for e in reversed(events) if e.get("type") == "result"), None) or {"raw": stdout[-2000:]}
    blocks = [b for e in events if e.get("type") == "assistant" for b in (e.get("message") or {}).get("content") or []]
    # cli_result stays the final message (week_read's frozen M3 input); cli_text and cli_thinking keep every message, so
    # the opus looks can be read like gpt-oss's "said"; the CLI version and served model record what ran (25 Oct probe).
    trace.append({"cli_returncode": rc, "cli_result": str(out.get("result", out.get("raw", "")))[:2000],
                  "stderr": stderr[-1000:], "cost_usd": out.get("total_cost_usd"), "turns": out.get("num_turns"),
                  "usage": out.get("usage"),
                  "cli_version": Path(exe).parents[1].name, "cli_model": init.get("model"),
                  "cli_mcp": init.get("mcp_servers"),
                  "cli_args": ["<state>" if a == state_text else "<system>" if a == instructions else a for a in cmd[1:]],
                  "cli_text": [b.get("text") for b in blocks if b.get("type") == "text"],
                  "cli_thinking": [b["thinking"] for b in blocks if b.get("type") == "thinking" and b.get("thinking")],
                  # the plan's usage windows as the CLI reports them (the 7-day window was at 83% on 29 Oct)
                  "cli_rate_limit": [e for e in events if e.get("type") == "rate_limit_event"][-3:]})
    call = json.loads(submit_file.read_text(encoding="utf-8")) if submit_file.exists() else None
    return call, trace
