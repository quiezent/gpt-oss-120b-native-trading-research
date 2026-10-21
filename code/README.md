# Code

A snapshot of Trader Training's code as of 21 October 2026. It runs on Windows with Python 3.11, against IB Gateway (paper port) through the TWS Gateway Owner's broker framework. That framework is not included here; `trader.py`'s `PaperWorld` and the research tools call it. Market data, news, checkpoints and run logs are not included.

| File | What it does |
|---|---|
| `live.py` | The live loop: 07:30 data refresh, looks every 30 minutes from 09:00 to 15:30 ET, fill syncs, the 16:15 record |
| `agent.py` | One look: gpt-oss on Tinker with native tool calls, or Claude through the Claude CLI |
| `tools.py`, `mcp_server.py` | The research tools and `submit`, the same for every model (MCP for Claude) |
| `paper.py` | Splits the paper account into sleeves by order reference; cash, positions and costs per sleeve |
| `trader.py` | Prompts, the market table and order parsing, shared by simulation and the paper account |
| `sim.py`, `simgw.py` | Daily-bar market simulator, and a point-in-time stand-in for the gateway |
| `reward.py` | The training reward (see the [training report](../reports/2026-10-18-training-rl1-to-rl3.md)) |
| `train.py`, `train2.py`, `sft2.py`, `eval2.py` | rl1; rl2 and rl3; the rejection-sampling SFT test; paired evaluation |
| `forecast.py`, `ml.py`, `ml_eval.py`, `ml_research/` | The ML forecast tool, and every model we tested for it (see the [ML report](../reports/2026-10-19-ml-forecast-tool.md)) |
| `p1.py`, `nav_runner.py`, `analyze_nav.py`, `tsla_counterfactual.py` | Our 14 October re-test of Codex's NAV4 and NAV9 |
| `lists/` | Stock universes, industries and the ETF list |

API keys are read from environment variables (`ml_research/keys.py`); none are in the code. No open-source licence has been chosen yet, so ask before reusing the code.
