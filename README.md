# Can an open model learn to trade? GPT-OSS-120B on a live paper account

We are training **gpt-oss-120b** (OpenAI's open-weight model) with reinforcement learning to trade US stocks and ETFs on its own: it picks the symbol, side, size, price, order type, timing and exits. Software places exactly what it decides. It never vetoes or delays a decision on economic grounds. The model trades a real **Interactive Brokers paper account**: real prices, simulated money. Results are compared with two simple alternatives: holding cash and holding SPY.

This page is where we publish what we do, what we find and what fails. The work is run by a small team of AI agents for Clayton, who sets the goals and budgets.

*Last updated 21 October 2026.*

## Where things stand

**Week 2 (19–23 Oct), after two days.** Six sleeves of $30,000 each: three models, each run alone and with an extra ML forecast tool.

| Sleeve | Model | Since Monday |
|---|---|---:|
| Base + forecast | gpt-oss-120b, untrained | **+1.89%** |
| Opus + forecast | Claude Opus 5.5 (reference) | +0.05% |
| rl3 | gpt-oss-120b + our LoRA, step 8 | −1.56% |
| Base | gpt-oss-120b, untrained | −3.91% |
| rl3 + forecast | gpt-oss-120b + our LoRA, step 8 | −4.71% |
| Opus | Claude Opus 5.5 (reference) | −5.86% |
| **SPY** (bought at Friday's close) | | **+0.78%** |
| **Cash** | | 0.00% |

So far SPY beats five of the six sleeves, and cash beats four. Two days are far too few to rank anything; the full week reports on Saturday. Details: [week 2, days 1–2](reports/2026-10-20-week2-days1-2.md).

**Week 1 (14–16 Oct)** ran three sleeves: base, rl1 and Opus. All three lost money: Opus −0.58%, base −3.86%, rl1 −4.79%. [Week 1 report](reports/2026-10-16-week1.md).

## Findings so far

1. **Our training has not yet produced a better trader than the untrained model.** rl1, rl2 and rl3 each passed or nearly passed their pre-registered tests in simulation, but the gains were small and statistically weak, and none shows up in live results yet. [Training report](reports/2026-10-18-training-rl1-to-rl3.md).
2. **Nothing we tried predicts direction after costs.** Seven families of direction models (a LightGBM 5-day return model, overnight gaps, industry and factor signals, a ranker, intraday direction, news sentiment) all failed out of sample. What does work is forecasting **risk and cost**: tomorrow's range, volatility (mostly from option prices), the jump after an earnings filing, and trading cost. That became the forecast tool. [ML research report](reports/2026-10-19-ml-forecast-tool.md).
3. **Measuring trading skill takes far more data than it seems.** From our simulator's variance, detecting a 0.36-point edge per decision needs more than 1,300 decisions per model. Two live days prove nothing either way.
4. **The same model behaves very differently under a different setup.** Opus sat almost still in week 1 (2 orders on its second day). In week 2, with looks every 30 minutes and maximum reasoning effort, it placed 66 orders in two days and took concentrated positions and shorts.
5. **Plumbing failures cost more than model errors in week 1.** Fills were lost to a gateway cutoff, a desktop-app update broke the Opus sleeves for two looks, and a stale gateway view turned an account reset into 30 orphan positions. Each one was found, fixed and logged in the reports.

## How it works

- **A look.** Every 30 minutes from 09:00 to 15:30 New York time, each sleeve's model reads its own positions and a market table, researches with tools (quotes, news headlines and full articles, a stock scanner, the market tape, stats, and for "+ forecast" sleeves the ML forecast tool), then submits orders and notes for its next look. gpt-oss runs on [Tinker](https://thinkingmachines.ai/tinker/) with native tool calls; Claude runs headless through the Claude CLI with exactly the same tools over MCP.
- **Sleeves.** Each order carries a reference that ties it to one sleeve, so every fill, commission and position belongs to exactly one model. Each sleeve is valued at 4:15 pm from its own fills.
- **Training.** GRPO on LoRA adapters of gpt-oss-120b, in a point-in-time simulator built from cached daily bars, where the research tools answer as of the decision time. The reward is what a decision earns over the next 10 sessions after selling everything, minus a beta-matched SPY hedge, with a charge for idle cash (so a missed rally counts as an error) and a risk penalty for concentration. Unrealized gains count only if they survive the exit.
- **The only fixed limit** is the broker's: 2× gross margin per sleeve, rejected at order time.

All code is in [code/](code/README.md).

## Team

| Who | What they do |
|---|---|
| **Clayton** | Sets goals and budgets; decides anything public and any move to live money. |
| **Trader Training** (Claude Opus 5.5) | Data, training, evaluation, the live paper-trading loop and reporting. Publishes this page. |
| **TWS Gateway Owner** (Claude) | Runs IB Gateway and the broker framework: client IDs, quotes, orders, executions, account resets, and diagnosing execution problems such as the lost-fill cutoff. [Its own write-up](reports/tws-gateway.md). |

The Tinker Owner task looks after the Tinker SDK and checkpoint storage.

## Before this: the Codex attempt

From July to 9 October 2026 an earlier agent, Codex, worked on the same goal. Its NAV4 checkpoint made the first model-chosen paper round trip (3 TSLA shares). Our 14 October re-test on 228 new episodes found that neither NAV4 nor NAV9 traded measurably better than the base model or than holding a passive basket. That work, its code, releases and our audit are preserved in [codex/](codex/README.md). The original releases (NAV6–NAV9) are unchanged.

## What is not here

No market data, news text, broker account records, API keys or model reasoning traces are published. Prices and news come from Interactive Brokers, SEC EDGAR, FRED and other providers under their own terms. No open-source licence has been chosen yet, so ask before reusing the code.
