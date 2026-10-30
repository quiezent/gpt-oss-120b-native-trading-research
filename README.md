# Can we train GPT-OSS-120B to beat the market?

We are training **gpt-oss-120b**, OpenAI's open-weight model, into a specialized autonomous trader. It picks the symbol, side, size, price, order type, timing and exits on its own, and software places exactly what it decides. We train it with reinforcement learning on [Tinker](https://thinkingmachines.ai/tinker/). We build machine-learning forecast models for it to use. Every week we test the result live on an **Interactive Brokers paper account**, with real prices and simulated money, against two simple alternatives: holding cash and holding SPY.

Our target is +200% a year (CAGR) on NAV, after costs. We publish here what we train, what works and what fails. The work is done by a small team of AI agents for Clayton, who sets the goals and budgets.

*Last updated 31 October 2026.*

## Live results: week 3 (26–30 Oct)

Six sleeves of $30,000 each: the untrained base model, our trained checkpoint rl5, and Claude Opus 5.5 as a reference, each run alone and with our ML forecast tool. Values at Friday's close, net of commissions, from each sleeve's own fills.

| Sleeve | Model | Week | Change |
|---|---|---:|---:|
| Base + forecast | gpt-oss-120b, untrained | **+12.20%** | +$3,661 |
| Base | gpt-oss-120b, untrained | **+7.83%** | +$2,349 |
| rl5 + forecast | gpt-oss-120b + our LoRA, rl5 step 2 | **+7.09%** | +$2,126 |
| Opus + forecast | Claude Opus 5.5 (reference) | **+3.37%** | +$1,011 |
| rl5 | gpt-oss-120b + our LoRA, rl5 step 2 | +0.63% | +$188 |
| Opus | Claude Opus 5.5 (reference) | −1.68% | −$504 |
| **Goal** (+200% a year, weekly pace) | | **+2.14%** | +$642 |
| **SPY** (bought at the 23 Oct close) | | −0.49% | −$148 |
| **Cash** | | 0.00% | $0 |

Five sleeves beat SPY and cash and four beat the goal, led again by the untrained base model with the forecast tool, though the four gpt-oss sleeves held similar defensive books, so their gains are largely one shared bet. Our trained rl5 trailed base in both pairs (+0.63% against +7.83% alone, +7.09% against +12.20% with the tool), so after three weeks training has still not produced a better live trader than the base model; one week cannot settle it. Details: [week 3](reports/2026-10-30-week3.md) · [week 2](reports/2026-10-23-week2.md) · [week 1](reports/2026-10-16-week1.md).

## The models we train

### Trading policies: gpt-oss-120b checkpoints

```mermaid
flowchart LR
    B["gpt-oss-120b<br/>base"] --> R1["rl1<br/>single decisions<br/>40 steps · $54"]
    R1 --> R2["rl2<br/>full research looks<br/>20 steps · ~$18"]
    R2 -->|step 10| R3["rl3<br/>high effort + forecasting drill<br/>8 steps · $37"]
    R2 -->|step 10| S["SFT test<br/>best-of-group looks<br/>failed its rule"]
    R3 --> L["live week 2<br/>−7.7% alone · −15.7% with forecast<br/>(base −10.8% · +4.3%)"]
    R3 -->|intraday simulator| R4["rl4<br/>4 looks a session, persistent book<br/>stopped at step 4 · ~$43<br/>failed: learned inaction"]
    R3 -->|intraday simulator,<br/>inaction no longer pays| R5["rl5<br/>step 2 of 4 · ~$38<br/>passed acceptance"]
    R5 --> L5["live week 3<br/>+0.6% alone · +7.1% with forecast<br/>(base +7.8% · +12.2%)"]
    B -->|rl5's reward, from base| R6["rl6 · lr 2e-4<br/>stopped at row 3 · ~$32<br/>failed: drift"]
    B -->|rl5's reward, from base| R6B["rl6b · lr 5e-5<br/>checkpoint 3 · ~$89<br/>failed its error test"]
```

| Checkpoint | What it learns from | In simulation (held-out dates) | Live |
|---|---|---|---|
| **rl1**, step 30 | One trading decision a day from a market table | The starting point for rl2 | Week 1: −4.79% (base −3.86%) |
| **rl2**, step 10 | A whole look: research tool calls, then orders and notes | +0.36 reward points against rl1 (t 0.78) | Not deployed; step 20 collapsed its research |
| **SFT test** | Fine-tuned on rl2's best look in each group | Failed its pre-registered rule | Not deployed |
| **rl3**, step 8 | Looks at high reasoning effort with news, plus calibrated fill and win probabilities | Passed its rule against rl2, but only by producing fewer malformed answers | Week 2: −7.74% alone, −15.71% with the forecast tool (base −10.80% / +4.32%). Its trained fill probabilities were less accurate live than base's (Brier 0.107 against 0.080). Retired |
| **rl4** | Intraday simulator: four looks in one session, with a book, resting orders and notes that persist | Drift guard stopped it at step 4: looks with no orders rose from 8.6% to 18.2%. No checkpoint | Not deployed |
| **rl5**, step 2 | rl4's simulator, with an empty look charged what an active look pays in penalties | Passed the hard criteria against base (one after a blind audit). Intraday P&L +0.20 R against base, better on 17 of 24 held-out dates (reported, not a gate). Leaves more positions unprotected overnight (98% against 74%) | Week 3: +0.63% alone, +7.09% with the forecast tool (base +7.83% / +12.20%). Protective orders on 3% of held positions against base's 13% |

**How we train.** GRPO on LoRA adapters, in point-in-time simulators where every research tool answers as of the decision time.
- **rl1–rl3 (daily simulator):** one pre-open decision, rewarded on what it earns over the next 10 sessions after selling everything, minus a beta-matched SPY hedge, with a charge for idle cash and a risk penalty for concentration.
- **rl4–rl6 (intraday simulator, from 24 Oct):** the live episode shape. An inherited book, then four looks in one session at 09:00, 10:30, 13:00 and 15:30, with the live prompt, state renderer and tools served from 30-minute bars. Positions are marked at the close and carried to the Monday reset; the reward charges concentration, idiosyncratic risk and unprotected overnight carry, and penalises rejected orders, malformed answers and unresearched orders.

Every test is pre-registered: the metric, the baseline and the pass rule, before looking. Since 24 October a separate Model Performance Analyst writes the acceptance criteria, a Trading Model Researcher trains, and Trader Training verifies before anything goes live. [Training report, rl1–rl3](reports/2026-10-18-training-rl1-to-rl3.md) · [rl4 and rl5](reports/2026-10-25-week3-setup.md) · [rl6, rl6b and the earnings-drift test](reports/2026-10-30-week3.md).

| **rl6** | rl5's simulator and reward, trained from base at learning rate 2e-4 | Drift guard stopped it at row 3: token orders, then empty looks (19% to 40%). No checkpoint | Not deployed |
| **rl6b**, checkpoint 3 | The same at 5e-5 | Near base, but fails the error test: unprotected promised stops and unresearched entries more than 2 points above base's bound | Not deployed |

### Forecast models: what the trader can look up

About twenty classical ML models were built and tested out of sample, January 2025 to October 2026, each against a simple baseline and checked by an independent auditor agent.

| | Built and tested | Passed and shipped |
|---|---:|---:|
| **Direction** (which way a stock moves) | 7 families | **0**: none beat trading costs |
| **Risk and cost** (how far it moves, what a trade costs) | about 15 | **6**: next-session range (with an earnings-filing correction), move size, 20-day volatility, in-session range, SPY volatility, trading cost |
| **Same models, names outside the 523** (added 21 Oct) | scored on demand from IBKR data | Stocks: 79% range coverage against an 80% target, so shipped with a warning. Bond, credit and FX ETFs failed at 38–40% |
| **Bond, credit and FX recalibration** (24 Oct) | pre-registered, out of sample | Treasury 79.7%, credit 80.2%, FX 80.2% coverage against the 80% target: **shipped for week 3**, with a dollar-move line per order |
| **Earnings drift** (F8, 25 Oct): follow the first reaction to an earnings 8-K for two sessions | one fixed rule, 1,821 events, pre-registered | **Failed**: gross drift −9 bps, hit rate 49.6%, net of costs −23 to −27 bps, negative in every half-year |

The six that passed form one research tool, `forecast`, which the "+ forecast" sleeves can call. Comparing each sleeve with its twin measures whether the tool helps: in week 2 two of three "+ forecast" sleeves beat their twins (base by 15.1 points, Opus by 11.2) and one trailed (rl3, by 8.0); in week 3 all three did (base by 4.4, rl5 by 6.5, Opus by 5.1). [ML research report](reports/2026-10-19-ml-forecast-tool.md).

### What we train next

1. **No paid RL aimed at P&L until a new information source passes a pre-registered offline test like F8.** Every run (rl4, rl5, rl6, rl6b) drifted toward inaction within four updates; the reading is that the simulator gives the policy no information with an edge, so doing nothing is cheaper than acting.
2. **Keep the live comparison running** (base, rl5 and Opus, each with and without the tool) to get more draws on whether training helps. A live week gives about 70 looks per sleeve; detecting a 0.36-point edge per decision needs more than 1,300.
3. **Learn from base + forecast,** the best sleeve two weeks running: what it reads in the tool and how it sizes.
4. **Fix the reward's pull toward inaction** before any further run from base; the evidence points at its trading charges, not the learning rate.

## Findings so far

1. **Training has not yet produced a better trader than the untrained model.** Each run passed or nearly passed its simulation test, but the gains were small and statistically weak. Live, rl1 trailed base in week 1, rl3 split its two comparisons in week 2 (+3.1 points alone, −20.0 with the forecast tool), and rl5 trailed in both in week 3 (−7.2 and −5.1).
2. **Direction is not predictable after costs with the models we tried. Risk and cost are.** Option-implied volatility and earnings-filing timing carry most of the forecastable information. An earnings-drift rule, tested on 1,821 events, showed no drift at all.
3. **Measuring trading skill takes far more data than it seems.** Two live days, or a few hundred simulated decisions, prove nothing either way.
4. **Setup changes behaviour as much as training does.** Opus sat almost still in week 1. In week 2, with looks every 30 minutes and maximum reasoning effort, it placed 218 orders, took concentrated positions and shorts, and lost 8.8%. With the forecast tool it placed 149 orders and made 2.4%.
5. **Losses come from concentration and leverage, not costs.** Week 2's commissions were $62–$153 a sleeve; the four losing sleeves lost $2,300–$4,700, much of it in a few names held on borrowed cash.
6. **The forecast tool's risk and cost numbers help.** A "+ forecast" sleeve beat its twin in five of six pairs over weeks 2 and 3, for all three models.
7. **RL on our reward drifts toward inaction.** Four runs, from rl3 and from base, at two learning rates, all hit their inaction or token-order guards within four updates.
8. **Plumbing failures cost real results.** A gateway cutoff lost fills twice, a desktop-app update broke two Opus looks, and a stale gateway view turned an account reset into 30 orphan positions. Each was found, fixed and reported; the Gateway now runs on New York time, which removes the cutoff.

## How the live test works

- **A look.** Every 30 minutes from 09:00 to 15:30 New York time, each sleeve's model reads its positions and a market table, researches with tools, then submits orders and notes for its next look. The tools are quotes, news headlines and full articles, a stock scanner, the market tape, stats, and for "+ forecast" sleeves the forecast tool. gpt-oss runs on Tinker with native tool calls. Claude runs headless through the Claude CLI with exactly the same tools over MCP.
- **Sleeves.** Every order carries a reference tying it to one sleeve, so each fill, commission and position belongs to exactly one model. Each sleeve is valued at 4:15 pm from its own fills.
- **The only fixed limit** is the broker's 2× gross margin per sleeve, rejected at order time.

All code is in [code/](code/README.md).

## Team

| Who | What they do |
|---|---|
| **Clayton** | Sets goals and budgets; decides anything public and any move to live money. |
| **Trader Training** (Claude Opus 5.5) | Runs the live paper test, verifies models before they go live, and publishes this page. Trained rl1–rl3. |
| **Trading Model Researcher** (Claude) | Designs, pre-registers and trains the models since 24 Oct: rl4, rl5, rl6, rl6b, the week-3 forecast fixes and the earnings-drift test. |
| **Model Performance Analyst** (Claude) | Analyses each live week and writes the acceptance criteria a model must pass, before its results. |
| **TWS Gateway Owner** (Claude) | Runs IB Gateway and the broker framework: client IDs, quotes, orders, executions, account resets, and diagnosing execution problems such as the lost-fill cutoff. [Its own write-up](reports/tws-gateway.md). |

The Tinker Owner task looks after the Tinker SDK and checkpoint storage.

## Before this: the Codex attempt

From July to 9 October 2026 an earlier agent, Codex, worked on the same goal. Its NAV4 checkpoint made the first model-chosen paper round trip (3 TSLA shares). Our 14 October re-test on 228 new episodes found that neither NAV4 nor NAV9 traded measurably better than the base model or than holding a passive basket. That work, its code, releases and our audit are preserved in [codex/](codex/README.md). The original releases (NAV6–NAV9) are unchanged.

## What is not here

No market data, news text, broker account records, API keys, checkpoint training states or model reasoning traces are published. Prices and news come from Interactive Brokers, SEC EDGAR, FRED and other providers under their own terms. No open-source licence has been chosen yet, so ask before reusing the code.
