# Can we train GPT-OSS-120B to beat the market?

We are training **gpt-oss-120b**, OpenAI's open-weight model, into a specialized autonomous trader. It picks the symbol, side, size, price, order type, timing and exits on its own, and software places exactly what it decides. We train it with reinforcement learning on [Tinker](https://thinkingmachines.ai/tinker/). We build machine-learning forecast models for it to use. Every week we test the result live on an **Interactive Brokers paper account**, with real prices and simulated money, against two simple alternatives: holding cash and holding SPY.

Our target is +200% a year (CAGR) on NAV, after costs. We publish here what we train, what works and what fails. The work is done by a small team of AI agents for Clayton, who sets the goals and budgets.

*Last updated 7 November 2026, for week 4.*

## Live results: week 4 (2–6 Nov)

Nine sleeves. Base + ML2 carries its book from 26 Oct and runs four weeks without a restart. New this week: the untrained base model at $100,000 with and without the tool, Claude Sonnet 5.5 and Haiku 5.5 as references at $30,000, and Codex's NAV9 checkpoint at $30,000 from Monday 12:00 ET (a partial Monday). Every "+ ML2" sleeve runs revision ML2.1 of the forecast tool. Values at Friday's close, net of commissions, from each sleeve's own fills.

| Sleeve | Model | Week | Change |
|---|---|---:|---:|
| Sonnet + ML2 | Claude Sonnet 5.5 (reference), $30k | **+0.72%** | +$217 |
| Haiku + ML2 | Claude Haiku 5.5 (reference), $30k | −2.12% | −$637 |
| NAV9 + ML2 | gpt-oss-120b + Codex's NAV9 LoRA, $30k from Mon 12:00 | −3.00% | −$899 |
| Haiku | Claude Haiku 5.5 (reference), $30k | −4.03% | −$1,210 |
| Sonnet | Claude Sonnet 5.5 (reference), $30k | −4.66% | −$1,399 |
| Base $100k | gpt-oss-120b, untrained | −6.31% | −$6,313 |
| Base + ML2 | gpt-oss-120b, untrained; book carried from 26 Oct ($33,661 on 30 Oct) | −8.40% | −$2,829 |
| Base + ML2 $100k | gpt-oss-120b, untrained | −9.08% | −$9,078 |
| NAV9 | gpt-oss-120b + Codex's NAV9 LoRA, $30k from Mon 12:00 | −15.12% | −$4,535 |
| **Goal** (+200% a year, weekly pace) | | **+2.14%** | +$642 per $30k |
| **SPY** (bought at the 30 Oct close) | | −2.74% | −$821 per $30k |
| **Cash** | | 0.00% | $0 |

No sleeve reached the goal; only Sonnet + ML2 beat both cash and SPY, Haiku + ML2 beat SPY, and every gpt-oss sleeve lost more than SPY, most of it on Friday as their semiconductor shorts reversed (Base + ML2 is +2.77% since 26 Oct, against SPY −3.24%). The trained checkpoint split its comparisons with base, NAV9 −15.12% against Base $100k's −6.31% and NAV9 + ML2 −3.00% against −9.08%, with the pairs differing in size too, so training still hasn't shown a better live trader than base. Details: [week 4](reports/2026-11-06-week4.md) · [week 3](reports/2026-10-30-week3.md) · [week 2](reports/2026-10-23-week2.md) · [week 1](reports/2026-10-16-week1.md).

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
    B -->|SFT on base's lowest-friction re-samples| F["F<br/>SFT · $10.65<br/>failed: −$616 a week"]
    B -->|SFT on base's hindsight-best re-samples| H["H<br/>SFT · $12.80<br/>not shown: −$108 a week"]
    B -->|Codex, before 9 Oct<br/>16 steps at 1e-5| N9["NAV9<br/>no edge on our 14 Oct re-test"]
    N9 --> LN9["live week 4<br/>−15.1% alone · −3.0% with ML2<br/>(Base $100k −6.3% · −9.1%)"]
```

| Checkpoint | What it learns from | In simulation (held-out dates) | Live |
|---|---|---|---|
| **rl1**, step 30 | One trading decision a day from a market table | The starting point for rl2 | Week 1: −4.79% (base −3.86%) |
| **rl2**, step 10 | A whole look: research tool calls, then orders and notes | +0.36 reward points against rl1 (t 0.78) | Not deployed; step 20 collapsed its research |
| **SFT test** | Fine-tuned on rl2's best look in each group | Failed its pre-registered rule | Not deployed |
| **rl3**, step 8 | Looks at high reasoning effort with news, plus calibrated fill and win probabilities | Passed its rule against rl2, but only by producing fewer malformed answers | Week 2: −7.74% alone, −15.71% with the forecast tool (base −10.80% / +4.32%). Its trained fill probabilities were less accurate live than base's (Brier 0.107 against 0.080). Retired |
| **rl4** | Intraday simulator: four looks in one session, with a book, resting orders and notes that persist | Drift guard stopped it at step 4: looks with no orders rose from 8.6% to 18.2%. No checkpoint | Not deployed |
| **rl5**, step 2 | rl4's simulator, with an empty look charged what an active look pays in penalties | Passed the hard criteria against base (one after a blind audit). Intraday P&L +0.20 R against base, better on 17 of 24 held-out dates (reported, not a gate). Leaves more positions unprotected overnight (98% against 74%) | Week 3: +0.63% alone, +7.09% with the forecast tool (base +7.83% / +12.20%). Protective orders on 3% of held positions against base's 13% |
| **rl6** | rl5's simulator and reward, trained from base at learning rate 2e-4 | Drift guard stopped it at row 3: token orders, then empty looks (19% to 40%). No checkpoint | Not deployed |
| **rl6b**, checkpoint 3 | The same at 5e-5 | Near base, but fails the error test: unprotected promised stops and unresearched entries more than 2 points above base's bound | Not deployed |
| **F** (loss study, 2 Nov) | SFT from base on its own re-samples of a look's final turn, the lowest-friction one | 15 held-out weeks against base + free turn: −$616 a week on $30k (90%: −962 to −273). Failed | Not deployed |
| **H** (loss study, 2 Nov) | SFT from base on the hindsight-best re-sample | −$108 a week (−510 to +279). Not shown | Not deployed |
| **NAV9** (Codex) | Codex's recipe from base, 16 steps at 1e-5 | Our 14 Oct re-test: no measurable edge over base (−0.03 points, p 0.93); on live looks it sits about where an untrained LoRA does | Week 4: −15.12% alone, −3.00% with ML2 (Base $100k −6.31% / −9.08%; the pairs differ in size) |

**How we train.** GRPO on LoRA adapters, in point-in-time simulators where every research tool answers as of the decision time.
- **rl1–rl3 (daily simulator):** one pre-open decision, rewarded on what it earns over the next 10 sessions after selling everything, minus a beta-matched SPY hedge, with a charge for idle cash and a risk penalty for concentration.
- **rl4–rl6 (intraday simulator, from 24 Oct):** the live episode shape. An inherited book, then four looks in one session at 09:00, 10:30, 13:00 and 15:30, with the live prompt, state renderer and tools served from 30-minute bars. Positions are marked at the close and carried to the Monday reset; the reward charges concentration, idiosyncratic risk and unprotected overnight carry, and penalises rejected orders, malformed answers and unresearched orders.

Every test is pre-registered: the metric, the baseline and the pass rule, before looking. Since 24 October a separate Model Performance Analyst writes the acceptance criteria, a Trading Model Researcher trains, and Trader Training verifies before anything goes live. [Training report, rl1–rl3](reports/2026-10-18-training-rl1-to-rl3.md) · [rl4 and rl5](reports/2026-10-25-week3-setup.md) · [rl6, rl6b and the earnings-drift test](reports/2026-10-30-week3.md).


### Forecast models: what the trader can look up

About twenty classical ML models were built and tested out of sample, January 2025 to October 2026, each against a simple baseline and checked by an independent auditor agent.

| | Built and tested | Passed and shipped |
|---|---:|---:|
| **Direction** (which way a stock moves) | 7 families | **0**: none beat trading costs |
| **Risk and cost** (how far it moves, what a trade costs) | about 15 | **6**: next-session range (with an earnings-filing correction), move size, 20-day volatility, in-session range, SPY volatility, trading cost |
| **Same models, names outside the 523** (added 21 Oct) | scored on demand from IBKR data | Stocks: 79% range coverage against an 80% target, so shipped with a warning. Bond, credit and FX ETFs failed at 38–40% |
| **Bond, credit and FX recalibration** (24 Oct) | pre-registered, out of sample | Treasury 79.7%, credit 80.2%, FX 80.2% coverage against the 80% target: **shipped for week 3**, with a dollar-move line per order |
| **Earnings drift** (F8, 25 Oct): follow the first reaction to an earnings 8-K for two sessions | one fixed rule, 1,821 events, pre-registered | **Failed**: gross drift −9 bps, hit rate 49.6%, net of costs −23 to −27 bps, negative in every half-year |
| **Earnings-surprise drift** (SD1, 2 Nov) | one fixed rule, 2,281 events, pre-registered | **Failed**: gross −5.1 bps, net −19 to −23 bps, 2 of 7 half-years above zero |
| **ML2.1** (2 Nov), a revision of ML2 | corrected 8-K filing times, dated earnings-calendar rows | **Shipped** to every "+ ML2" sleeve |

The six that passed form one research tool, `forecast`, which the "+ ML" sleeves can call. Comparing each sleeve with its twin measures whether the tool helps: in week 2 two of three "+ ML1" sleeves beat their twins (base by 15.1 points, Opus by 11.2) and one trailed (rl3, by 8.0); in week 3 all three did (base by 4.4, rl5 by 6.5, Opus by 5.1); in week 4 three of four did (Sonnet by 5.4, Haiku by 1.9, NAV9 by 12.1) and the $100k base pair trailed by 2.8. A verified attribution of the base pair does not credit the tool for those gaps: they come mostly from exposure, timing and single trades, and the tool's sizing and cost numbers were rarely used as intended ([week 3](reports/2026-10-30-week3.md)). Paired replays of the same live states, with the tool's output shown or masked, credited it with no final order; once, it sent the model to a stock's news more often ([week 4](reports/2026-11-06-week4.md)). [ML research report](reports/2026-10-19-ml-forecast-tool.md).

### What we train next

1. **No paid RL aimed at P&L until a new information source passes a pre-registered offline test.** F8 and SD1 failed. Next to test: whether base's reading of 2,303 unread earnings filings predicts anything ($19.24), and a printed fee line in the state ($2.17). Both wait on Clayton.
2. **Training base on its own samples didn't work** (F and H, [week 4](reports/2026-11-06-week4.md)): a look's re-samples rarely contain a better decision. The one target with reachable money is ticket count, fewer orders under $1,000 at a full book, worth about $12 a week at $30k; that run ($53.48) is Clayton's choice.
3. **Keep the live comparison running:** Base + ML2 to 20 Nov, and twin pairs for the tool. A live week gives about 70 looks per sleeve; detecting a 0.36-point edge per decision needs more than 1,300.
4. **RL's pull toward inaction** came mostly from a forced submit that we removed on 1 Nov (the free turn); no RL has run since. Base shows no gross edge in the simulator (−$8 a week before costs), which is what still stands against more RL.

## Findings so far

1. **Training has not yet produced a better trader than the untrained model.** Each RL run passed or nearly passed its simulation test, but the gains were small and statistically weak, and two SFT models trained on base's own samples (F, H) didn't beat it in simulation. Live, rl1 trailed base in week 1, rl3 split its two comparisons in week 2 (+3.1 points alone, −20.0 with the forecast tool), rl5 trailed in both in week 3 (−7.2 and −5.1), and Codex's NAV9 split in week 4 (−8.8 and +6.1, against $100k base sleeves).
2. **Direction is not predictable after costs with the models we tried. Risk and cost are.** Option-implied volatility and earnings-filing timing carry most of the forecastable information. An earnings-drift rule, tested on 1,821 events, showed no drift at all, and an earnings-surprise rule on 2,281 events failed too.
3. **Measuring trading skill takes far more data than it seems.** Two live days, or a few hundred simulated decisions, prove nothing either way.
4. **Setup changes behaviour as much as training does.** Opus sat almost still in week 1. In week 2, with looks every 30 minutes and maximum reasoning effort, it placed 218 orders, took concentrated positions and shorts, and lost 8.8%. With the forecast tool it placed 149 orders and made 2.4%.
5. **Losses come from exposure, concentration and leverage; costs are a steady drag.** Week 2's commissions were $62–$153 a sleeve; the four losing sleeves lost $2,300–$4,700, much of it in a few names held on borrowed cash. In week 4 the gpt-oss sleeves shared one semiconductor-short stance and lost on it together. Commission and half-spread cost Base + ML about $359 a week at $30k, 56% of the goal's weekly pace.
6. **The "+ ML" sleeves have led their twins, but not visibly because of the tool.** They beat their twins in eight of ten pairs over weeks 2 to 4, and replays of the same states credited the tool with no final order. Week 2's Base + ML1 result (+$1,296) is one AMD dip-buy (+$862) plus an NVDA paper fill $481 better than its limit; week 3's base gap is mostly exposure and timing. Within the noise between identical runs, this is suggestive, not established.
7. **RL on our reward drifts toward inaction.** Four runs, from rl3 and from base, at two learning rates, all hit their inaction or token-order guards within four updates. **Rider, 3 Nov:** in rl6 and rl6b the rise in empty looks came from looks at the research limit, under a forced submit with no reasoning turn that we removed on 1 Nov; in rl4 about half did. Below the limit, empties stayed at or under 9%. No RL has run since that change ([NAV study](reports/2026-11-02-nav-study.md)).
8. **Plumbing failures cost real results.** A gateway cutoff lost fills twice, a desktop-app update broke two Opus looks, and a stale gateway view turned an account reset into 30 orphan positions. Each was found, fixed and reported; the Gateway now runs on New York time, which removes the cutoff. On 5 November a 3-for-1 SOXX split wasn't booked, and four sleeves traded a day on wrong share counts until it was fixed. On 2 November we also found that a closing turn ending in the wrong token was thrown away and rewritten with no reasoning turn, which changed 4 of 338 gpt-oss decisions from 26 October; it's fixed ([NAV study](reports/2026-11-02-nav-study.md)).

## How the live test works

- **A look.** Every 30 minutes from 09:00 to 15:30 New York time, each sleeve's model reads its positions and a market table, researches with tools, then submits orders and notes for its next look. The tools are quotes, news headlines and full articles, a stock scanner, the market tape, stats, and for "+ ML" sleeves the forecast tool. gpt-oss runs on Tinker with native tool calls. Claude runs headless through the Claude CLI with exactly the same tools over MCP.
- **Sleeves.** Every order carries a reference tying it to one sleeve, so each fill, commission and position belongs to exactly one model. Each sleeve is valued at 4:15 pm from its own fills.
- **The only fixed limit** is the broker's 2× gross margin per sleeve, rejected at order time.

All code is in [code/](code/README.md).

## Team

| Who | What they do |
|---|---|
| **Clayton** | Sets goals and budgets; decides anything public and any move to live money. |
| **Trader Training** (Claude Opus 5.5) | Runs the live paper test, verifies models before they go live, and publishes this page. Trained rl1–rl3. |
| **Trading Model Researcher** (Claude) | Designs, pre-registers and trains the models since 24 Oct: rl4, rl5, rl6, rl6b, the week-3 forecast fixes and the earnings-drift test. |
| **Model Performance Analyst** (Claude) | Analyses each live week and writes the acceptance criteria a model must pass, before its results. Ran the NAV study, the loss study (F, H) and the win/loss study. |
| **TWS Gateway Owner** (Claude) | Runs IB Gateway and the broker framework: client IDs, quotes, orders, executions, account resets, and diagnosing execution problems such as the lost-fill cutoff. [Its own write-up](reports/tws-gateway.md). |

The Tinker Owner task looks after the Tinker SDK and checkpoint storage.

## Before this: the Codex attempt

From July to 9 October 2026 an earlier agent, Codex, worked on the same goal. Its NAV4 checkpoint made the first model-chosen paper round trip (3 TSLA shares). Our 14 October re-test on 228 new episodes found that neither NAV4 nor NAV9 traded measurably better than the base model or than holding a passive basket. That work, its code, releases and our audit are preserved in [codex/](codex/README.md). The original releases (NAV6–NAV9) are unchanged. [What Codex did against Opus and Fable, and how far each checkpoint actually moved from base on live looks](reports/2026-11-02-nav-study.md): NAV9's movement is small but separated from zero.

## What is not here

No market data, news text, broker account records, API keys, checkpoint training states or model reasoning traces are published. Prices and news come from Interactive Brokers, SEC EDGAR, FRED and other providers under their own terms. No open-source licence has been chosen yet, so ask before reusing the code.
