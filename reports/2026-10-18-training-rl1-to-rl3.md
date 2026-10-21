# Training gpt-oss-120b to trade: rl1, rl2 and rl3 (14–18 October 2026)

*Trader Training, written 21 October 2026.*

All three runs are LoRA adapters on gpt-oss-120b, trained with GRPO on [Tinker](https://thinkingmachines.ai/tinker/). They learn in a simulator replayed from cached IBKR daily bars on a point-in-time stock universe.

## The reward

A decision is scored on what it earns over the next 10 sessions:

- every position is **sold at the open of session 10**, with costs, so unrealized gains count only if they survive the exit;
- minus a **shadow SPY hedge** sized by each holding's beta, which removes the market's move;
- plus a **market premium on net exposure** (about 7.6% a year), so idle cash gives something up and a missed rally is an error;
- minus a **risk penalty** on the book's variance, including correlated holdings, so concentration costs something;
- minus a **penalty for malformed answers** and, from rl2, for orders rejected at the 2× margin limit.

This replaced the earlier Codex reward (marked NAV at the end of an episode), which credited open positions and taught its models to buy and rarely exit.

## The runs

| Run | Started from | What a sample is | Steps | Tinker cost | Result |
|---|---|---|---:|---:|---|
| **rl1** | base | One daily decision from a market table | 40 | $54.19 | Step 30 went live in week 1 |
| **rl2** | rl1 step 40 | A full research look: tool calls, then orders and notes; half the groups also get the forecast tool | 20 | about $18 | Step 10: +0.36 points against rl1 on 24 held-out dates (t 0.78). Step 20: research collapsed, with one −522 tail |
| **SFT test** | rl2 step 10 | Rejection-sampling fine-tune on the best look of each group (208 kept of 240) | 26 | about $12 | Failed its pre-registered rule; rl2 step 10 stayed |
| **rl3** | rl2 step 10 | Research looks at high reasoning effort, with news, plus a forecasting drill that asks for calibrated fill and win probabilities | 8 | $36.97 | Passed its pre-registered rule against rl2 step 10; went live in week 2 |

## What rl3's pass means

rl3 met its pre-registered rule: drill score +0.027 (t 2.17) and looks +0.15 (t 0.47, enough for non-inferiority), with research behaviour inside its bounds. But:

- the whole drill gain came from fewer malformed or cut-off answers, not better forecasts (parsed-answer score −0.003);
- both models scored below the trivial per-bucket base rate, and their fill and win probabilities don't beat constants;
- rl3 does end fewer looks without a decision (5.5% against 7.3%).

So rl3 is a slightly more reliable rl2, not a better forecaster or trader.

## Why progress is hard to see

- **Statistical power.** From the simulator's variance, the standard error is about 0.46 reward points per 192 looks. Detecting a +0.36-point edge needs more than 1,300 looks per arm. Each of our evaluations used far fewer.
- **Effort matters.** At high reasoning effort rl2 makes about 10 research calls a look against 5 at low effort, costs about $0.015–0.02 a look against $0.005, and takes larger positions.
- **Simulator and live differ.** The simulator has daily bars and no live news; a news simulator was built but not used for training, because the model never called `article` in it. Closing that gap comes first.

## Budget

Clayton set a $400 Tinker budget. About $142 had been spent by 18 October: rl1 $54, rl2 about $18, the SFT test about $12, rl3 with its evaluation about $52, plus smaller runs. Live gpt-oss looks at high effort cost about $0.015 each, about $4 a day for four sleeves.

## Rules we keep

- Pre-register every training test: the metric, the baseline and the pass rule, before looking.
- Never compare Opus in the simulator on dates before July 2026. Its training data runs to June 2026, so it may know what happened.

Code: [code/train.py](../code/train.py) (rl1), [code/train2.py](../code/train2.py) (rl2, rl3), [code/reward.py](../code/reward.py), [code/simgw.py](../code/simgw.py), [code/eval2.py](../code/eval2.py), [code/sft2.py](../code/sft2.py).
