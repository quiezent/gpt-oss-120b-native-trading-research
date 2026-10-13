# NAV re-test: method and full results

*Claude, 14 October 2026. The short version is in [README.md](README.md).*

## Method
- **Harness.** [code/p1.py](code/p1.py) re-implements Codex's P1 simulator interface; [code/validate_p1.py](code/validate_p1.py) checked it against Codex's saved runs (12/12 initial prompts token-identical, 235/235 tool results byte-identical, step-to-step prompt growth identical). [code/nav_runner.py](code/nav_runner.py) runs the episodes: strict Harmony parsing with no repair, the model's analysis tokens kept in context, 12 decisions, a 32,768-token context with the full 8,192-token output allowance, as in Codex's controlled comparison.
- **Models.** Base `openai/gpt-oss-120b`, NAV4 and NAV9 (sampler paths in the README).
- **World.** AAPL, MSFT, SPY daily bars (IBKR TRADES, split-adjusted, not dividend-adjusted; identical to Codex's data on all 494 overlapping sessions). $10,000 cash; long-only DAY limit orders filled only at the next open, 7.5 bps against the trader; commission max($1, $0.005/share); $2,500 per order, $5,000 per symbol, $9,000 gross, $500 minimum cash, 4 orders per session.
- **Windows.** Starts every 10 sessions from 2025-11-10 (after every NAV training outcome) with horizons of 20 sessions (21 starts) and 63 sessions (17 starts), ending by 2026-10-12. That is 38 windows.
- **Sampling.** Temperature 1 (the training temperature), seeds 1000 and 1001, identical across models: 76 episodes per model, 228 in all.
- **Scoring.** As in training: marked NAV at the horizon, after the evaluator advances any sessions the model didn't reach. I also report the return after selling everything at the last close with the same costs, realized P&L, and how many episodes ended holding positions.
- **Benchmarks.** Cash (0% in this simulator); $2,500 of SPY bought at the start with the rest in cash; and a passive basket ($2,500 limit buy of each of the three symbols at the start, limit = start close x 1.02, about 75% of the account, held). Both run through the same simulator. For reference, SPY's own close-to-close return over the same windows averaged +2.58% (+1.23% at 20 sessions, +4.26% at 63), which the simulator's limits ($5,000 per symbol, $9,000 gross) would not allow.
- **Statistics.** Each window's two seeds are averaged, then models are paired by window. p values are two-sided sign-flip permutation tests; intervals are 95% bootstrap intervals that resample start dates (windows that share a start move together).

## Results

| | mean marked % | mean liquidated % | realized $/ep | closed flat | still holding | FINAL / STEP / MALF / CTX | cost $ |
|---|---:|---:|---:|---:|---:|---|---:|
| cash | +0.00 | | | | | | |
| passive basket | +1.26 | | | | | | |
| $2,500 of SPY, rest in cash | +0.36 | | | | | | |
| base | +0.92 | +0.88 | -1.4 | 13/76 | 54/76 | 43 / 33 / 0 / 0 | 2.39 |
| nav4 | +0.61 | +0.57 | -2.8 | 6/76 | 58/76 | 42 / 34 / 0 / 0 | 2.39 |
| nav9 | +0.89 | +0.85 | +5.1 | 15/76 | 49/76 | 43 / 31 / 2 / 0 | 2.51 |

Paired by case (seed-averaged), cluster bootstrap over start dates:

| comparison | mean diff pp | wins / losses | sign p | permutation p | 95% CI |
|---|---:|---|---:|---:|---|
| base - passive basket | -0.34 | 19 / 18 | 1.000 | 0.517 | [-1.48, +0.69] |
| nav4 - passive basket | -0.65 | 14 / 24 | 0.143 | 0.176 | [-1.76, +0.39] |
| nav9 - passive basket | -0.37 | 19 / 19 | 1.000 | 0.407 | [-1.39, +0.58] |
| nav4 - base | -0.31 | 17 / 19 | 0.868 | 0.451 | [-1.27, +0.62] |
| nav9 - base | -0.03 | 17 / 20 | 0.743 | 0.932 | [-0.76, +0.69] |
| nav9 - nav4 | +0.28 | 19 / 18 | 1.000 | 0.503 | [-0.56, +1.13] |

By horizon (mean marked %):

| horizon | base | nav4 | nav9 | passive basket | $2,500 SPY |
|---|---|---|---|---|---|
| 20 | +0.44 | -0.07 | +0.33 | +0.59 | +0.11 |
| 63 | +1.52 | +1.45 | +1.59 | +2.09 | +0.68 |

Fit of each model's episode returns to the passive basket's return in the same window (slope, R²): base 0.4, 0.36; NAV4 0.51, 0.46; NAV9 0.5, 0.46.

Windows overlap and horizons are nested, so the permutation p values are optimistic; the bootstrap resamples start dates but not neighbouring overlap. Realized $ per episode medians: base −1.5, NAV4 −2.0, NAV9 −1.0 (NAV9's mean is lifted by two episodes).

Cost columns are upper bounds at list prices with no prefix caching.

## Reproduce
Fetch IBKR daily TRADES bars for AAPL, MSFT and SPY into `data/trades/` (any source with the same bars works; volume is divided by 40 to match the units the models were trained on), then:

```bash
python code/nav_runner.py replay --horizons 20 63 --every 10 --seeds 2 --tag oos1
python code/analyze_nav.py oos1
```
