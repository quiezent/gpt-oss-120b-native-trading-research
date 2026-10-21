# The ML forecast tool: what we built, what passed, what failed (17–19 October 2026)

*Trader Training, written 21 October 2026.*

The language models in this project decide every trade. We asked whether classical machine learning could give them better information, as one extra research tool. We built and tested about twenty models in three waves over one weekend. Each had a pre-registered bar, an out-of-sample walk-forward test from January 2025 to October 2026 with monthly retraining, and an independent auditor agent who looked for lookahead and tried to refute the result.

## Headline

- **Nothing that predicts direction passed.** Every return or direction model failed after costs.
- **Risk and cost forecasts did pass.** They became one tool, `forecast`, offered only to the "+ forecast" sleeves from 19 October.

## What ships

One call returns, for up to 20 symbols:

| Forecast | How it is built | Out-of-sample R² | Simple baseline |
|---|---|---:|---:|
| Next-session high-low range | HAR model on recent range-based volatility | 0.37 | 0.29 (20-day average range) |
| … corrected after an SEC 8-K filing | Range plus an event correction (mostly earnings) | 0.51 on event days | 0.02 uncorrected |
| Next-day move size | Recent moves plus option-implied volatility | 0.14 | 0.13 (previous model) |
| 20-day volatility | Regression on the stock's 30-day implied vol | 0.63 | 0.50 (previous LightGBM model) |
| Range over the next 60 minutes | Intraday model on 30-minute bars | 0.58 | 0.49 |
| SPY 20-day volatility | Median-calibrated SPY implied vol | 0.28 | −0.20 (trailing 20-day vol) |
| Round-trip trading cost | Spread plus square-root impact plus commission | 0.970 (log scale) | 0.962 |

It also gives a 10–90% band on the range, a liquidity tier, a position-size scale for the market, and the day's scheduled US economic releases, with Kalshi expectations as a calendar line. The tool's own description tells the model plainly that it says how much, not which way.

**Honest caveats.** The range gain is modest: a plain five-day average range does nearly as well outside the April 2025 shock. Move size is barely better than its baseline. The stock universe is today's S&P 500, so it is survivor-biased. The volatility and earnings-day forecasts are the clear wins.

## What failed

| Family | Result |
|---|---|
| LightGBM 5-day return relative to SPY (price and volume features) | Rank correlation −0.035 (t −1.0); top decile trailed bottom by 0.52% per 5 days |
| Overnight gap reversal | The only statistically real effect (t 4.5), but −9 bps a day after costs, and mostly gone by 10:00 |
| Industry and factor signals | No variant reached t 2 |
| Cross-sectional ranker | Best t 1.04, about what a placebo produces; negative after costs |
| Intraday direction (four variants) | 1.6–3 bps edge before costs against about 20 bps of costs |
| News uplift and sentiment | Failed its own bar; the gain vanished after dropping 10 dates |
| Anomaly detector, regime labels, drawdown probabilities | No skill beyond base rates |
| Chronos (time-series foundation model) range blend | Passed narrowly on paper, but served invented ranges for short histories; not shipped |
| Macro risk flags around releases | Every variant lost slightly on release mornings |

The bar for a direction model was: rank-correlation t-stat of at least 2, a positive top-minus-bottom decile spread after costs, and positive correlation in at least two of the last three half-years.

## Process notes

- Each model was pre-registered before its test: the metric, the baseline and the pass rule.
- Auditor agents reran each result, recomputed event timing and ran placebo, lag and lead tests. They found one leak, in a macro note that carried 2026 statistics into 2025 simulation days, and it was removed before shipping. They found none in what ships.
- The training simulator serves the same forecasts strictly out of sample, as of the close before each decision, so a checkpoint trained with the tool sees what it will see live.

## Does the tool help the traders?

Too early to say. After two live days the "+ forecast" sleeve is ahead of its twin for base and Opus and behind for rl3 ([week 2 report](2026-10-20-week2-days1-2.md)). The week's result, and more weeks after it, will tell.

Code: [code/forecast.py](../code/forecast.py), [code/ml.py](../code/ml.py) and the research modules in [code/ml_research/](../code/ml_research).
