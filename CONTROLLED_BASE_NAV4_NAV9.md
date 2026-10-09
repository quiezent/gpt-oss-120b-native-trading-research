# Controlled base versus NAV4 versus NAV9 comparison

2026-10-09

I ran 36 inference-only trajectories to compare the same GPT-OSS-120B base model with my frozen NAV4 and NAV9 checkpoints. NAV9 achieved the highest mean across these 12 controlled cases, while NAV4 led at 63 sessions. Every model had a negative five-session mean. Most trading cases retained inventory, so I have not established consistent autonomous exits or prospective profit.

## What I held constant

I used the same AAPL, MSFT and SPY daily OHLCV source, starting dates of May 1, June 1 and July 1, 2026, and horizons of 1, 5, 20 and 63 trading sessions. Each trajectory began in a separate $10,000 cash-only book with identical initial observations and prompt tokens. Subsequent account and order observations followed that model's own actions; the paired historical market path stayed the same.

Every model used the same `functions.pa_tws` simulator contract, at most 12 decisions, a 32,768-token joint context and an unchanged 8,192-token output reserve. Temperature was 0, seed 42, with one rollout per model/case and balanced rotating arm order. I used generated actions and arguments unchanged, without investment routing, action repair, automatic prompt trimming, sampled-call retries or forced liquidation.

Orders filled at the next session's open with a 7.5bps adverse adjustment and commission of `max($1, $0.005/share)` per fill. Token prices were identical: $0.33/M input and $0.84/M output, with the same maximum $2.158755840 sampling allowance per arm. Equal ceilings and rates do not imply equal actual token usage or spend.

## Returns after simulated trading costs

Each horizon row is the arithmetic mean of three starting-date cases. The overall row averages all 12 cases; it is not a compounded portfolio return.

| Horizon | Base | NAV4 | NAV9 |
| --- | ---: | ---: | ---: |
| 1 session | +0.0000% | +0.0000% | +0.0000% |
| 5 sessions | -0.3028% | -0.3695% | -0.5344% |
| 20 sessions | +2.4764% | +2.2018% | +4.0762% |
| 63 sessions | +3.8868% | +8.0993% | +7.8466% |
| All 12 cases | +1.5151% | +2.4829% | +2.8471% |

| Start | Horizon | Base | NAV4 | NAV9 |
| --- | ---: | ---: | ---: | ---: |
| 2026-05-01 | 1 | +0.0000% | +0.0000% | +0.0000% |
| 2026-05-01 | 5 | +1.5647% | +0.1162% | +1.7534% |
| 2026-05-01 | 20 | +3.2315% | +3.2315% | +5.1558% |
| 2026-05-01 | 63 | +9.0163% | +5.1072% | +7.9967% |
| 2026-06-01 | 1 | +0.0000% | +0.0000% | +0.0000% |
| 2026-06-01 | 5 | -2.6108% | -1.2247% | -3.3836% |
| 2026-06-01 | 20 | -4.3613% | -3.7123% | -2.0819% |
| 2026-06-01 | 63 | +0.2745% | +6.8604% | +4.0928% |
| 2026-07-01 | 1 | +0.0000% | +0.0000% | +0.0000% |
| 2026-07-01 | 5 | +0.1376% | +0.0000% | +0.0270% |
| 2026-07-01 | 20 | +8.5591% | +7.0863% | +9.1546% |
| 2026-07-01 | 63 | +2.3696% | +12.3302% | +11.4503% |

NAV9 versus base: **6 wins, 3 losses, 3 ties**. NAV9 versus NAV4: **6 wins, 3 losses, 3 ties**. NAV4 versus base: **4 wins, 4 losses, 4 ties**. I retain cash-only cases and every decision-limit or refusal outcome.

## Tool competence, closure and costs

| Model | FINAL / decision limit | BUY / SELL fills | Traded and closed | Open inventory cases | Cash-only cases | Samples | Estimated inference cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BASE | 8 / 4 | 23 / 4 | 2 | 7 | 3 | 82 | $0.31394505 |
| NAV4 | 9 / 3 | 27 / 3 | 0 | 8 | 4 | 81 | $0.2974155 |
| NAV9 | 8 / 4 | 29 / 3 | 1 | 8 | 3 | 97 | $0.32884146 |

All three models generated executable simulated trades. That corrects any interpretation that NAV4 is uniquely capable of trading under a common interface. It does not establish equivalent competence through the actual native CLI or IBKR broker.

The host returned 13 refusals: 12 risk/rate refusals and one base order-reference format refusal (`pa:spy1` was too short). There were no Harmony parse failures, context stops or provider failures. Refusals remained visible to the model; I did not repair their arguments.

`FINAL` is protocol completion, not portfolio closure. Five fills, two base and three NAV4, occurred after model stop when the evaluator advanced historical time. Terminal NAV includes marked remaining inventory. All terminal open-order counts were zero, including DAY-order expiry; this is not proof of model-directed cancellation or a complete broker lifecycle.

I captured and accounted all **260 samples** and **265 total requests**: **2,633,637 input tokens** and **84,645 output tokens**. The sampling estimate is **$0.94020201**, not a provider invoice. The full $7 reservation remains retained; I assert no refund or allowance release. The owned provider session closed, the operation finished once, and independent review matched the saved action-ledger replay, fill costs, cash, inventory, NAV and paired statistics.

## What this changes and what remains unproved

I now have a comparison that holds the interface and resource ceilings constant. The checkpoints change the trade behavior and horizon results, but neither trained model uniformly beats base. NAV9 leads overall and at 20 sessions; NAV4 leads at 63; base has the least-negative five-session mean. None produced a positive one-session result.

These are overlapping, development-exposed windows with one rollout per arm/case, not blind validation or statistical superiority. Daily simulated prices and friction omit news, live bid/ask, intraday fills, dividends and actual broker execution. Positive autonomous NAV over days through three months remains unproved. This inference-only comparison produced no broker orders, training updates or new checkpoint.

This is a separate experiment from the earlier NAV9 versus NAV8 training evaluation. Its improved interface completion and marked returns do not replace that earlier failure record. NAV9 remains the latest actual trained checkpoint described in the [project README](README.md).

[Exact sanitized metrics and model identities](evidence/controlled_base_nav4_nav9_20261009.json) · [Sanitized independent audit](evidence/controlled_base_nav4_nav9_20261009_audit.json)
