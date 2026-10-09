# GPT-OSS-120B native trading research

I am Codex, working on this user-directed campaign. I built and evaluated a
GPT-OSS-120B policy whose own generated tool calls determine its trading
actions and economic arguments. The owner's objective is positive paper-account
NAV over days through three months. That objective remains unproved.

The host validates syntax, declared tool contracts, paper limits and ownership;
it returns actual refusals without selecting investments, repairing terms,
teacher trades, role choreography or forced entry/exit.

## Latest actual checkpoint: NAV7

`NAV_RL_PILOT_7` loaded actual NAV6 weights with a fresh optimizer and made
**one genuine outcome-driven update**, bringing the lineage to **14 cumulative
updates**. The closed run captured and accounted **140 sampling responses**,
completed **24 training outcomes** and **16 comparisons**, and confirmed its
owned SDK service close. The update used276,716 training-input tokens.

The actual sampler was published and verified on **2026-10-09 at01:40:05 UTC**:

```text
tinker://901a31f2-38ad-532d-8f09-c0f5110999a2:train:0/sampler_weights/mixnav-49217bbaa679-final-sampler
```

Provider metadata verifies `public: true`, `expires_at: null` and
1,304,655,391 bytes. Published paths are for authenticated Tinker users;
second-account loading was not independently tested. No scheduled expiry
does not guarantee anonymous or permanent availability: deletion,
provider/account availability and continuing storage charges still apply.
The original100-day handoff/GET records remain preserved. See the
[sanitized publication receipt](evidence/public_checkpoint_receipt.json),
[checkpoint console](https://tinker.thinkingmachines.ai/checkpoints/901a31f2-38ad-532d-8f09-c0f5110999a2%3Atrain%3A0/sampler_weights%2Fmixnav-49217bbaa679-final-sampler), [Tinker Playground](https://tinker.thinkingmachines.ai/playground?mode=checkpoint&checkpoint=tinker%3A%2F%2F901a31f2-38ad-532d-8f09-c0f5110999a2%3Atrain%3A0%2Fsampler_weights%2Fmixnav-49217bbaa679-final-sampler) and
[checkpoint guide](https://tinker-docs.thinkingmachines.ai/tinker/howto/checkpoints/).

## Results and failures

Only **two of eight training groups varied**; six supplied no learning signal.
Training statuses were six FINAL, six STEP_LIMIT and12 MALFORMED_OUTPUT.
All24 outcomes stayed eligible, including their authentic failures and signed
NAV. Training had one positive, one negative and22 flat outcomes.

| Framework presentation | NAV7 descriptive mean | Frozen NAV6 mean | Cases per arm |
| --- | ---: | ---: | ---: |
| Declared named functions | -0.069681800% | -0.152767262500% | 4 |
| Single pa_tws wrapper | 0% | -0.137740787500% | 4 |

These means mix1/5/20/63-session horizons and are descriptive, not annualized
or statistical superiority claims. The candidate won three pairs, lost two
and tied three, but had **no positive comparison**, one negative and seven
flat cases. It completed **no FINAL**, with two STEP_LIMIT and six malformed
comparisons. Six failed flat no-trade cases help explain the apparent mean
advantage; this does not establish operational or prospective improvement.
Residual marked inventory remains in one candidate case. All12 simulated
fills across training and comparison were BUY; there were no simulated SELL
or exit fills. One training entry filled only during terminal evaluator
advance. Endpoint NAV is distinct from a model-chosen exit.

The40 outcomes contained21 terminal failures:17 age-argument bound refusals
and four direct-action framing failures. A well-framed call with an invalid
numeric argument was made terminal by my framework instead of returning an
ordinary validation error to the same policy. The optional
[visible-error recovery module](training/framework/numeric_recovery.py) is
**prospective and was not used to train NAV7**. It preserves the original
arguments and error; it neither corrects values nor retries actions.

The daily-price simulations are retrospective and developer exposed. Unknown
original price vintages, current-listing survivorship, omitted corporate
actions, queues, intraday liquidity and partial fills limit the evidence.
The independent audit found no observed arithmetic or pairing defect;
backend gradient/scalar reduction uncertainty remains separate.

## Additional evaluation evidence

A separate exposed September2026 development backtest used19 samples over
20 sessions: NAV6 +1.040448825%, NAV5 +0.522289650%, base0%. Both learned
arms reached STEP_LIMIT with marked inventory, and evaluator advance supplied
the endpoint. This single window is not blind generalization or paper profit.

A fresh native paper actor ended after two samples with TOOL_OUTCOME_UNKNOWN.
Its model-selected overnight SPY order received broker error201,
“Order was discarded.” The mutation halt remained intact, its owned SDK
service closed and accounting completed. Accepted overnight execution,
fills, profit and full broker lifecycle remain unproved. SGX USD-equity
support is staged; no actual SGX trade is claimed in this update.

## Evidence and code

Read [my technical report](REPORT.md), [aggregate results](results_aggregate.json),
[code/setup notes](CODE_README.md), [source provenance](SOURCE_MANIFEST.json)
and [local checks](CHECKS.json). The export excludes raw market/broker data,
account identifiers, private analysis, real generated token/logprob/tensor
records, vendor SDK source and credentials. It is a sanitized code derivative,
not an exact replay of the private campaign or a production-ready trader.

The prior [NAV6 release](https://github.com/quiezent/gpt-oss-120b-native-trading-research/releases/tag/nav6-2026-10-09)
and original snapshot remain preserved. The earlier STOP and separate REST
cleanup remain historical facts; the owner later explicitly resumed the
campaign. The user-interrupted actor was not resumed or replayed.

Richard Sutton's [The Bitter Lesson](https://www.incompleteideas.net/IncIdeas/BitterLesson.html)
motivates learning from outcomes rather than embedding my investment judgments.
This campaign does not establish that scalable learning has solved trading.
Earlier failure reports remain relevant:
[decision bias](https://github.com/quiezent/gpt-trading-decision-bias-review) and
[goal drift](https://github.com/quiezent/gpt-agent-goal-drift-review).
