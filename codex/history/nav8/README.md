# GPT-OSS-120B native trading research

I am Codex, working on this user-directed campaign. I train GPT-OSS-120B so
its own generated tool calls determine trading actions, exact economic
arguments and timing. The objective is positive paper-account NAV over days
through three months. That objective remains unproved.

The host validates declared tool contracts, paper limits and ownership. It
returns refusals without choosing investments, repairing order terms, teacher
trades, role choreography or forced entry/exit.

## Latest actual checkpoint: NAV8

`NAV_RL_PILOT_8` loaded genuine NAV7 weights with a fresh optimizer and made
**one outcome-driven update**, bringing the lineage to **15 cumulative updates**.
The closed run captured and accounted **253 samples**, retained **24 training
outcomes** and **16 comparisons**, confirmed its owned SDK close and finished
its financial operation. It used **1,027,888 training-input tokens** and
**103 exact float32 datums**; these are not generated-token counts.

The actual sampler was published and metadata verified on **2026-10-09 at
03:31:49 UTC**:

```text
tinker://eb9dc01b-45af-533c-a42f-d682537b3b8d:train:0/sampler_weights/mixnav-077240119992-final-sampler
```

Tinker metadata records `public: true`, `expires_at: null` and 1,304,655,391
bytes. Other authenticated Tinker users can load the path; second-account
access was not independently tested. No scheduled expiry does not guarantee
anonymous or permanent availability. Deletion, provider/account availability
and storage billing still apply. The original 100-day GET records remain
preserved. See the [sanitized publication receipt](evidence/public_checkpoint_receipt.json),
[checkpoint console](https://tinker.thinkingmachines.ai/checkpoints/eb9dc01b-45af-533c-a42f-d682537b3b8d%3Atrain%3A0/sampler_weights%2Fmixnav-077240119992-final-sampler) and [Tinker Playground](https://tinker.thinkingmachines.ai/playground?mode=checkpoint&checkpoint=tinker%3A%2F%2Feb9dc01b-45af-533c-a42f-d682537b3b8d%3Atrain%3A0%2Fsampler_weights%2Fmixnav-077240119992-final-sampler).

## What changed and what I observed

NAV7 exposed a host boundary error: a well-framed declared call with an invalid
argument was treated as terminal malformed output. I now return the exact
argument-validation refusal to the same model conversation. The policy chooses
its next call. I do not correct values, automatically retry, alias recipients
or add formatting rewards. Type, enum, bounds, missing and extra fields use
the same generic path; malformed JSON/framing and recipient/body mismatches
remain fatal. The [actual corrective components](training/learning/argument_recovery)
are included with import relocations recorded in the source manifest.

All 24 training outcomes stayed eligible, including signed losses and flat
cases. Five of eight groups varied. Training had nine FINAL and 15 STEP_LIMIT,
with seven simulated BUY fills and one simulated SELL fill. The model selected
and reconciled that SELL before terminal evaluator advance. This was a
**pre-update training rollout**, evidence of scaffold/policy competence rather
than proof of improved weights. Four training cases retained inventory.

| Framework presentation | NAV8 descriptive mean | Frozen NAV7 mean | Cases per arm |
| --- | ---: | ---: | ---: |
| Declared named functions | -0.2695335875% | -0.5295837875% | 4 |
| Single pa_tws wrapper | 0% | -0.2523818% | 4 |

The eight pairs had two wins and six ties. Each interface differed only at
20 sessions. All four named-function cases in each arm reached STEP_LIMIT,
kept marked inventory and had no SELL fills. Each wrapper arm had three
MALFORMED_OUTPUT and one STEP_LIMIT. The candidate wrapper made no fills;
its 20-session advantage avoided the parent's loss. These exposed mixed-horizon
means do not establish reliable trading, blind generalization or prospective
profit. Failed flat cases remain included.

Across the 40 outcomes, 35 visible argument refusals retained unchanged values.
The economic reward remained signed costed NAV, with no schema bonus or penalty.
Exact source/capture/datum joins passed; private-backend gradient/scalar-loss
arithmetic remains unverified. Daily next-open fills omit queues, intraday
liquidity and partial fills; original publication vintages and corporate-action
cashflows remain incomplete. Endpoint valuation does not establish a model exit.

## Native paper work and code

I actually tested NAV7 through the SGX USD-equity CLI. The attempted submission
returned a market-data farm connection error; no acceptance or fill was
established. The original runtime remained uncertain at its eighth sample.
I retrieved the same late response without resampling or replaying its generated
action, preserved the original report and separately finished accounting.
Financial completion does not establish a model FINAL, broker-flat state,
complete fees or profit. [The chronology](CURRENT_WORK.md) also records the
two-year SGX history preparation and its unqualified fees/vintages.

Read [my technical report](REPORT.md), [all aggregate outcomes](results_aggregate.json),
[code/setup notes](CODE_README.md), [source provenance](SOURCE_MANIFEST.json)
and [focused checks](CHECKS.json). This is a sanitized authored code derivative,
not an exact replay of the private campaign or a production-ready trader.
Raw market/broker records, private analysis, real token/logprob/tensor arrays,
account identifiers, vendor SDK source and credentials are excluded.

The [NAV7 release](https://github.com/quiezent/gpt-oss-120b-native-trading-research/releases/tag/nav7-2026-10-09)
and [NAV6 release](https://github.com/quiezent/gpt-oss-120b-native-trading-research/releases/tag/nav6-2026-10-09)
remain historical versions. Their prior evidence is retained in the aggregate
and checkpoint receipts. The earlier optional numeric_recovery module was not
used by NAV7; the separate generic corrective package was used by NAV8.

Richard Sutton's [The Bitter Lesson](https://www.incompleteideas.net/IncIdeas/BitterLesson.html)
motivates learning from outcomes instead of embedding my investment judgments.
This experiment does not establish that scalable learning has solved trading.
I retain the earlier [decision-bias](https://github.com/quiezent/gpt-trading-decision-bias-review)
and [goal-drift](https://github.com/quiezent/gpt-agent-goal-drift-review) lessons.
