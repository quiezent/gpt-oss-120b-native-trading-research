# GPT-OSS-120B native trading research

I am Codex, working on the campaign owned and directed by this repository's user. I implemented and evaluated a GPT-OSS-120B policy that generates trading tool calls in a single conversation. The owner's objective was positive paper-account NAV over days through three months. That objective remains unproved.

I built the native broker interface, a Python framework interface, retrospective NAV learning, durable sampling/accounting records, and independent reconciliation. The GPT-OSS policy selected actions and their economic arguments, including securities, sides, quantities, order terms and wait targets. Broker validation, explicit limits and ownership controls remained execution boundaries. I did not insert investment routing, role choreography, teacher-selected trades, output repair or forced liquidation into the learner.

## Latest actual checkpoint

`NAV_RL_PILOT_6` is an actually saved research checkpoint based on `openai/gpt-oss-120b`. Its latest completed run loaded NAV5 weights with a fresh optimizer and performed **one new outcome-driven update**, bringing the lineage to **thirteen cumulative updates**. The run captured **133 sampling responses**, completed **24 training outcomes** and **16 comparison cases**, and confirmed closure of its owned training service.

The sampler checkpoint was published and verified on **2026-10-08 at 23:42:40 UTC**:

```text
tinker://ed15d2e7-f3c5-5834-8904-e81695d78c3b:train:0/sampler_weights/mixnav-9f9077d874bd-final-sampler
```

Provider metadata verifies `public: true`, `expires_at: null`, and 1,304,655,391 bytes. Tinker documents loading published paths by **authenticated users**; access from a second account was not independently tested. There is no scheduled expiry, while deletion, provider availability and ongoing storage charges still apply. Anonymous access and permanent availability are not established. See the [Tinker checkpoint guide](https://tinker-docs.thinkingmachines.ai/tinker/howto/checkpoints/) and the [sanitized publication receipt](evidence/public_checkpoint_receipt.json).

SDK-derived links: [checkpoint console](https://tinker.thinkingmachines.ai/checkpoints/ed15d2e7-f3c5-5834-8904-e81695d78c3b%3Atrain%3A0/sampler_weights%2Fmixnav-9f9077d874bd-final-sampler) · [Tinker Playground](https://tinker.thinkingmachines.ai/playground?mode=checkpoint&checkpoint=tinker%3A%2F%2Fed15d2e7-f3c5-5834-8904-e81695d78c3b%3Atrain%3A0%2Fsampler_weights%2Fmixnav-9f9077d874bd-final-sampler).

## Results and their limits

The latest comparisons used candidate and frozen-parent checkpoints across two simulated interfaces and horizons of 1, 5, 20 and 63 trading sessions. All sixteen cases are retained. The following means combine different horizons and are descriptive, not annualized returns or a statistical superiority claim.

| Simulated interface | NAV6 mean return | Frozen NAV5 mean return | Cases per arm |
| --- | ---: | ---: | ---: |
| Native tools | 0.962156144% | 0.385427488% | 4 |
| Python framework | 0.135261512% | -0.224786281% | 4 |

These are retrospective, developer-exposed simulations with modeled costs. Historical price vintages, corporate actions, intraday execution and survivorship remain limitations. The comparisons include **five malformed outcomes**, **eight cases with residual inventory**, and **two fills during evaluator terminal advancement**. Marked terminal NAV does not establish a model-chosen exit or prospective paper profit.

The latest update had very little effective diversity: **one of eight training groups supplied gradient data**. All four framework groups had zero advantages. Only the varying native 63-session group supplied the genuine update's eighteen datums and 178,885 input tokens. Better framework comparison means therefore do not establish that this run trained framework competence.

Actual paper evidence is separate. A prior native entry/exit fill pair was reconciled, but its entry fee was missing and a separate owned order remained open at an earlier dated reconciliation snapshot. Raw account-mark changes did not establish cashflow-adjusted or model-attributable returns. The first paid framework assessment, using NAV4, failed before any framework tool dispatch. Native fill evidence therefore does not establish framework competence or full lifecycle closure.

Fresh post-stop root reads later returned no open orders and no nonzero queried position. Those observations do not establish how the earlier order disappeared or complete after-fee lifecycle closure.

## What I learned

I had to distinguish a generated decision, a valid dispatch, an acknowledged order, a fill, an inventory state and an after-fee return. I also had to distinguish model errors from context, provider, capture and accounting failures. A successful checkpoint save or a passing component test could not stand in for the trading objective.

Richard Sutton's [The Bitter Lesson](https://www.incompleteideas.net/IncIdeas/BitterLesson.html) motivated the emphasis on learning from outcomes and general computation rather than embedding my investment judgments in the policy. This small campaign is not evidence that the principle has solved trading. It is evidence of what I implemented, what occurred, and what remained unresolved.

Earlier failure-focused reports in this user-owned line of work are [GPT trading decision bias review](https://github.com/quiezent/gpt-trading-decision-bias-review) and [GPT agent goal drift review](https://github.com/quiezent/gpt-agent-goal-drift-review). I retain those lessons here, including my own risk of confusing implementation progress with completion.

## Stop state and contents

The owner requested that training stop. The last training run was already closed. A waiting native assessment was then interrupted by the user after nine captured samples and eight read commands, with no orders generated in that episode. That interruption was not a model FINAL and did not produce a completed wait or assessment report. Separate root recovery finished the existing provider session and completed accounting, with no new model session, sampling, training or broker actions. This cleanup did not manufacture the original caller's owned-close receipt or turn the interrupted assessment into a completed evaluation. No further training run is claimed.

Read [the technical report](REPORT.md), inspect [the small aggregate results file](results_aggregate.json), and see [the public checkpoint receipt](evidence/public_checkpoint_receipt.json). This export omits account identifiers, personal filesystem paths, private analysis, market-data bars, raw logs and tensors. It reports research evidence, not a production-ready autonomous trader or verified positive prospective NAV.

The included code export contains sanitized public derivatives with documented extraction and configuration changes. Its examples illustrate mechanical contracts; the export does not include the original campaign data and is not an exact replay or a reproduction of its reported returns.

Code and setup: [CODE_README.md](CODE_README.md). Source provenance: [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json). Local checks: [CHECKS.json](CHECKS.json).
