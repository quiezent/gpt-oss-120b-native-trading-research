# Implementation, results and unresolved failures

I prepared this account as Codex for the user who owns and directed the campaign. The user set the investment objective and the execution limits. I implemented the experiment and examined its retained evidence; GPT-OSS-120B generated the policy's tool calls. I report that distinction because an agent's software work and a model's investment behavior are different results.

## The system I implemented

I connected a single Harmony conversation to native broker commands and, separately, a Python framework interface. The policy chose every economic action and argument. The host performed schema validation, mechanical argument mapping, explicit-limit checks, account/ownership checks and once-only dispatch protection. Refusals were observations returned to the policy. No host-authored investment strategy, trade labels, forced entry or forced exit supplied the economic decisions.

I preserved the entire generated conversation prefix. Local clock and wait operations allowed model-selected timing without sampling while waiting. I persisted raw responses before parsing and bound successful training capture to request, claim, usage and token identities. Unknown provider or broker outcomes remained uncertain, and failed model output was not repaired or replayed.

For retrospective learning, I used signed terminal marked NAV after declared modeled costs:

```text
reward = (terminal NAV - initial NAV - net external cashflows) / initial NAV
```

I centered the unchanged returns within each eligible rollout group. Schema success did not earn a bonus, and I did not select training examples by return sign. Complete source-classified model failures could retain their actual economic outcome, including earlier positions and fees. Uncertain capture, provider, broker or provenance evidence required exclusion of the whole group. Prospective refinements to public-body classification were staged later and do not retroactively relabel the completed run.

The simulator's terminal evaluator advanced to the predeclared endpoint. This could settle pending model orders; it did not force liquidation. Residual inventory stayed in marked NAV and in the report.

## Latest completed learning run

The actual `NAV_RL_PILOT_6` lineage has thirteen cumulative outcome-driven optimizer updates. Its latest run loaded the saved NAV5 weights with a fresh optimizer, used rank-8 LoRA on attention and MLP components, a learning rate of `1e-5`, and the importance-sampling loss interface. It completed one new update, eight groups of three training rollouts, and sixteen candidate/parent comparisons. There were 133 captured and accounted sampling responses.

The twenty-four training outcomes comprised ten FINAL, eleven MALFORMED_OUTPUT and three STEP_LIMIT outcomes. Their source-bound integrity review admitted all twenty-four under the executed failure policy. The update, checkpoint saves and owned-service closure are genuine. Those facts do not independently verify the provider's internal gradient or scalar-loss reduction arithmetic; earlier diagnostic uncertainty remains preserved.

The effective learning signal was narrow. Seven of the eight groups had zero advantages, including all four framework groups. Only the native 63-session group varied. Its eighteen native datums, with 178,885 input tokens, supplied the genuine update. The framework comparisons may reflect effects of that shared-weight update, but they do not establish a framework-specific training signal, reliable framework behavior or diverse policy learning.

## All sixteen latest comparison cases

Each row compares one candidate case with one frozen-parent case. Returns include the declared simulated costs. The two interfaces used the same economic simulation assumptions.

| Horizon, trading sessions | Simulated interface | NAV6 return | Frozen NAV5 return |
| --- | --- | ---: | ---: |
| 1 | Native tools | 0.000000000% | 0.000000000% |
| 1 | Python framework | 0.000000000% | 0.000000000% |
| 5 | Native tools | -0.410246650% | -0.602094250% |
| 5 | Python framework | 0.000000000% | 0.000000000% |
| 20 | Native tools | 0.232763000% | -1.340397900% |
| 20 | Python framework | -0.057826975% | -0.899145125% |
| 63 | Native tools | 4.026108225% | 3.484202100% |
| 63 | Python framework | 0.598873025% | 0.000000000% |

Across the sixteen outcomes, five were FINAL, five MALFORMED_OUTPUT and six STEP_LIMIT. Eight retained inventory. Two fills occurred during terminal evaluator advancement. I retain every malformed and partial outcome in the aggregate; I do not publish only winning or well-formed cases.

The data and historical evaluation windows were developer-exposed. The current-listed universe carries survivorship bias; original publication vintages and historical corrections are unknown. Dividend cashflows and corporate actions were not reconstructed. Daily next-open fills omit queues, partial fills and intraday liquidity. The table establishes descriptive outcomes under those assumptions, not blind generalization, uniform improvement or prospective IBKR profitability.

## Actual paper evidence

Earlier native model decisions produced a genuine paper entry/exit fill pair that later read-only reconciliation joined. A missing entry fee prevented complete after-fee trade PnL. A separate owned order remained open in an earlier completed root reconciliation, recorded on 2026-10-08 at 18:37 UTC. This is a dated historical observation. The native order and fill evidence belongs to those earlier decisions, not to the later framework failure.

Fresh root reads after the user interruption, ending between 23:44:58 and 23:45:03 UTC, returned an empty open-order list and no nonzero queried position. The earlier order's disappearance has an unknown cause. This five-read set did not obtain completed-order history, so I do not infer cancellation, a model-selected closure or complete owned lifecycle proof. A broker PnL field was observed, but the earlier entry fee and complete return attribution remained unresolved.

Account NAV observations were sequential snapshots, not atomic statements of strategy performance. Raw mark changes were observed, but complete cashflow history, earlier fees and model attribution were not established. The principal's funding/reset attestation covered only an earlier interval ending at 11:01:58 UTC; it did not extend automatically to later observations. I therefore keep net and model-attributable returns unknown.

The first paid framework assessment used NAV4, captured one sample and failed before any API or transport invocation. It addressed `functions.framework_cancel` while the declared generic recipient was `functions.pa_tws`. The generated sequence was within its effective output allowance, so this was an address/schema failure rather than context exhaustion. No output alias, repair or retry converted it into a successful action. This was a separate earlier assessment, not an actual named-framework paper test of NAV6.

## Failures I preserved and the repairs they motivated

- **Context exhaustion:** an earlier clock-enabled native episode captured fifteen samples, then required 34,338 joint tokens against a 32,768-token limit while appending a completed tool receipt. No sixteenth sample occurred. I reproduced the boundary and added durable exception diagnostics and lossless compact presentation; I did not trim the old history or resume the failed episode.
- **Semantic claims:** some native responses relabelled stale observations as current, confused account measures, reversed limit-order reasoning, or claimed unobserved fees, fills or returns. Valid broker argument handling did not make those claims sound.
- **Tool addressing:** generic and named operation declarations must agree with the actual parser and renderer before sampling. I staged a named-operation framework adapter, but that unrun preparation is not a successful paper evaluation or an additional update.
- **Evidence and test quality:** copied fixtures exposed real ABI and lifecycle-parser mistakes. I preserved failed fixtures, distinguished synthetic checks from authentic dispatch evidence, and narrowed later checks to the changed contracts.
- **Measurement:** absent fees remained unknown. Provider-service closure did not establish broker flatness, and a declared sampling reservation was not an invoice or realized cost.

The earlier [decision-bias review](https://github.com/quiezent/gpt-trading-decision-bias-review) and [goal-drift review](https://github.com/quiezent/gpt-agent-goal-drift-review) document related software-gate and completion failures. I applied the same lesson to this work: a repaired component, a saved checkpoint and an attractive simulation result each needed its own evidence boundary.

Richard Sutton's [The Bitter Lesson](https://www.incompleteideas.net/IncIdeas/BitterLesson.html) influenced the decision to let an outcome learner discover policy behavior through general computation. My implementation still supplied the environment, observation format and execution boundaries. This campaign does not demonstrate that outcome learning alone produces reliable trading behavior.

## Published checkpoint and stopping

I have an actual saved sampler checkpoint whose provider metadata now verifies `public: true` and no scheduled expiry. Tinker documents loading published paths by authenticated users; access from a second account was not independently tested. The exact path, SDK-derived console/Playground links and metadata are in [the sanitized publication receipt](evidence/public_checkpoint_receipt.json). Retention follows the [Tinker checkpoint documentation](https://tinker-docs.thinkingmachines.ai/tinker/howto/checkpoints/), with continuing storage charges and provider availability.

The user requested that training stop. The latest learning run had already closed. A later native assessment waiting on a model-selected time was interrupted after nine captured samples and eight reads, with no order actions in that episode. It did not produce a model FINAL, a completed wait or a completed assessment report. Separate root recovery finished the existing provider session and completed the accounting operation, without creating a new model session or making sampling, training or broker calls. The original caller's owned-close receipt was not recovered or manufactured. The full episode allowance remained retained; cleanup did not imply a refund. I do not count the interruption, cleanup, staged successor work or checkpoint publication as new learning.

Positive prospective NAV, generalization, reliable economic interpretation, paper framework competence and full owned lifecycle closure remain unproved. This public report deliberately omits private account details, personal paths, market-data bars, raw model analysis, transport logs and tensors.

The source export is a sanitized public derivative, not the exact campaign execution archive. Extraction and configuration transformations are documented separately. Focused examples cannot reproduce the historical returns without the omitted data, and prospective named-tool preparations do not count as completed experiments.
