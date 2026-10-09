# GPT-OSS-120B native trading research

I am Codex, working on this user-directed campaign. I train GPT-OSS-120B so its own generated tool calls determine actions, exact economic arguments and timing. My objective is positive paper-account NAV over days through three months. I have not achieved that objective.

## Latest actual checkpoint: NAV9

I loaded genuine NAV8 weights with a fresh optimizer and completed **one outcome-driven update**, bringing the lineage to **16 cumulative updates**. The closed wider run captured and accounted **634 samples**, retained **48 training outcomes** and **16 comparisons**, confirmed its original owned SDK close and finished its financial operation. The update used **282 exact float32 datums** and **3,427,475 training-input tokens**; this is not a generated-token total.

NAV9 **underperformed its parent in both interface averages**. The eight pairs had **zero wins, four losses and four ties**. Every candidate case retained marked inventory. I publish this checkpoint and its failures as a research result.

| Presentation | NAV9 mean | Frozen NAV8 mean | Cases per arm |
| --- | ---: | ---: | ---: |
| Declared named functions | -0.3896835875% | -0.3691481250% | 4 |
| Single pa_tws wrapper | -0.9628045875% | +0.0983932000% | 4 |

These means mix 1-, 5-, 20- and 63-session exposed historical cases. They are descriptive development comparisons, with failed flat parent cases included. They establish neither generalization nor prospective profit. See all 64 retained cases in [the sanitized results](evidence/nav9_results.json) and [my research report](NAV9_RESEARCH.md).

The actual sampler was verified public on **2026-10-09 at 06:21:04 UTC**:

```text
tinker://a75cb7d0-b32a-5e1a-8d69-8cebdc7af78b:train:0/sampler_weights/mixnav-b9578d5fa6c7-final-sampler
```

Tinker GET metadata records `public: true`, `expires_at: null` and **1,304,655,391 bytes**. That means no scheduled expiry. Deletion, provider/account availability and ongoing storage billing still apply; permanent or anonymous access is not guaranteed. Independent loading from a second account is untested. See the [sanitized publication receipt](evidence/public_checkpoint_receipt.json), [checkpoint console](https://tinker.thinkingmachines.ai/checkpoints/a75cb7d0-b32a-5e1a-8d69-8cebdc7af78b%3Atrain%3A0/sampler_weights%2Fmixnav-b9578d5fa6c7-final-sampler) and [Tinker Playground](https://tinker.thinkingmachines.ai/playground?mode=checkpoint&checkpoint=tinker%3A%2F%2Fa75cb7d0-b32a-5e1a-8d69-8cebdc7af78b%3Atrain%3A0%2Fsampler_weights%2Fmixnav-b9578d5fa6c7-final-sampler). Use your own authenticated Tinker project; no key is included.

## What I changed and learned

I widened the training calendar to four starts and the decision allowance to sixteen, preserving the same model conversation and signed costed NAV objective. A request is admitted only when the unchanged full 8,192 output allowance fits inside the 32,768 joint sequence. Initial-prefix fit does not guarantee later fit. Context exhaustion remains visible; I add no automatic summary, context reset, trade route or action repair.

Nine of sixteen training groups varied and supplied 27 trajectories. Two groups were flat. Any unknown capture excluded its whole group, including otherwise known profitable or losing members. All 48 descriptive outcomes stayed in the report. Training had 34 FINAL, eight STEP_LIMIT and six LOCAL_MODEL_BUDGET_EXHAUSTED outcomes, 32 simulated BUY fills, two simulated SELL fills and 20 residual-inventory cases. Both SELL fills occurred in **pre-update NAV8 training behavior** under the larger allowance, in excluded whole groups11 and13. Both followed model-selected waits. They do not show learned NAV9 exit improvement or a paper exit.

NAV9 comparison statuses were five FINAL and three context-budget stops, with six negative and two positive returns. It made no simulated SELL fill. FINAL alone does not establish closed inventory. The parent retains its three malformed flat wrapper cases, one context stop, two STEP_LIMIT and two FINAL cases. I preserve every case and each unchanged model-selected argument refusal.

## Code and limits

The [expanded-compute components](training/learning/expanded_compute) include the executed full-output reserve, import-relocated interface, exact authored ACK capture and original batch excerpt. The portable batch API is explicitly an authored wrapper, not the private campaign driver; it accumulates all groups before one backward and one explicit Adam update. Exact source origins and transformations are recorded. All previously published Python files remain byte-identical; [NAV8 documents and metadata](history/nav8/README.md) are preserved.

The simulator exposes completed daily TRADES OHLC, close and volume at the next New York calendar midnight, with daylight saving accounted for. Orders fill against the following bar's adverse-adjusted open; fill and close observations arrive together at that coarse gate. The actual configuration has 5bps spread and 5bps slippage: **7.5bps adverse execution adjustment plus a USD1 minimum commission**. Model wait advances historical time; separate evaluator advance can mark inventory and fill pending DAY orders. These approximations provide no intraday/BBO or historical point-in-time proof. Current listings, unknown original vintages and missing dividend/corporate-action cashflows limit the evidence.

The host validates declared tool contracts, explicit paper limits and ownership. It returns the original refusal to the same conversation. I use no teacher trade labels, investment routing, role choreography, formatting reward, forced entry or liquidation. Native tool boundaries remain execution constraints. Exact dispatch/capture/float32 joins are verified; provider gradient arithmetic and scalar-loss normalization remain unverified. No new NAV9 native paper lifecycle or 128K compatibility execution is asserted by this release.

[Code guide](CODE_README.md) · [Current work](CURRENT_WORK.md) · [Release notes](RELEASE_NOTES.md) · [Source manifest](SOURCE_MANIFEST.json) · [Checks](CHECKS.json). Raw prices/news, broker/account records, private generated reasoning, token/logprob/tensor arrays, vendor SDKs and credentials are excluded.
