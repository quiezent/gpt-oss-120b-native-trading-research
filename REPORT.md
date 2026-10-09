# What I learned from NAV7

I resumed this user-directed campaign after the owner asked me to continue
testing and learning. I kept the same objective: a GPT-OSS-120B policy whose
own tool calls determine its actions, targeting positive paper-account NAV
over days through three months. The objective remains unproved.

I first tested the published NAV6 policy against the paper broker with a
fresh finite session. The policy selected an overnight SPY limit buy. Its
first order reference was refused by the native validator; the policy then
generated a different reference itself. The actual submission received
broker error201, “Order was discarded.” The actor stopped with an uncertain
tool outcome after two samples and660 generated tokens, retained its mutation
halt and closed its owned SDK service. Financial accounting completed. I do
not turn that SDK close into broker order closure or infer a fill or profit
from a rejected submission. Overnight acceptance remains unproved.

I also ran a separate inference-only comparison over one exposed20-session
September2026 historical window. It used19 samples across NAV6, NAV5 and
base. Costed marked returns were +1.040448825%, +0.522289650% and0%.
Both learned arms reached the decision limit with residual marked inventory.
Terminal evaluator advance supplied the endpoint valuation; it did not
establish a policy-selected exit. The small window, current-listed universe,
unknown original price vintages and simplified next-open fills prevent a
generalization or prospective-profit claim.

The next learner used genuine NAV6 weights with a fresh optimizer and a
declared framework interface. Its policy could directly call the actual
named framework functions, rather than encoding their operation names in a
single wrapper recipient. The costed NAV objective, exact model arguments,
whole-group eligibility and zero-variation skip remained separate from API
formatting. I did not add a compliance reward, teacher trade or investment
router. The run has now closed with one genuine update,14 cumulative updates,
140 captured/accounted samples,24 training outcomes and16 comparisons.
Its owned SDK close and root accounting finish are confirmed. Only two of
eight training groups varied. Candidate comparisons had no FINAL, two
STEP_LIMIT and six malformed cases, with zero positive, one negative and
seven flat returns. Named-function descriptive means were -0.069681800%
versus NAV6 -0.152767262500%; wrapper means were0% versus -0.137740787500%.
The three wins, two losses and three ties include six failed flat no-trade
candidate cases. These are exposed mixed-horizon comparisons, not proof of
operational improvement. All12 simulated fills were BUY with no exit;
one training entry filled only during terminal evaluator advance.

The run exposed a host error I need to describe precisely. Some well-framed
framework calls failed an argument bound for max_account_age_seconds. The
running framework classified that refusal as terminal malformed output,
instead of returning an ordinary tool error to the same policy conversation.
That prevented the policy from seeing the refusal and choosing its own next
call. I have staged a generic repair that keeps strict JSON/framing/recipient
checks, retains the unchanged arguments and returns the original validation
error visibly. It neither fixes the values nor retries an action. That repair
was not used by the NAV7 learner, and I do not credit its future behavior
to the new weights. Rejected exact numeric values remain unknown in the
receipt-only diagnosis; I have not decoded private model analysis to invent
them.

I have also staged support for USD-denominated SGX equities. Staged source
and a review of an open-market interface are not an actual SGX trade or an
accepted order. The existing overnight halt remains preserved. I will report
future broker observations separately rather than claiming the upgrade has
already improved execution.

I published the genuine NAV7 sampler and verified provider metadata at
01:40:05 UTC on2026-10-09: public=true, expires_at=null and1,304,655,391 bytes.
The exact URI is recorded in the sanitized publication receipt. The original
100-day research handoff and GET vintage remain preserved. “No scheduled expiry”
does not guarantee permanent availability: deletion, account/provider access
and continuing storage charges remain relevant. The existing NAV6 release
and checkpoint evidence will stay available as historical versions.

The code release is an authored sanitized derivative. It excludes private
analysis, generated token/logprob arrays, raw prices and news, broker/account
identifiers, vendor SDK source and credentials. The optional numeric-error
recovery module is explicitly prospective; the executed learner source and
the later repair are distinct. Synthetic component checks do not establish
economic model performance or exact replay of the private campaign.

I am still distinguishing model-selected decisions, validation, submission,
broker acknowledgement, fills, inventory, fees and attributable NAV. The
next checkpoint and a positive retrospective mark are research progress;
they do not complete the trading objective.

The parent [NAV6 release](https://github.com/quiezent/gpt-oss-120b-native-trading-research/releases/tag/nav6-2026-10-09) remains preserved. Current aggregate outcomes and opaque source digests are in [results_aggregate.json](results_aggregate.json); those digests identify retained private records rather than released raw data.
