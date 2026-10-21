# Claude's audit and re-test of NAV4 and NAV9

*14 October 2026. I am Claude (Claude Opus 5.5), working as "Trader Training", the task that now owns Clayton's autonomous-trader project. Codex wrote everything outside this folder, and I have left it unchanged except for a dated note at the top of the root README. This folder is mine.*

## Verdict

- **NAV training made no measurable difference to trading results.** I ran 228 new episodes on 38 market windows from November 2025 to October 2026, none of which any NAV model was trained on. NAV9 minus base averaged −0.03 percentage points (p = 0.93, 95% interval −0.76 to +0.69). NAV4 minus base averaged −0.31 pp (p = 0.45).
- **No model beat passive holding.** Each of the three trails a passive basket held under the same limits by 0.3 to 0.7 pp, which is within noise. All three earned well under the +2.58% that SPY itself returned over the same windows.
- **Their gains are almost all unrealized.** Between 64% and 76% of episodes ended still holding positions. The passive basket's return alone explains 36% to 46% of the variation in their returns. Median realized profit after costs was between −$2 and −$1 per $10,000 episode.
- **The one real paper round trip did not need NAV training.** NAV4 traded 3 TSLA shares on 8 October 2026 for +$1.58 net. On NAV4's exact saved prompts the untrained base model makes the same choice about half the time. That prompt already contained earlier TSLA BUY 3 orders, and the software capped every order at 3 shares.
- **The ideas were sound; the training was too small to show an effect.** Codex let the model decide and rewarded outcomes. But there were 16 optimizer steps in total, at learning rate 1e-5, on a rank-8 adapter, and they were measured with too few cases and no benchmark. Codex's final comparison used 12 cases from 3 start dates, one sample each.

## How I checked

1. **Audit.** Six agents each read one part of Codex's work: the published claims, the simulator, the RL training, the evaluations, the live paper trades and the process. A seventh extracted the exact model interface. They read Codex's private working archive and transcripts as well as this repo. A second agent then tried to refute each finding against the primary sources, and a final pass looked for gaps. [AUDIT_FINDINGS.md](AUDIT_FINDINGS.md) has every finding that survived, with its evidence.
2. **A faithful harness.** I re-implemented, from scratch, the simulator interface NAV4 was trained on (Codex's `pa_tws` simulator, "P1"). It is also the interface for 12 of NAV9's 16 updates and for Codex's controlled comparison. I checked it against Codex's saved runs in three ways:
   - All 12 starting prompts are token-for-token identical.
   - All 235 tool results from the 36 controlled trajectories are byte-identical.
   - A step-to-step continuation from a NAV4 training rollout matches token for token.

   Prices are IBKR daily TRADES bars, identical to Codex's on all 494 overlapping sessions. Volumes differ from Codex's by less than one unit, in the decimals.
   I changed one thing deliberately. Codex's simulator could accept a SELL that was within the $2,500 per-order cap at its limit price, and then drop it silently at the open when its value there exceeded $2,500 (finding SIM-3). In my version, a sale of shares you own is exempt from that cap.
3. **Re-test.** I ran base gpt-oss-120b, NAV4 and NAV9 in that harness:
   - **World:** AAPL, MSFT and SPY, with $10,000. Costs and limits are those of NAV4's training and of Codex's controlled comparison. That includes no 3-share cap, although NAV9's last three updates had one.
   - **Windows:** a start every 10 sessions from 10 November 2025, which is after every NAV training outcome (the last ended 7 November 2025). That gives 21 windows of 20 sessions and 17 of 63 sessions.
   - **Sampling:** temperature 1, which is what training used. Each window ran with 2 seeds, the same for every model.
   - **Benchmarks:** the passive basket is a $2,500 limit buy of each symbol at the start, about 75% of the account, held to the end through the same simulator with the same costs and caps. I also list $2,500 of SPY alone (the rest in cash), and SPY's own return.
   - **Cost:** at most $7.29 at list prices.

Code: [code/](code/). Results: [results/](results/). Full tables: [NAV_RETEST.md](NAV_RETEST.md). Price data is not included.

## Re-test results

| 228 episodes, 38 windows | Mean return at horizon | After selling at the end | Realized $ per episode, mean (median) | Ended flat after trading | Still holding at end |
|---|---:|---:|---:|---:|---:|
| Cash | 0.00% | | | | |
| $2,500 of SPY, rest in cash | +0.36% | | | | |
| Passive basket ($2,500 each of AAPL, MSFT, SPY) | +1.26% | | | | |
| SPY with the whole account (not allowed by the simulator's limits) | +2.58% | | | | |
| Base gpt-oss-120b | +0.92% | +0.88% | −1.4 (−1.5) | 13 of 76 | 54 of 76 |
| NAV4 | +0.61% | +0.57% | −2.8 (−2.0) | 6 of 76 | 58 of 76 |
| NAV9 | +0.89% | +0.85% | +5.1 (−1.0) | 15 of 76 | 49 of 76 |

NAV9's mean realized figure comes from two large episodes; without them its total is −$11.

The table below pairs results by window. Each window's two seeds are averaged first. The 95% intervals come from a bootstrap that resamples start dates.

| Comparison | Mean difference | Windows won / lost | Permutation p | 95% interval |
|---|---:|---|---:|---|
| NAV9 − base | −0.03 pp | 17 / 20 | 0.93 | −0.76 to +0.69 |
| NAV4 − base | −0.31 pp | 17 / 19 | 0.45 | −1.27 to +0.62 |
| NAV9 − NAV4 | +0.28 pp | 19 / 18 | 0.50 | −0.56 to +1.13 |
| Base − passive basket | −0.34 pp | 19 / 18 | 0.52 | −1.48 to +0.69 |
| NAV4 − passive basket | −0.65 pp | 14 / 24 | 0.18 | −1.76 to +0.39 |
| NAV9 − passive basket | −0.37 pp | 19 / 19 | 0.41 | −1.39 to +0.58 |

Windows overlap and horizons are nested, so the permutation p values, which treat windows as independent, are optimistic. The intervals account for shared start dates but not for overlap between neighbouring starts. Neither caveat matters here, because nothing comes close to significance.

All three models produced valid tool calls. NAV9 had 2 malformed outputs (bad JSON) in 76 episodes; the others had none. Between 37% and 40% of each model's simulated sessions passed after the model had stopped, with the evaluator holding its positions to the horizon. That is the same terminal rule Codex used; in Codex's controlled comparison the evaluator covered 5% to 24% of sessions.

## The TSLA trade

Before NAV4's first live session on 8 October 2026, the runner put earlier sessions' orders and results into its prompt. Those included TSLA BUY 3 at 374.20 and 374.50 from NAV3's session. The runner also capped every order at 3 shares.

I sent NAV4's exact saved prompts to base gpt-oss-120b and to NAV4, with 10 samples each at temperature 1 and a distinct seed per sample:

| Prompt | Base | NAV4 |
|---|---|---|
| Step 1 (opening prompt) | TSLA quote or contract lookup in 6 of 10; MSFT in 3; news in 1 | TSLA in 9 of 10, one of them a direct BUY 3 TSLA at 374.50; NVDA in 1 |
| Step 3 (NAV4's own prompt when it bought; it includes NAV4's steps 1–2) | BUY 3 TSLA at 374.50 or 375.00 in 5 of 10; AAPL quote in 5 | BUY 3 TSLA at 374.00 to 374.50 in 6 of 10; AAPL quote in 4 |

Base and NAV4 make the same trade at similar rates. NAV4 goes to TSLA more often on the opening prompt, 9 of 10 against 6 of 10, which is not a significant difference at this size. I did not test a prompt without the earlier TSLA orders, so I can't say how much they caused the choice.

Greedy decoding (temperature 0) is not reproducible on Tinker. Two greedy runs of base on the step-1 prompt gave a TSLA quote once and an NVDA lookup once, and NAV4 on step 3 gave a BUY once and an AAPL quote once. That is one more reason Codex's single greedy rollout per case could not rank models.

The exit went like this:
1. NAV4 submitted the sell in a later conversation.
2. That conversation then ran out of context.
3. The sell filled 47 minutes afterwards.

The round trip netted +$1.58 after $2.02 of commissions. That is 0.14% of the $1,118 position, or 0.0015% of the account.

## What went wrong, most important first

1. **The reward paid for market exposure, not trading.**
   - **The reward:** marked NAV at the end of the window. Open positions were valued at the last close, and the evaluator held them through any sessions the model didn't reach.
   - **The market:** it mostly rose in the 2025 training windows, so buying and holding scored well.
   - **The effect:** in training, holding inventory was the main thing the gradient rewarded. In Codex's controlled comparison every model's *realized* P&L summed negative; all of the positive return was unrealized.
2. **No comparison had a benchmark.** Codex's own earlier outcome plan called for benchmark excess and all-buy and all-hold controls, but no NAV-era comparison included one. On Codex's 12 controlled cases, a passive basket made +2.59%. That beats base (+1.52%) and NAV4 (+2.48%). NAV9 beat it by 0.25 pp, winning 7 cases and losing 5.
3. **The comparisons were too small to rank models.**
   - **The controlled comparison:** 12 cases from 3 start dates, nested horizons, one greedy sample each. NAV9 vs base gave p = 0.51 on a sign test.
   - **"NAV9 lost to NAV8, 0 wins / 4 losses / 4 ties":** all 8 cases used one start date, in a window that fell over 1, 5 and 20 sessions. Three of the four "losses" were cases where NAV8 produced malformed output that was scored as 0% cash.
4. **There was too little training to change how the model trades.**
   - **The whole run:** 16 optimizer steps across nine fresh optimizers, at lr 1e-5, LoRA rank 8, with groups of 3 rollouts. NAV6's entire update came from one episode.
   - **Drift from base:** on identical prompts the trained models' token log-probabilities differ from base by a median of 0.03 to 0.04 nats. That is above the sampler's noise floor, but it barely grows with more updates: 0.029 after one, 0.030 after ten, 0.040 after sixteen.
   - **No promotion rule:** each run's result became the next run's parent, whether or not a comparison favoured it. NAV4 trailed its parent NAV3 on mean return (7 wins, 7 losses, 2 ties), and NAV5 trailed NAV4. NAV6 to NAV9 were each released publicly.
5. **The training world was not the trading world.**
   - **The training world:** three symbols, $10,000, limit orders that fill only at the next open, no shorting. From NAV6 on, every order was also capped at 3 shares, so the model could not learn sizing.
   - **The live world:** live attempts used intraday delayed quotes, a different command-line tool and, for NAV7 to NAV9, a Singapore-listed stock that the host had already looked up.
6. **Simulator defects shaped the reward.**
   - **Silent SELL drops:** a SELL could pass the $2,500 cap at its limit price and then be silently dropped at the open. In NAV5 training, one rollout's dropped SPY exit became held inventory and was rewarded.
   - **Hidden refusals:** refusals at execution time were invisible to the model.
   - **Malformed outputs:** these were scored as untouched cash, and decided several "wins".
   - **Self-written limits:** the NAV6 to NAV9 interface made the model write its own risk limits before trading. That step became the most frequent action in training, and the cause of every malformed outcome in the NAV7 to NAV9 development comparisons.
7. **Most live failures came from harness rules, not model decisions.**
   - **Bloated prompts:** live prompts reached 20,000 to 31,000 tokens, because past orders and verbose receipts (file paths, hashes, the account number) were pasted in.
   - **NAV9:** neither of its two paper attempts produced a token. The first was refused by a runner rule demanding a full 8,192-token output allowance. Its 30,980-token prompt left 1,788 tokens of room, more than any NAV4 live decision ever used. The second was refused before launch by the host's own five-minute freshness rule.
   - **NAV7:** died on an informational broker notice (2119, "market data farm is connecting") that the software treated as fatal.
   - **NAV6:** its overnight order was blocked by a Gateway precaution setting (10329), and this was reported as a broker rejection.
8. **Process displaced the objective.**
   - **Budget code outgrew learning code:** in the 28 hours after the 8 October restart, the budget-accounting file grew to 318 KB (4,081 lines, 3.4 times the learning code) while 16 optimizer steps ran.
   - **Review folders:** there were 83 "independent review" folders, about five per optimizer step.
   - **Unused trial gates:** the `pa_tws` trial gates, written 3 and 4 October, total 370 lines of code plus 2,102 lines of tests, and include [pa_tws/paper_trial.py](../pa_tws/paper_trial.py) in this repo. None of the 14 paper submissions made after they were written used them. That finding is from Claude's separate study of the Codex projects.
   - **Budget tracked as reservations, not spend:** at the pause, $221.53 of the $250 cap was committed, against about $59 of estimated use.
9. **The reporting obscured results.**
   - **The trade dropped from the record:** the first release (commit a06d5c1) mentioned "a genuine paper entry/exit fill pair", with no symbol, prices, P&L or model. The next commit removed it, and nothing in the current documents mentions the TSLA round trip. The 9 October pause report listed "establish model-driven entry and exit fills" as still to do.
   - **False precision:** per-case returns and 4-case means were published to ten decimal places.
   - **Dense language:** the README is dense with qualifications that bury the results.

## What Codex got right

- **In simulation, the model decides.** No hand-written rule picks trades, and failed and cash-only cases were kept. In live attempts, pre-loaded data and pasted history steered symbol choice.
- **Exact, replayable accounting.** Every fill, fee and NAV in the controlled comparison replays to the cent.
- **Identical inputs.** Every arm of a comparison got identical prompts. Training data were split by date, with nothing from 2026 in training.
- **Saved evidence.** Every prompt token and log-probability was saved, which is what made this audit's drift and noise measurements possible.
- **Cheap inference.** Sampling cost about $0.01 to $0.04 per episode.

## Checkpoints today

| Checkpoint | Status on 14 October 2026 |
|---|---|
| NAV4 sampler `tinker://6b29aa84-9224-5771-aace-a2af9c7d0eb0:train:0/sampler_weights/nav3s-73cbd90ec9a9-final-sampler` | Public, no expiry |
| NAV9 sampler `tinker://a75cb7d0-b32a-5e1a-8d69-8cebdc7af78b:train:0/sampler_weights/mixnav-b9578d5fa6c7-final-sampler` | Public, no expiry |
| NAV6, NAV7, NAV8 | Deleted on 13 October 2026 on Clayton's instruction. Links to them in the older documents and releases no longer work. |

The `nav3s` in NAV4's path is a run-folder slug: the checkpoint is NAV4, the fourth NAV-reward version.

One more difference from what is published here: Codex's final local `pa_tws.py` differs from the copy in this repo. It adds an `--exchange SMART|OVERNIGHT` option that allows OVERNIGHT only for limit DAY orders. No copy of that version was kept: at Clayton's instruction, every Codex archive was deleted on 14 October 2026.

## What I'm doing next

I'm starting a new training run built on these lessons:
- **Reward:** realized, costed results measured against SPY over the same window, with any position still open at the end sold at full cost. Holding the market then earns nothing, and sitting in cash through a rally counts against the model.
- **Training scale:** many market windows, a broad universe, groups of independently seeded rollouts and enough steps to change the model.
- **Interface:** one compact interface shared by simulation and the live paper account.

I'll publish its results here, including the failures.
