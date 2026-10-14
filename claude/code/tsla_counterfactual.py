"""Base vs NAV4 on NAV4's exact saved prompts from its 8 Oct 2026 live TSLA episode.

Step 1 is the opening prompt; step 3 is the prompt on which NAV4 placed BUY 3 TSLA at 374.50
(it includes NAV4's own steps 1-2). Each sample uses its own seed: one Tinker call with
num_samples=5 and a single seed returned five identical samples.

    python tsla_counterfactual.py

NAV4's saved live prompts (data/codex) were removed on 14 Oct 2026 at Clayton's instruction, so this no longer
runs; its results are in runs/tsla_counterfactual/.
"""

import json
import re
from pathlib import Path

import tinker

from nav_runner import BASE, CALL, MODELS, PROJECT_ID, RET, get_tok

ROOT = Path(__file__).resolve().parent
EP = ROOT / "data/codex/nav4_live/episode"   # prompts contain the paper account id: not published
OUT = ROOT / "runs/tsla_counterfactual/results_v2.json"


def first_action(text):
    m = re.search(r"<\|message\|>(\{.*?\})<\|call\|>", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except ValueError:
            return {"unparsed": m.group(1)[:200]}
    return {"final_or_other": text[-200:]}


def main():
    tok = get_tok()
    svc = tinker.ServiceClient(project_id=PROJECT_ID)
    samplers = {"base": svc.create_sampling_client(base_model=BASE),
                "nav4": svc.create_sampling_client(model_path=MODELS["nav4"])}
    jobs = []
    for step in (1, 3):
        prompt = tinker.ModelInput.from_ints(json.loads((EP / f"step-{step:04d}-model.raw").read_text())["prompt_tokens"])
        for arm, sc in samplers.items():
            jobs.append((step, arm, 0.0, 2408, sc.sample(prompt, 1, tinker.SamplingParams(
                temperature=0.0, seed=2408, max_tokens=4096, stop=[RET, CALL]))))
            for seed in range(101, 111):
                jobs.append((step, arm, 1.0, seed, sc.sample(prompt, 1, tinker.SamplingParams(
                    temperature=1.0, seed=seed, max_tokens=4096, stop=[RET, CALL]))))
    rows = []
    for step, arm, t, seed, fut in jobs:
        seq = fut.result().sequences[0]
        a = first_action(tok.decode(list(seq.tokens)))
        rows.append({"step": step, "arm": arm, "temperature": t, "seed": seed, "sample_tokens": len(seq.tokens), "action": a})
        print(step, arm, t, seed, len(seq.tokens), json.dumps(a)[:160], flush=True)
    OUT.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
