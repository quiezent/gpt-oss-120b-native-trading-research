"""Worked example of the shared harness (ml_eval.py): the 5-session return model from ml.py, re-scored.
Every research family follows this shape: build features/targets from cached data, walk_forward, score, save."""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ml          # noqa: E402
import ml_eval     # noqa: E402
import trader      # noqa: E402

d = trader.Data()
X, y, V, yv, f = ml.panel(d)
t0 = next(i for i, s in enumerate(d.dates) if s >= "20250102")
pred = ml_eval.walk_forward(lambda idx: ml.rows(X, ml.FEATS, idx), y, steps_from=130, test_from=t0, horizon=5,
                            retrain_every=21, fit=lambda Xa, ya: ml.fit(Xa, np.clip(ya, *np.nanpercentile(ya, [1, 99])), "huber"),
                            predict=lambda m, Xa: m.predict(Xa))
res = ml_eval.score(pred, y, list(range(t0, len(d.dates) - 6)), 5, d.dates)
print(json.dumps(res, indent=1))
