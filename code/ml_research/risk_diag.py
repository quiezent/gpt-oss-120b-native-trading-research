"""Post-hoc diagnostics for family "risk" (reads runs/ml/research/risk_allpreds.npz; changes no variant or pass)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "ml_research"))
import risk     # noqa: E402
import trader   # noqa: E402

d = trader.Data()
F, S, Y, N, V, sig20 = risk.build(d)
P = np.load(ROOT / "runs/ml/research/risk_allpreds.npz")
nd = len(d.dates)
t0 = next(i for i, s in enumerate(d.dates) if s >= "20250102")
out = {}

# vol20: split R^2 into a time component (daily cross-sectional means) and a within-day component
st = np.arange(t0, nd - 20)
y = Y["vol20"][:, st]
for k in ("vol20_mlpy", "vol20_har", "vol20_gbm", "vol20_naivecal"):
    p = P[k][:, st]
    ok = ~np.isnan(p) & ~np.isnan(y) & ~np.isnan(N["vol20"][:, st])
    yy, pp = np.where(ok, y, np.nan), np.where(ok, p, np.nan)
    ym, pm = np.nanmean(yy, 0), np.nanmean(pp, 0)
    r2t = 1 - np.nanmean((ym - pm) ** 2) / np.nanvar(ym)
    yd, pd_ = yy - ym, pp - pm
    r2x = 1 - np.nanmean((yd - pd_) ** 2) / np.nanvar(yd)
    out[k] = {"r2_time_component": round(float(r2t), 3), "r2_within_day": round(float(r2x), 3),
              "mean_bias_log": round(float(np.nanmean(pp - yy)), 3)}

# tail: who is in the lowest predicted decile, and the AUC for stocks only
p, yt = P["tail5_gbm"][:, t0:nd - 5], Y["tail5"][:, t0:nd - 5]
ok = ~np.isnan(p) & ~np.isnan(yt)
q10 = np.quantile(p[ok], 0.1)
low = ok & (p <= q10)
etf = np.array([s in d.etf for s in d.sym])
out["tail_low_decile"] = {"share_etf": round(float(low[etf].sum() / low.sum()), 3),
                          "observed_rate_etf": round(float(yt[low & etf[:, None]].mean()), 3),
                          "observed_rate_stock": round(float(yt[low & ~etf[:, None]].mean()), 3),
                          "top_symbols": [str(d.sym[j]) for j in np.argsort(-low.sum(1))[:12]]}
stock = ok & ~etf[:, None]
out["tail_auc_stocks_pooled"] = risk.auc(p[stock], yt[stock])
months = sorted({d.dates[t][:6] for t in range(t0, nd - 5)})
rows = []
for m in months:
    cols = [i for i, t in enumerate(range(t0, nd - 5)) if d.dates[t][:6] == m]
    rows.append({"month": m, "pred": round(float(np.nanmean(p[:, cols])), 3), "obs": round(float(np.nanmean(yt[:, cols])), 3)})
out["tail_monthly_mean"] = rows
print(json.dumps(out, indent=1))
(ROOT / "runs/ml/research/risk_diag.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
