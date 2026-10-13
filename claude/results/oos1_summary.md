# NAV re-test oos1: 228 episodes, 38 cases (starts 2025-11-10..2026-08-28, horizons [20, 63])

| | mean marked % | mean liquidated % | realized $/ep | closed flat | still holding | FINAL / STEP / MALF / CTX | cost $ |
|---|---:|---:|---:|---:|---:|---|---:|
| cash | +0.00 | | | | | | |
| passive basket | +1.26 | | | | | | |
| $2,500 of SPY, rest in cash | +0.36 | | | | | | |
| base | +0.92 | +0.88 | -1.4 | 13/76 | 54/76 | 43 / 33 / 0 / 0 | 2.39 |
| nav4 | +0.61 | +0.57 | -2.8 | 6/76 | 58/76 | 42 / 34 / 0 / 0 | 2.39 |
| nav9 | +0.89 | +0.85 | +5.1 | 15/76 | 49/76 | 43 / 31 / 2 / 0 | 2.51 |

Paired by case (seed-averaged), cluster bootstrap over start dates:

| comparison | mean diff pp | wins / losses | sign p | permutation p | 95% CI |
|---|---:|---|---:|---:|---|
| base - passive basket | -0.34 | 19 / 18 | 1.000 | 0.517 | [-1.48, +0.69] |
| nav4 - passive basket | -0.65 | 14 / 24 | 0.143 | 0.176 | [-1.76, +0.39] |
| nav9 - passive basket | -0.37 | 19 / 19 | 1.000 | 0.407 | [-1.39, +0.58] |
| nav4 - base | -0.31 | 17 / 19 | 0.868 | 0.451 | [-1.27, +0.62] |
| nav9 - base | -0.03 | 17 / 20 | 0.743 | 0.932 | [-0.76, +0.69] |
| nav9 - nav4 | +0.28 | 19 / 18 | 1.000 | 0.503 | [-0.56, +1.13] |

By horizon (mean marked %):

| horizon | base | nav4 | nav9 | passive basket | $2,500 SPY |
|---|---|---|---|---|---|
| 20 | +0.44 | -0.07 | +0.33 | +0.59 | +0.11 |
| 63 | +1.52 | +1.45 | +1.59 | +2.09 | +0.68 |
