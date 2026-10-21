"""The training reward for one decision: what it earned over 10 sessions beyond its own market exposure.

For a decision made before session d on a sleeve worth V0 at the previous close, in percentage points:

    R = 100 * [ r_book - r_hedge + sum_t (MU * e_t - GAMMA/2 * v_t) ] - MALFORMED * m

  r_book   (cash after selling everything at the open of session d+H, with costs) / V0 - 1. The decision's
           orders work by their own tif for H sessions; then all orders are cancelled and every position
           is closed at market. Unrealized gains count only if they survive that exit.
  r_hedge  P&L of a shadow SPY book, / V0, no costs: each position the sleeve starts with, and each fill
           of symbol s, adds beta_s x its signed notional in SPY at the same point of the bar as the fill:
           SPY's open for fills at the open (including gaps through a limit or stop), SPY's close for fills
           at the close, and for intraday fills SPY's low + (high - low) x the fill's relative position in
           the symbol's own range. It closes at SPY's open on session d+H. Subtracting it removes the
           market's realized move from each sample, whatever the order type.
  e_t      beta-weighted net exposure at the close of each session t (sum beta_s * value_s / NAV_t).
  MU       expected market premium per session for that exposure (3e-4, about 7.6% a year): idle cash
           gives it up, which makes a missed rally an error whichever way the market went.
  v_t      ex-ante variance of the book: (e_t * sigma_m)^2 + w' S w, where S is the covariance of the
           holdings' 60-session residual returns (after their SPY beta) through the previous close, so
           correlated holdings (share classes, one sector) count as concentration. GAMMA = 4.
  m        1 if the model's output was malformed (the day is then simulated with no orders).

reject_penalty (rl2; 0 for rl1's evaluation) is subtracted for each of the decision's orders that was rejected: bad
fields, or insufficient margin (gross exposure above 2x the sleeve), as IBKR would reject it.

parts["orders"] (not a number; R does not use it) lists, for each of the decision's place orders in input order:
  placed       the order reached the market and was not rejected (bad fields, margin, or a symbol without data)
  filled       it filled within the H sessions; None if it was still working at the end of them (a GTC order), as
               that is neither a fill nor an expiry
  fill_session index in data.dates of the session it filled, fill_price its fill price (after slippage)
  win          1 if the close of session fill_session + 5 is above fill_price x 1.002 (BUY) or below x 0.998 (SELL),
               else 0; None if not filled or that close is past the data
  p_fill, p_win  the model's probabilities (trader.prob: a number in [0, 1], else None); type, action, tif as placed
calibration_bonus(parts["orders"]) scores p_win (Brier reward 1 - (p - y)^2, the forecasting recipe's); the trainer
applies its weight. prob_scores() reports p_fill and p_win apart, next to trivial baselines.
"""

import math

MU, GAMMA, MALFORMED, HORIZON = 3e-4, 4.0, 1.0, 10
REJECT_CAP = 3          # rl2: at most 3 rejected orders are charged (3 x 0.25 < MALFORMED)
BETA_CLIP = 3.0
DEFAULT_RESID = 0.02


def _risk_inputs(feats, s):
    x = feats.get(s) or {}
    b = x.get("beta")
    e = x.get("resid")
    b = 1.0 if b is None or (isinstance(b, float) and math.isnan(b)) else max(-BETA_CLIP, min(BETA_CLIP, b))
    e = DEFAULT_RESID if e is None or (isinstance(e, float) and math.isnan(e)) else e
    return b, e


def branch(world, call, data, d, feats, sigma_m, horizon=HORIZON, reject_penalty=0.0):
    """Run one sampled decision on a cloned SimWorld standing before session d. Returns the reward parts."""
    import numpy as np
    import trader
    m = world.m
    spy = data.k["SPY"]
    if d + horizon >= len(data.dates):
        raise ValueError("window runs past the data")
    v0 = m.nav()
    # symbols this branch can hold: the inherited book plus everything the decision orders
    syms = sorted(set(m.positions) | {str(o.get("symbol", "")).upper().strip() for o in (call or {}).get("place", [])
                                      if isinstance(o, dict)} & set(data.k) | {o.symbol for o in m.open_orders()})
    syms = [s for s in syms if s in data.k]
    pos_ix = {s: i for i, s in enumerate(syms)}
    cov = data.resid_cov(d, syms) if syms else np.zeros((0, 0))
    # shadow SPY book: h shares, hc cash
    h, hc = 0.0, 0.0
    spy_c0 = data.c[spy, d - 1]
    for s, (q, _) in m.positions.items():
        b, _ = _risk_inputs(feats, s)
        x = b * q * m.last(s) / spy_c0
        h += x
        hc -= x * spy_c0
    n_orders0 = len(m.orders)
    notes = world.act(call) if call else []
    n_orders1 = len(m.orders)
    # each place order and its sim order (SimWorld.act places, in input order, every order clean_order accepts)
    new = iter(m.orders[n_orders0:n_orders1])
    mine = [(raw if isinstance(raw, dict) else {}, None if trader.clean_order(raw)[1] else next(new))
            for raw in (call or {}).get("place", [])]
    seen = len(m.fills)
    premium = risk = 0.0
    gross_sum = 0.0

    def spy_price(f):
        t = f.session
        if f.phase == "close":
            return data.c[spy, t]
        if f.phase == "intraday":
            j = data.k.get(f.symbol)
            lo, hi = (data.l[j, t], data.h[j, t]) if j is not None else (f.raw, f.raw)
            frac = 0.5 if not hi > lo else min(1.0, max(0.0, (f.raw - lo) / (hi - lo)))
            return data.l[spy, t] + frac * (data.h[spy, t] - data.l[spy, t])
        return data.o[spy, t]

    def hedge_new_fills():
        nonlocal h, hc, seen
        for f in m.fills[seen:]:
            px = spy_price(f)
            b, _ = _risk_inputs(feats, f.symbol)
            notional = (f.shares if f.side == "BOT" else -f.shares) * f.price
            x = b * notional / px
            h += x
            hc -= x * px
        seen = len(m.fills)

    for t in range(horizon):
        world.advance()
        hedge_new_fills()
        nav = m.nav()
        if nav <= 0:
            nav = 1e-9
        e = gross = 0.0
        w = np.zeros(len(syms))
        for s, (q, _) in m.positions.items():
            b, _ = _risk_inputs(feats, s)
            ws = q * m.last(s) / nav
            e += b * ws
            gross += abs(ws)
            if s in pos_ix:
                w[pos_ix[s]] = ws
        var_idio = float(w @ cov @ w) if len(syms) else 0.0
        premium += MU * e
        risk += 0.5 * GAMMA * ((e * sigma_m) ** 2 + var_idio)
        gross_sum += gross
    working = {id(o) for o in m.open_orders()}        # still working after H sessions: fill outcome unresolved
    world.liquidate()
    hedge_new_fills()
    end = d + horizon
    hc += h * data.o[spy, end]
    v_end = m.nav()     # all positions closed at the open of session d+H (any that couldn't be stay marked)
    r_book = v_end / v0 - 1
    r_hedge = hc / v0
    malformed = 0 if call else 1
    # rejected by IBKR's rules: bad fields, or margin (gross > 2x). A symbol outside the cached universe isn't the
    # model's error (IBKR would accept it), so it isn't counted. Capped so a rejected look always beats a malformed one.
    rejected = (sum(1 for n in notes if n.startswith("order rejected"))
                + sum(1 for o in m.orders[n_orders0:n_orders1]
                      if o.status == "Rejected" and not (o.message or "").startswith("no market data")))
    R = (100 * (r_book - r_hedge + premium - risk) - MALFORMED * malformed
         - reject_penalty * min(rejected, REJECT_CAP))
    orders = []
    for raw, o in mine:
        filled = o is not None and o.status == "Filled"
        win = None
        if filled and o.filled_session + 5 < len(data.dates) and o.symbol in data.k:
            c5 = data.c[data.k[o.symbol], o.filled_session + 5]
            if not np.isnan(c5):
                win = int(c5 > o.fill_price * 1.002) if o.action == "BUY" else int(c5 < o.fill_price * 0.998)
        orders.append({"symbol": o.symbol if o else None, "type": o.order_type if o else None, "action": o.action if o else None,
                       "tif": o.tif if o else None, "placed": o is not None and o.status != "Rejected",
                       "filled": None if o is not None and id(o) in working else filled,
                       "fill_session": o.filled_session if filled else None, "fill_price": o.fill_price if filled else None,
                       "win": win, "p_fill": trader.prob(raw.get("p_fill")), "p_win": trader.prob(raw.get("p_win"))})
    return {"R": R, "book_pp": 100 * r_book, "hedge_pp": 100 * r_hedge, "premium_pp": 100 * premium,
            "risk_pp": 100 * risk, "malformed": malformed, "rejected": rejected, "fills": len(m.fills), "commissions": m.commissions,
            "financing": m.financing, "avg_gross": gross_sum / horizon, "residual_positions": len(m.positions), "orders": orders}


MARKET = ("MKT", "MOO", "MOC")
WIN_BASE = {"BUY": 0.524, "SELL": 0.428}   # win's base rate for a MOO at every 2025 training session and symbol (n 123,804)


def calibration_bonus(orders):
    """The look's bonus: Brier skill of p_win over its filled orders whose win is resolved, mean of
    [1 - (p - y)^2] - [1 - (b - y)^2] with b the side's base rate (WIN_BASE), a missing or invalid p_win scoring 0 in
    place of the first term; 0 if no order is scored. Answering the base rate earns 0 on average, always 0.5 -0.001
    (BUY) or -0.005 (SELL). p_fill is not scored: most fills are known when the order is placed (MKT, MOO and MOC fill, far limits and stops
    don't), so its Brier reward paid for padding the order list with them (review, 18 Oct: four unfillable limits with
    p_fill 0 earned +0.25). p_win needs a fill, which costs commission and risk."""
    import trader
    ys = [(trader.prob(o.get("p_win")), float(o["win"]), WIN_BASE[o["action"]]) for o in orders
          if o.get("filled") and o.get("win") is not None]
    if not ys:
        return {"bonus": 0.0, "n_scored": 0}
    return {"bonus": sum((1 - (p - y) ** 2 if p is not None else 0.0) - (1 - (b - y) ** 2) for p, y, b in ys) / len(ys),
            "n_scored": len(ys)}


def prob_scores(orders):
    """p_fill and p_win apart, over many looks' orders: n scored, share given, Brier reward (missing p = 0), and the
    same items' trivial baselines: always 0.5 (0.75), the best constant (their own base rate, in sample), and the rule
    a forecaster could know beforehand: for p_win the side's 2025 base rate (WIN_BASE, the bonus's reference), for
    p_fill the order type (1 for MKT/MOO/MOC, else the other orders' base rate, in sample). p_fill counts only orders
    whose fill was resolved within the horizon."""
    import trader
    out = {}
    for k, y, due in (("p_fill", "filled", lambda o: o.get("placed") and o.get("filled") is not None),
                      ("p_win", "win", lambda o: o.get("filled") and o.get("win") is not None)):
        xs = [(trader.prob(o.get(k)), float(o[y]), o) for o in orders if due(o)]
        if not xs:
            out[k] = {"n": 0}
            continue
        n = len(xs)
        base = sum(y for _, y, _ in xs) / n
        other = [y for _, y, o in xs if o.get("type") not in MARKET]
        rule = ((lambda o: 1.0 if o.get("type") in MARKET else sum(other) / len(other)) if k == "p_fill"
                else (lambda o: WIN_BASE[o["action"]]))
        br = lambda f: round(sum(1 - (f(o) - y) ** 2 for _, y, o in xs) / n, 4)
        out[k] = {"n": n, "given": round(sum(p is not None for p, _, _ in xs) / n, 4),
                  "brier_reward": round(sum(1 - (p - y) ** 2 if p is not None else 0.0 for p, y, _ in xs) / n, 4),
                  "base_rate": round(base, 4), "best_constant": br(lambda o: base),
                  ("order_type_rule" if k == "p_fill" else "train_base_rate"): br(rule)}
    return out
