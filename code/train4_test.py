"""rl5's EMPTY_COST rule (train4.substantive / order_stats): token actions count as empty, real ones do not.

    python train4_test.py      # prints ALL OK
"""
import trader, intrasim, train4
data = trader.Data(); intra = intrasim.Intra(data); d = data.ix["20260415"]
w = trader.SimWorld(data, d - 1, 30000.0)
w.act({"place": [{"symbol": "AAPL", "action": "BUY", "quantity": 40, "type": "MOO", "tif": "DAY"}], "cancel": []})
w.advance()
iw = intrasim.IntraWorld(w, intra); feats, _ = data.features(d)
r1 = {"NVDA": 0.03, "AAPL": 0.025}
nv, aa = iw.m.last("NVDA"), iw.m.last("AAPL")
iw.act({"place": [{"symbol": "NVDA", "action": "BUY", "quantity": 10, "type": "LMT", "limit": round(nv * (1 - 3 * 0.03), 2), "tif": "GTC"},
                  {"symbol": "NVDA", "action": "BUY", "quantity": 10, "type": "LMT", "limit": round(nv * (1 - 0.5 * 0.03), 2), "tif": "GTC"}],
        "cancel": []})
st = iw.state(d, feats)
far_ref, near_ref = st["orders"][0]["ref"], st["orders"][1]["ref"]
cases = {
    "cancel far order": {"place": [], "cancel": [far_ref]},
    "cancel near order": {"place": [], "cancel": [near_ref]},
    "LMT $1k at 1.5 r1": {"place": [{"symbol": "NVDA", "action": "BUY", "quantity": 8, "type": "LMT", "limit": round(nv * (1 - 1.5 * 0.03), 2), "tif": "DAY"}], "cancel": []},
    "LMT $1k at 0.8 r1": {"place": [{"symbol": "NVDA", "action": "BUY", "quantity": 8, "type": "LMT", "limit": round(nv * (1 - 0.8 * 0.03), 2), "tif": "DAY"}], "cancel": []},
    "full stop at 1.8 r1": {"place": [{"symbol": "AAPL", "action": "SELL", "quantity": 40, "type": "STP", "stop": round(aa * (1 - 1.8 * 0.025), 2), "tif": "GTC"}], "cancel": []},
    "TRAIL 3% (1.2 r1)": {"place": [{"symbol": "AAPL", "action": "SELL", "quantity": 40, "type": "TRAIL", "trail_percent": 3, "tif": "GTC"}], "cancel": []},
    "TRAIL 8% (3.2 r1)": {"place": [{"symbol": "AAPL", "action": "SELL", "quantity": 40, "type": "TRAIL", "trail_percent": 8, "tif": "GTC"}], "cancel": []},
    "MKT $1k": {"place": [{"symbol": "NVDA", "action": "BUY", "quantity": 6, "type": "MKT", "tif": "DAY"}], "cancel": []},
    "1-share MKT": {"place": [{"symbol": "NVDA", "action": "BUY", "quantity": 1, "type": "MKT", "tif": "DAY"}], "cancel": []},
    "exit 40 AAPL MKT": {"place": [{"symbol": "AAPL", "action": "SELL", "quantity": 40, "type": "MKT", "tif": "DAY"}], "cancel": []},
}
expect = {"cancel far order": True, "cancel near order": False, "LMT $1k at 1.5 r1": True, "LMT $1k at 0.8 r1": False,
          "full stop at 1.8 r1": False, "TRAIL 3% (1.2 r1)": False, "TRAIL 8% (3.2 r1)": True, "MKT $1k": False,
          "1-share MKT": True, "exit 40 AAPL MKT": False}
bad = 0
for k, call in cases.items():
    call["notes"] = "n"
    x = train4.order_stats(call, st, iw, [], r1)
    ok = x["empty"] == expect[k]
    bad += not ok
    print(f"{k:22} empty={x['empty']!s:5} expected={expect[k]!s:5} {'ok' if ok else 'FAIL'} dist={[round(v, 2) for v in x['limit_dist_r1']]}")
print("ALL OK" if not bad else f"{bad} FAILED")
