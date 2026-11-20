"""carry_record on a small synthetic repo tree (ledgers, look records, bars) in a temporary folder. Offline.

    python carry_record_test.py
"""
import csv
import json
import tempfile
from pathlib import Path

import carry_record as C

DAYS = ["20261102", "20261103", "20261104", "20261105", "20261106", "20261109", "20261110"]


def check(name, ok, got=""):
    print(("ok   " if ok else "FAIL ") + name + (f": {got}" if got != "" else ""))
    return ok


def wj(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def wl(p, rows, tail=""):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows) + tail, encoding="utf-8")


def daily(root, sym, rows):
    wj(root / "data/bars" / C.fname(sym), {"symbol": sym, "what": "ADJUSTED_LAST",
       "bars": [{"date": d, "open": o, "high": max(o, c), "low": min(o, c), "close": c, "volume": 1000} for d, (o, c) in sorted(rows.items())]})


def bars30(root, sym, rows):
    b = []
    for d, (o, c) in sorted(rows.items()):
        b += [{"date": f"{d} 09:30:00 US/Eastern", "open": o, "close": o, "volume": 10},
              {"date": f"{d} 15:30:00 US/Eastern", "open": c, "close": c, "volume": 10}]
    wj(root / "data/bars30" / C.fname(sym), {"symbol": sym, "bars": b})


N = [0]


def fill(sleeve, sym, side, n, px, day, t="10:00:00"):
    N[0] += 1
    return {"exec_id": f"e{N[0]}", "sleeve": sleeve, "symbol": sym, "side": side, "shares": n, "price": px,
            "commission": 1.0, "ref": f"tt-{sleeve}-{N[0]:010x}", "time": f"{day} {t} US/Eastern"}


def look(root, day, stem, positions, hhmm):
    wj(root / "runs/live" / f"{day[:4]}-{day[4:6]}-{day[6:]}" / f"{stem}.json",
       {"time": f"{day[:4]}-{day[4:6]}-{day[6:]}T{hhmm[:2]}:{hhmm[2:]}:01-05:00", "state": {
           "value": 30000.0, "positions": [{"symbol": s, "quantity": q, "close": c} for s, q, c in positions]}})


def tree(root):
    """Two weeks. Sleeve a runs 2-10 Nov (archived week, then live: one run). Sleeve b restarts on Mon 9 Nov."""
    daily(root, "SPY", {d: (500.0 + i, 501.0 + i) for i, d in enumerate(DAYS)})
    bars30(root, "SPY", {d: (500.0 + i, 501.0 + i) for i, d in enumerate(DAYS)})
    # LNG: long carried 3->4 Nov (10 x (103 - 101) = +20) and over the weekend 6->9 Nov
    daily(root, "LNG", {"20261103": (100.0, 101.0), "20261104": (103.0, 104.0), "20261105": (104.0, 104.0),
                        "20261106": (104.0, 105.0), "20261109": (107.0, 106.0), "20261110": (106.0, 106.0)})
    bars30(root, "LNG", {"20261103": (100.0, 101.0), "20261104": (103.0, 104.0), "20261105": (104.0, 104.0),
                         "20261106": (104.0, 105.0), "20261109": (107.0, 106.0), "20261110": (106.0, 106.0)})
    # SHT: short 5 carried 3->4 Nov: -5 x (49 - 50.5) = +7.5
    daily(root, "SHT", {"20261103": (50.0, 50.5), "20261104": (49.0, 49.0)})
    bars30(root, "SHT", {"20261103": (50.0, 50.5), "20261104": (49.0, 49.0)})
    # DAY: opened and closed on 3 Nov, never carried
    daily(root, "DAY", {"20261103": (20.0, 20.0)})
    bars30(root, "DAY", {"20261103": (20.0, 20.0)})
    # SPL: 3-for-1 from 5 Nov. data/bars is split-adjusted (4 Nov close 100), the 30-minute bars of 4 Nov are not (297):
    # k = 3, raw close 300; 9 shares become 27 at 100; gap 27 x (99 - 100) = -27
    daily(root, "SPL", {"20261104": (99.0, 100.0), "20261105": (99.0, 98.0)})
    bars30(root, "SPL", {"20261104": (297.0, 300.0), "20261105": (99.0, 98.0)})
    # DIV: ex 5 Nov, 0.50 a share. data/bars before 5 Nov is dividend-adjusted by 0.99: raw close 4 Nov = 50.0;
    # gap 100 x (49.6 - 50.0) = -40; the 50.00 dividend is its own column
    daily(root, "DIV", {"20261104": (49.698, 49.5), "20261105": (49.6, 49.7)})
    bars30(root, "DIV", {"20261104": (50.2, 50.0), "20261105": (49.6, 49.7)})
    # NOB: no 30-minute bars at all, no arrival quote: k = 1, open from the daily bar, marked approximate: 10 x (31 - 30)
    daily(root, "NOB", {"20261104": (30.0, 30.0), "20261105": (31.0, 32.0)})
    # NOO: no bar on 6 Nov and no early fill: no open (gap 0, noopen); 6 Nov's close from the 9 Nov 09:00 look marks
    daily(root, "NOO", {"20261105": (20.0, 20.0)})
    # FRI: b's Friday book (restarted Monday): in no run
    daily(root, "FRI", {"20261106": (10.0, 10.0), "20261109": (12.0, 12.0)})
    bars30(root, "FRI", {"20261106": (10.0, 10.0), "20261109": (12.0, 12.0)})

    start1 = "2026-11-01T12:00:00+00:00"
    start2 = "2026-11-08T12:00:00+00:00"
    wk1 = [fill("a", "LNG", "BOT", 10, 100.0, "20261103"), fill("a", "SHT", "SLD", 5, 50.0, "20261103"),
           fill("a", "DAY", "BOT", 7, 20.0, "20261103", "11:00:00"), fill("a", "DAY", "SLD", 7, 20.0, "20261103", "14:00:00"),
           fill("a", "SHT", "BOT", 5, 49.0, "20261104"),
           fill("a", "SPL", "BOT", 9, 297.0, "20261104"), fill("a", "DIV", "BOT", 100, 50.0, "20261104"),
           fill("a", "NOB", "BOT", 10, 30.0, "20261104"),
           fill("a", "SPL", "SLD", 27, 98.0, "20261105"), fill("a", "DIV", "SLD", 100, 49.7, "20261105"),
           fill("a", "NOB", "SLD", 10, 32.0, "20261105"), fill("a", "NOO", "BOT", 5, 20.0, "20261105"),
           fill("a", "NOO", "SLD", 5, 21.0, "20261106", "10:15:00"),
           fill("b", "FRI", "BOT", 100, 10.0, "20261106")]
    arch = root / "runs/archive/week1"
    wj(arch / "sleeves.json", {"a": {"start_cash": 30000.0, "start": start1, "model": "base", "ml": True, "ml_tool": "ML2"},
                               "b": {"start_cash": 30000.0, "start": start1, "model": "base"}})
    wl(arch / "fills.jsonl", wk1)
    wl(arch / "daily.jsonl", [{"date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "sleeve": s} for d in DAYS[:5] for s in "ab"])
    wl(arch / "dividends.jsonl", [{"sleeve": "a", "symbol": "DIV", "ex_date": "20261105", "quantity": 100, "per_share": 0.5,
                                   "amount": 50.0}])
    wj(root / "runs/sleeves.json", {"a": {"start_cash": 30000.0, "start": start1, "model": "base", "ml": True, "ml_tool": "ML2"},
                                    "b": {"start_cash": 30000.0, "start": start2, "model": "base"}})
    # the carried sleeve's rows stay in the live ledger too (reset_week --carry): the same exec_ids, booked once
    wl(root / "runs/fills.jsonl", [f for f in wk1 if f["sleeve"] == "a"] + [fill("a", "LNG", "SLD", 4, 106.5, "20261109")])
    wl(root / "runs/splits.jsonl", [{"symbol": "SPL", "ex_date": "20261105", "ratio": 3}])
    # look records: 3 Nov's last look agrees with the fills before it; 4 Nov's first look shows SHT wrong
    look(root, "20261103", "a-1530", [("LNG", 10, 101.0), ("SHT", -5, 50.5)], "1530")
    look(root, "20261104", "a-0900", [("LNG", 10, 101.0), ("SHT", -4, 50.5)], "0900")
    look(root, "20261109", "c-0900", [("NOO", 5, 20.5)], "0900")


def rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


ok = []
with tempfile.TemporaryDirectory() as d:
    root = Path(d) / "repo"
    tree(root)
    out = Path(d) / "carry"
    prow, nrow = C.record(root, out)
    P = {(r["sleeve"], r["D"], r["sym"]): r for r in prow}
    NT = {(r["sleeve"], r["D"]): r for r in nrow}
    g = lambda s, D, x: P[(s, D, x)]["pnl_gap"] if (s, D, x) in P else None
    ok += [check("a long carried 3->4 Nov: 10 x (103 - 101)", abs(g("a", "20261104", "LNG") - 20.0) < 1e-9, g("a", "20261104", "LNG")),
           check("a short carried 3->4 Nov: -5 x (49 - 50.5)", abs(g("a", "20261104", "SHT") - 7.5) < 1e-9, g("a", "20261104", "SHT")),
           check("the short's cover on 4 Nov is its session leg: -5 x (49 - 49) + 5 x (49 - 49)",
                 abs(P[("a", "20261104", "SHT")]["pnl_sess"]) < 1e-9 and P[("a", "20261104", "SHT")]["exited"] == -5),
           check("opened and closed the same day: not carried", ("a", "20261104", "DAY") not in P),
           check("the weekend night 6->9 Nov of a continuing run: 10 x (107 - 105), weekend",
                 abs(g("a", "20261109", "LNG") - 20.0) < 1e-9 and P[("a", "20261109", "LNG")]["weekend"] is True
                 and NT[("a", "20261109")]["weekend"] is True),
           check("a weekday night is not a weekend", P[("a", "20261104", "LNG")]["weekend"] is False),
           check("a split between P and D: 27 shares at 100, gap 27 x (99 - 100)",
                 P[("a", "20261105", "SPL")]["q"] == 27 and abs(P[("a", "20261105", "SPL")]["CP"] - 100.0) < 1e-9
                 and abs(g("a", "20261105", "SPL") + 27.0) < 1e-9, (P[("a", "20261105", "SPL")]["q"], g("a", "20261105", "SPL"))),
           check("a dividend on D: gap on unadjusted prices 100 x (49.6 - 50.0), the 50.00 in div_usd",
                 abs(g("a", "20261105", "DIV") + 40.0) < 1e-6 and abs(NT[("a", "20261105")]["div_usd"] - 50.0) < 1e-9,
                 (g("a", "20261105", "DIV"), NT[("a", "20261105")]["div_usd"])),
           check("no 30-minute bars: daily open, marked approximate",
                 abs(g("a", "20261105", "NOB") - 10.0) < 1e-9 and P[("a", "20261105", "NOB")]["approx_open"] is True
                 and P[("a", "20261105", "NOB")]["open_src"] == "daily_k" and NT[("a", "20261105")]["n_approx"] == 1
                 and abs(NT[("a", "20261105")]["approx_usd"] - 300.0) < 1e-9),
           check("an open from the 09:30 bar is not approximate", P[("a", "20261105", "DIV")]["approx_open"] is False),
           check("no open at all: gap 0, noopen, the close-to-close in the session leg",
                 g("a", "20261106", "NOO") == 0.0 and P[("a", "20261106", "NOO")]["noopen"] is True
                 and P[("a", "20261106", "NOO")]["approx_open"] is False and NT[("a", "20261106")]["n_noopen"] == 1
                 and abs(P[("a", "20261106", "NOO")]["pnl_sess"] - 5.0) < 1e-9, P.get(("a", "20261106", "NOO"))),
           check("a restarted sleeve's Friday book is in no run: no night into Monday for b",
                 ("b", "20261109") not in NT and not any(k[2] == "FRI" for k in P)),
           check("b's nights of its first week are recorded", ("b", "20261106") in NT and NT[("b", "20261106")]["positions"] == 0),
           check("one run for a across the archive and the live ledger", {r["run"] for r in nrow if r["sleeve"] == "a"} == {"a/W4"}
                 and NT[("a", "20261109")]["ml_tool"] == "ML2" and NT[("a", "20261109")]["fam"] == "base+tool"),
           check("duplicate exec_ids booked once: the weekend book is 10 LNG", P[("a", "20261109", "LNG")]["q"] == 10),
           check("the last look of 3 Nov agrees", NT[("a", "20261104")]["last_look_ok"] is True),
           check("the first look of 4 Nov differs (SHT -4 against -5)", NT[("a", "20261104")]["first_look_ok"] is False),
           check("no look file: no check", NT[("a", "20261105")]["first_look_ok"] is None)]
    n4 = NT[("a", "20261104")]
    # value at the close of 3 Nov: 30000 - 10 x 100 + 5 x 50 - 4 commissions (DAY in and out at 20) + 10 x 101 - 5 x 50.5
    v = 30000 - 1000 + 250 - 4.0 + 1010 - 252.5
    ok += [check("value at the close of P", abs(n4["value_P"] - v) < 1e-9, n4["value_P"]),
           check("gross, net and their multiples", abs(n4["gross_usd"] - 1262.5) < 1e-9 and abs(n4["net_usd"] - 757.5) < 1e-9
                 and abs(n4["gross_x"] - 1262.5 / v) < 1e-12 and abs(n4["net_x"] - 757.5 / v) < 1e-12),
           check("gap in $ and as % of value", abs(n4["gap_usd"] - 27.5) < 1e-9 and abs(n4["gap_pct"] - 100 * 27.5 / v) < 1e-12),
           check("no beta history: beta 1.0, so beta_net = net_x; chip 0", abs(n4["beta_net"] - n4["net_x"]) < 1e-12
                 and n4["chip_tilt"] == 0.0),
           check("top names by weight", n4["top5"].startswith("LNG L ") and "SHT S " in n4["top5"], n4["top5"])]

    # files: re-running a night replaces its rows
    a1 = rows(out / "positions.csv")
    n1 = rows(out / "nights.csv")
    C.record(root, out, {"20261105"})
    C.record(root, out, {"20261105"})
    a2, n2 = rows(out / "positions.csv"), rows(out / "nights.csv")
    ok += [check("re-running a night replaces its rows", len(a2) == len(a1) and len(n2) == len(n1)
                 and sum(r["D"] == "20261105" for r in a2) == sum(r["D"] == "20261105" for r in a1), (len(a1), len(a2))),
           check("approx_open is a column of positions.csv", "approx_open" in a2[0] and a2[0]["approx_open"] in ("True", "False"))]

    # update(): only the nights not yet recorded, only once their bars are in
    out2 = Path(d) / "carry2"
    s1 = C.update("2026-11-05", root, out2)
    have = {r["D"] for r in rows(out2 / "nights.csv")}
    s2 = C.update("2026-11-05", root, out2)
    ok += [check("update records the nights up to the day", have == {"20261103", "20261104", "20261105"}, (s1, have)),
           check("update again: nothing new", s2.startswith("carry: nothing new"), s2)]
    spy30 = json.loads((root / "data/bars30/SPY.json").read_text())
    spy30["bars"] = [b for b in spy30["bars"] if b["date"][:8] < "20261106"]
    wj(root / "data/bars30/SPY.json", spy30)
    s3 = C.update("2026-11-06", root, out2)
    ok += [check("the night into 6 Nov waits for its 30-minute bars", "waiting for bars: 20261106" in s3
                 and "20261106" not in {r["D"] for r in rows(out2 / "nights.csv")}, s3)]

    # update() never raises
    half = root / "runs/fills.jsonl"
    half.write_text(half.read_text() + '{"exec_id": "e99", "time": "20261110 10:0', encoding="utf-8")
    s4 = C.update("2026-11-10", root, Path(d) / "carry3")
    ok += [check("a half-written fills line: nothing recorded, the line reported", s4.startswith("carry: nothing recorded; waiting")
                 and "unreadable" in s4 and not (Path(d) / "carry3" / "nights.csv").exists(), s4)]
    (root / "runs/sleeves.json").write_text('{"a": {"start_cash": 3', encoding="utf-8")
    s5 = C.update("2026-11-10", root, Path(d) / "carry4")
    s6 = C.update("2026-11-10", Path(d) / "nothing-here", Path(d) / "carry5")
    (root / "runs/sleeves.json").write_text(json.dumps({"a": {"start_cash": 30000.0, "start": "2026-11-01T12:00:00+00:00",
                                                              "model": "base"}}), encoding="utf-8")
    (root / "data/bars/SPY.json").write_text("", encoding="utf-8")
    s7 = C.update(None, root, Path(d) / "carry6")
    ok += [check("a half-written sleeves.json: a status, no exception", s5.startswith("carry: failed"), s5),
           check("no repo at all: a status, no exception", isinstance(s6, str) and s6.startswith("carry:"), s6),
           check("an empty SPY bar file: a status, no exception", s7 == "carry: no SPY daily bars; nothing recorded", s7)]

    def fresh(name):
        r = Path(d) / name
        tree(r)
        return r

    # labels come from each night's own day, not from the latest sleeves.json: a is held in the live sleeves.json
    # (model "hold", model_held "base") and its ml_tool there is now ML3
    r1 = fresh("labels")
    wj(r1 / "runs/sleeves.json", {"a": {"start_cash": 30000.0, "start": "2026-11-01T12:00:00+00:00", "model": "hold",
                                        "model_held": "base", "ml": True, "ml_tool": "ML3"},
                                  "b": {"start_cash": 30000.0, "start": "2026-11-08T12:00:00+00:00", "model": "base"}})
    wl(r1 / "runs/archive/week1/daily.jsonl", [{"date": f"{x[:4]}-{x[4:6]}-{x[6:]}", "sleeve": s, "model": "base",
                                                **({"ml_tool": "ML2.1"} if s == "a" and x == "20261104" else {})}
                                               for x in DAYS[:5] for s in "ab"])
    _, nr1 = C.record(r1, Path(d) / "carry-labels")
    A = {r["D"]: r for r in nr1 if r["sleeve"] == "a"}
    ok += [check("a held sleeve's nights keep their family (model_held)", {r["fam"] for r in A.values()} == {"base+tool"},
                 sorted({r["fam"] for r in A.values()})),
           check("ml_tool of an archived night: that week's sleeves.json", A["20261104"]["ml_tool"] == "ML2", A["20261104"]["ml_tool"]),
           check("ml_tool from P's daily record where it names one", A["20261105"]["ml_tool"] == "ML2.1", A["20261105"]["ml_tool"]),
           check("ml_tool of a night in the live week: the live sleeves.json", A["20261110"]["ml_tool"] == "ML3", A["20261110"]["ml_tool"])]

    # a missing ledger: nothing is recorded, and the status says which
    r2 = fresh("noledger")
    (r2 / "runs/sleeves.json").unlink()
    s8 = C.update("2026-11-10", r2, Path(d) / "carry-n1")
    r3 = fresh("noarchfills")
    (r3 / "runs/archive/week1/fills.jsonl").unlink()
    s9 = C.update("2026-11-10", r3, Path(d) / "carry-n2")
    ok += [check("no runs/sleeves.json: nothing recorded", "waiting for ledgers: live/sleeves.json" in s8
                 and not (Path(d) / "carry-n1" / "nights.csv").exists(), s8),
           check("no archived fills.jsonl: nothing recorded", "waiting for ledgers: week1/fills.jsonl" in s9
                 and not (Path(d) / "carry-n2" / "nights.csv").exists(), s9)]

    # a daily bar that comes a day late: the night is recorded at once, then re-recorded when the bar is in
    r4 = fresh("latebar")
    with (r4 / "runs/archive/week1/fills.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(fill("a", "NOX", "BOT", 10, 40.0, "20261104")) + "\n")
        f.write(json.dumps(fill("a", "NOX", "SLD", 10, 43.0, "20261105", "10:15:00")) + "\n")
    daily(r4, "NOX", {"20261104": (40.0, 40.0)})
    o4 = Path(d) / "carry-late"
    s10 = C.update("2026-11-05", r4, o4)
    x0 = [r for r in rows(o4 / "positions.csv") if r["sym"] == "NOX"]
    daily(r4, "NOX", {"20261104": (40.0, 40.0), "20261105": (42.0, 43.0)})
    s11 = C.update("2026-11-06", r4, o4)
    x1 = [r for r in rows(o4 / "positions.csv") if r["sym"] == "NOX"]
    ok += [check("a night with a missing daily bar: recorded, bars_ok False", len(x0) == 1 and x0[0]["bars_ok"] == "False"
                 and abs(float(x0[0]["pnl_gap"]) - 30.0) < 1e-9, (s10, x0)),
           check("the bar comes: the night is re-recorded from it, once", len(x1) == 1 and x1[0]["bars_ok"] == "True"
                 and abs(float(x1[0]["pnl_gap"]) - 20.0) < 1e-9 and "re-recorded for late daily bars: 20261105" in s11, (s11, x1)),
           check("the next update does not re-record it", "re-recorded" not in C.update("2026-11-06", r4, o4))]

    # archive sets in week order: week10 after week2, so week10's record of an exec_id wins
    r5 = fresh("order")
    (r5 / "runs/archive/week1").rename(r5 / "runs/archive/week2")
    w10 = r5 / "runs/archive/week10"
    first = json.loads((r5 / "runs/archive/week2/fills.jsonl").read_text().splitlines()[-1])     # b's: not in the live ledger
    wj(w10 / "sleeves.json", {"b": {"start_cash": 30000.0, "start": "2026-11-01T12:00:00+00:00", "model": "base"}})
    wl(w10 / "fills.jsonl", [{**first, "price": 99.0}])
    L5 = C.Ledgers(r5)
    ok += [check("archive sets in week order", [x[0] for x in L5.sets] == ["week2", "week10", "live"], [x[0] for x in L5.sets]),
           check("the later week's record of an exec_id wins", [f["price"] for f in L5.fills if f["exec_id"] == first["exec_id"]] == [99.0])]

    # look files of an unexpected shape are skipped, not fatal
    r6 = fresh("looks")
    wj(r6 / "runs/live/2026-11-09/z-0900.json", [1, 2, 3])
    wj(r6 / "runs/live/2026-11-06/a-0900.json", {"time": "2026-11-06T09:00:01-05:00", "state": {"positions": [{"quantity": 5}]}})
    s12 = C.update("2026-11-10", r6, Path(d) / "carry-looks")
    ok += [check("a look that is a JSON list, and a position with no symbol: recorded", s12.startswith("carry: recorded"), s12)]

    # recent_names: what live.refresh() adds to its bars refresh (16 Nov 2026), from every ledger set
    r7 = fresh("recent")
    wl(r7 / "runs/fills.jsonl", [fill("a", "NEWCO", "BOT", 5, 10.0, "20261112"), fill("a", "NEWCO", "SLD", 5, 11.0, "20261113"),
                                 fill("a", "OLDCO", "BOT", 5, 10.0, "20261020")])
    (r7 / "runs/archive/week5").mkdir(parents=True, exist_ok=True)
    wl(r7 / "runs/archive/week5/fills.jsonl", [fill("b", "ARCH", "BOT", 3, 20.0, "20261110")])
    got = C.recent_names(10, r7, "2026-11-16")
    ok += [check("recent_names: names with a fill in the last 10 days, the archives' too, not older ones",
                 {"NEWCO", "ARCH"} <= got and "OLDCO" not in got, sorted(got))]
    ok += [check("recent_names never raises: a folder that is not there gives an empty set",
                 C.recent_names(10, Path(d) / "no-such-root", "2026-11-16") == set())]
    wl(r7 / "runs/splits.jsonl", [{"symbol": "NEWCO", "ex_date": "20261113", "ratio": 3}, {"symbol": "NEWCO", "ex_date": "20261113", "ratio": 3}])
    ok += [check("a split row written twice is read once", len(C.Ledgers(r7).splits) == 1)]
print("ALL OK" if all(ok) else "FAILED")
