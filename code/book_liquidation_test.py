"""book_liquidation.py on IBKR's real liquidation file of Mon 16 Nov 2026 and the sleeves' books as they stood: offline,
on temporary copies of runs/fills.jsonl and runs/notes.json (nothing under runs/ is written; client 11 is not read).

    python book_liquidation_test.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import book_liquidation as B  # noqa: E402
import paper  # noqa: E402

ok = []


def check(name, good, detail=""):
    ok.append(bool(good))
    print(("ok   " if good else "FAIL ") + name + (f": {detail}" if detail and not good else ""))


# ---------------------------------------------------------------- share_out
check("share_out: n covers all -> each its whole position", B.share_out(10, {"a": 4, "b": 6}) == {"a": 4, "b": 6}
      and B.share_out(99, {"a": 4, "b": 6}) == {"a": 4, "b": 6})
s = B.share_out(134, {"base-ml": 20, "base-90k": 100, "base-90k-b": 10, "base-90k-ml": 26, "cand1-ml": 30, "cand2-ml": 30})
check("share_out: SPY's 134 of 216 in proportion, whole shares, summing to 134, none above its position",
      sum(s.values()) == 134 and all(0 < q <= w for q, w in zip(s.values(), [20, 100, 10, 26, 30, 30])), s)
import random  # noqa: E402
rnd, bad = random.Random(7), []
for _ in range(3000):
    w = {f"s{i}": rnd.randint(1, 900) for i in range(rnd.randint(1, 9))}
    n = rnd.randint(1, sum(w.values()) + 50)
    g = B.share_out(n, w)
    if sum(g.values()) != min(n, sum(w.values())) or any(q > w[k] or q < 0 for k, q in g.items()):
        bad.append((n, w, g))
    if B.share_out(n, w) != g:
        bad.append(("not deterministic", n, w))
check("share_out on 3,000 random cases: sums to min(n, total), never above a position, deterministic", not bad, bad[:2])

# ---------------------------------------------------------------- allocate, on the real file and the books as they stood
src = json.loads(B.SRC.read_text(encoding="utf-8"))
fills = src["fills"]
positions = B.positions_before(fills, paper.sleeves())     # as the script takes them: one second before IBKR's first fill
booked_already = any(str(f["exec_id"]).startswith("liq-20261116-") for f in paper.fills())
now_books = {s: {k: v["quantity"] for k, v in paper.book(s)["positions"].items()} for s in paper.sleeves()}
if not booked_already:
    check("before --go, the positions as of 15:51:02 equal the books now in the 12 names (no sleeve fill in them since)",
          all(positions[s].get(k, 0) == now_books[s].get(k, 0) for s in positions for k in {f["symbol"] for f in fills}))
rows, unowned, summary = B.allocate(fills, positions)
bad_rows = []
for bad in ({"liquidation": 0, "client_id": 0}, {"liquidation": 1, "client_id": 10}):
    try:
        B.allocate([dict(fills[0], **bad)], positions)
        bad_rows.append(bad)
    except ValueError:
        pass
try:
    B.allocate([dict(fills[0]), dict(fills[0], side="SLD" if fills[0]["side"] == "BOT" else "BOT", exec_id="x")], positions)
    bad_rows.append("both sides")
except ValueError:
    pass
check("a row that is not an IBKR liquidation fill (liquidation 0, or another client) and a name traded both ways are refused",
      not bad_rows, bad_rows)
OWNER = {"GLD": -219, "INTC": 279, "IONQ": 55, "NVDA": 348, "QQQ": 22, "SOXX": 12, "XLE": -350, "XLK": 110, "XLV": -113}
check("the unowned excess per name equals what IBKR's fills and the sleeves' books leave (and the Owner's figures, with GLD's "
      "-219 = his -231 less fable-ml's short 12 left in place)", unowned == OWNER, unowned)
nets = {}
for f in fills:
    nets[f["symbol"]] = nets.get(f["symbol"], 0) + B.signed(f)
check("per name: booked + unowned = IBKR's net, and the booked side is IBKR's side",
      all(sum(r["shares"] for r in rows if r["symbol"] == k) + abs(unowned.get(k, 0)) == abs(n) for k, n in nets.items())
      and all((r["side"] == "BOT") == (nets[r["symbol"]] > 0) for r in rows))
check("each booked fill reduces its sleeve's position toward zero, never past it",
      all(0 < r["shares"] <= abs(positions[r["sleeve"]][r["symbol"]])
          and (positions[r["sleeve"]][r["symbol"]] < 0) == (r["side"] == "BOT") for r in rows))
check("a sleeve on the other side of a name is untouched (fable-ml's GLD short 12 when IBKR sold GLD)",
      positions["fable-ml"].get("GLD") == -12 and not any(r["sleeve"] == "fable-ml" and r["symbol"] == "GLD" for r in rows))
check("commissions: the booked rows' share of each name's IBKR commission, plus the unowned share, is the whole",
      all(abs(sum(r["commission"] for r in rows if r["symbol"] == k) / max(1, sum(r["shares"] for r in rows if r["symbol"] == k))
              - sum(f["commission"] for f in fills if f["symbol"] == k) / sum(f["shares"] for f in fills if f["symbol"] == k)
              * (sum(f["shares"] for f in fills if f["symbol"] == k) / abs(n))) < 1e-9 for k, n in nets.items()))
check("prices: each row at IBKR's volume-weighted price for its name, at IBKR's last fill time in it",
      all(abs(r["price"] - sum(f["shares"] * f["price"] for f in fills if f["symbol"] == r["symbol"])
              / sum(f["shares"] for f in fills if f["symbol"] == r["symbol"])) < 1e-9
          and r["time"] == max(f["time"] for f in fills if f["symbol"] == r["symbol"]) for r in rows))
recs = [B.fill_record(r) for r in rows]
check("fill records: unique exec_ids liq-20261116-<SYM>-<sleeve>, refs ibkr-liquidation-20261116-<SYM>, sleeve set, "
      "no ref that paper.sleeve_of maps to a sleeve", len({x["exec_id"] for x in recs}) == len(recs)
      and all(x["exec_id"] == f"liq-20261116-{x['symbol']}-{x['sleeve']}" and x["ref"] == f"ibkr-liquidation-20261116-{x['symbol']}"
              and paper.sleeve_of(x["ref"]) is None for x in recs))

# ---------------------------------------------------------------- booked into a copy of the ledger: the books after
tmp = Path(tempfile.mkdtemp())
real_fills, real_notes = paper.FILLS, paper.NOTES
paper.FILLS, paper.NOTES = tmp / "fills.jsonl", tmp / "notes.json"
shutil.copy(real_fills, paper.FILLS)
try:
    if booked_already:                              # start the copy from the ledger before the booking
        keep = [l for l in paper.FILLS.read_text(encoding="utf-8").splitlines() if '"liq-20261116-' not in l]
        paper.FILLS.write_text("\n".join(keep) + "\n", encoding="utf-8")
    before = {s: {k: v["quantity"] for k, v in paper.book(s)["positions"].items()} for s in paper.sleeves()}
    for x in recs:
        paper.append(paper.FILLS, x)
    after = {s: {k: v["quantity"] for k, v in paper.book(s)["positions"].items()} for s in paper.sleeves()}
    want = {s: dict(p) for s, p in before.items()}
    for r in rows:
        q = want[r["sleeve"]][r["symbol"]] + (r["shares"] if r["side"] == "BOT" else -r["shares"])
        if q:
            want[r["sleeve"]][r["symbol"]] = q
        else:
            want[r["sleeve"]].pop(r["symbol"])
    check("after booking, every sleeve's book equals its book before plus its liquidation rows (closed names gone; SPY "
          "partly), and no other sleeve or name changes", after == want,
          {s: (after[s], want[s]) for s in after if after[s] != want[s]})
    check("SPY: the six short sleeves keep 82 shares short between them (216 - 134)",
          sum(after[s].get("SPY", 0) for s in after) == sum(before[s].get("SPY", 0) for s in before) + 134)
    ev = paper.events_since_last_look.__doc__ is not None
    by = {}
    for r in rows:
        by.setdefault(r["sleeve"], []).append(r)
    notes = {s: B.note_for(s, rs, before, [{"symbol": "XLE", "action": "SELL", "quantity": 66, "order_type": "TRAIL",
                                             "tif": "GTC", "ref": "tt-base-ml-3030a5d44d", "price_text": ""}] if s == "base-ml" else [])
             for s, rs in by.items()}
    check(f"a note for each of the {len(notes)} sleeves IBKR touched, each naming every name it closed in that sleeve",
          set(notes) == {r["sleeve"] for r in rows} and all(all(f" {r['symbol']} at $" in notes[s] for r in by[s]) for s in notes))
    check("base-ml's note names its XLE trailing stop as an order still resting in a closed name; a note with no resting "
          "orders says nothing of orders", "SELL 66 XLE TRAIL GTC [tt-base-ml-3030a5d44d]" in notes["base-ml"]
          and "orders still resting" not in notes["fable"])
    check("the notes say what happened and that no model or software chose it, and give no instruction",
          all("IBKR closed positions on its own (a margin liquidation; no model or software chose them)" in t
              and not any(w in t.lower() for w in ("you should", "consider", "please", "cancel")) for t in notes.values()))
finally:
    paper.FILLS, paper.NOTES = real_fills, real_notes
check("nothing under runs/ was written by this test", paper.FILLS == real_fills and not (tmp / "unowned.json").exists())

print("ALL OK" if all(ok) else f"FAILED ({ok.count(False)})")
sys.exit(0 if all(ok) else 1)
