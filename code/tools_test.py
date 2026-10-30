"""tools.run's text on a fake gateway: names on quote lines and scanner rows (industries.json, then contract details,
cached in names.json), the bars and article cut notes, the empty news line. Offline; lists/ is copied to a temp dir.

    python tools_test.py
"""
import json
import shutil
import tempfile
from pathlib import Path

import tools


class FakeGW:
    def __init__(self):
        self.detail_calls = []

    def quotes(self, syms):
        return [{"symbol": s, "error": "200: No security definition"} if s == "SGOV.US" else
                {"symbol": s, "last": 100.5, "close": 100.0, "bid": 100.4, "ask": 100.6, "volume": 1000} for s in syms]

    def details(self, s):
        self.detail_calls.append(s)
        known = {"NANO": {"long_name": "NANO NUCLEAR ENERGY INC", "industry": "Industrial", "stock_type": "COMMON"},
                 "SGOV": {"long_name": "ISHARES 0-3 MONTH TREASURY BOND ETF", "industry": "", "stock_type": "ETF"}}
        if s not in known:
            raise RuntimeError("200: no contract")
        return {"symbol": s, **known[s]}

    def bars(self, sym, duration, bar_size, what, rth):
        return [{"date": f"2026-08-{i % 28 + 1:02d}", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1} for i in range(63)]

    def news(self, sym, limit):
        return []

    def article(self, provider, article_id):
        return {"text": "x" * 5000}

    def scan(self, code, rows, **f):
        return [{"rank": 0, "symbol": "NANO"}, {"rank": 1, "symbol": "AAPL"}]


def check(name, got, want=(), absent=()):
    ok = all(w in got for w in want) and not any(a in got for a in absent)
    print(("ok   " if ok else "FAIL ") + name + ": " + got.replace("\n", " / ")[:300])
    return ok


with tempfile.TemporaryDirectory() as d:
    shutil.copy(tools.LISTS / "industries.json", d)
    tools.LISTS, tools._names = Path(d), None
    tools.news_live = False
    gw = FakeGW()
    gw.news_archive = True                     # the simulator path for news/article (no livenews)
    sim = [check("quote, sim: name from industries.json only", tools.run(gw, "quote", {"symbols": ["AAPL", "NANO"]}),
                 ["AAPL: last 100.50", "| APPLE INC (Technology)"], ["NANO NUCLEAR"])]
    del gw.news_archive
    tools._names = None
    ok = sim + [
        check("quote, live: details for a name not in industries.json", tools.run(gw, "quote", {"symbols": ["NANO", "SGOV", "SGOV.US"]}),
              ["NANO: last 100.50", "| NANO NUCLEAR ENERGY INC (Industrial)", "| ISHARES 0-3 MONTH TREASURY BOND ETF (ETF)",
               "SGOV.US error"]),
        check("names cached in names.json", json.dumps(json.loads((Path(d) / "names.json").read_text(encoding="utf-8"))),
              ["NANO", "SGOV"]),
        check("scanner rows carry names, no second details call", tools.run(gw, "scanner", {"scan": "TOP_PERC_GAIN"})
              + f" calls={gw.detail_calls}", ["rank sym last today% name", "NANO NUCLEAR ENERGY INC (Industrial)",
                                               "APPLE INC (Technology)", "calls=['NANO', 'SGOV']"]),
        check("bars cut note", tools.run(gw, "bars", {"symbol": "SGOV", "bar_size": "1 day", "duration": "3 M"}),
              ["(the last 60 of 63 bars)"]),
    ]
    gw.news_archive = True
    ok += [check("news empty", tools.run(gw, "news", {"symbol": "XLE"}), ["no headlines for XLE (ETFs and funds usually"]),
           check("article cut note", tools.run(gw, "article", {"provider": "BRFG", "article_id": "1"})[-60:],
                 ["[cut at 4,000 of 5,000 characters]"])]
print("ALL OK" if all(ok) else "FAILED")
