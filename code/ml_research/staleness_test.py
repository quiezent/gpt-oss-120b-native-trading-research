"""forecast.run's staleness label under a mocked clock (F12 in runs/ml/research/forecast_model_acceptance_criteria.md).

    .venv\\Scripts\\python.exe ml_research\\staleness_test.py

The label ("(stale: not from the last session)") compares the cache's close with the last session before today in New
York: _prev_session(_today_et()), skipping weekends and live.HOLIDAYS. The machine runs UTC+8, so after 12:00 ET (11:00
in winter) its own date is already tomorrow; until 21 Oct that made every result after 12:00 ET read stale. A Sunday
smoke cannot see this, so the clock is mocked: datetime.datetime.now (what _today_et reads) returns a fixed instant in
the zone asked for, or the machine's UTC+8 wall time when none is given, and forecast.date.today returns the machine's
date, so a regression to the local date fails here (checked below by putting the pre-fix _today_et back).

Offline: the forecast cache is a one-row file in a temp folder; the 8-K check, the macro line and the session ranges are
stubbed, and the gateway refuses every attribute.
"""

import datetime as dtmod
import json
import sys
import tempfile
from datetime import date, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import forecast  # noqa: E402
from ml_research import events, macro_feats  # noqa: E402

REAL_DT, NY, MYT = dtmod.datetime, ZoneInfo("America/New_York"), timezone(timedelta(hours=8))
STALE = "(stale: not from the last session)"

# (ET wall clock, cache close, labelled stale?, why)
CASES = [
    ("2026-10-21 13:00", "20261020", False, "21 Oct bug: after 12:00 ET the machine's date is 22 Oct"),
    ("2026-10-21 13:00", "20261019", True, "same clock, a cache one session behind"),
    ("2026-10-21 11:59", "20261020", False, "one minute before the machine's date turns"),
    ("2026-10-21 12:00", "20261020", False, "the minute it turns (00:00 UTC+8)"),
    ("2026-10-21 23:30", "20261020", False, "late evening ET"),
    ("2026-11-03 11:30", "20261102", False, "winter time: the machine's date turns at 11:00 ET"),
    ("2026-10-26 08:00", "20261023", False, "Monday after the 07:30 refresh: Friday's close"),
    ("2026-10-26 07:00", "20261022", True, "Monday before the refresh: the cache ends Thursday"),
    ("2026-10-26 14:00", "20261023", False, "Monday after 12:00 ET (Tuesday on the machine)"),
    ("2026-10-25 10:00", "20261023", False, "Sunday: the last session is Friday"),
    ("2026-10-25 10:00", "20261022", True, "Sunday with Thursday's cache (the Sunday smoke before Monday's refresh)"),
    ("2026-11-27 08:00", "20261125", False, "the day after Thanksgiving (26 Nov): the last session is Wednesday"),
    ("2026-11-27 08:00", "20261124", True, "the day after Thanksgiving, a cache one session behind"),
    ("2026-11-27 13:00", "20261125", False, "the day after Thanksgiving after the machine's date turned (Saturday)"),
    ("2026-12-28 08:00", "20261224", False, "Monday after Christmas (Friday): the last session is Thursday 24 Dec"),
    ("2027-01-19 08:00", "20270115", False, "Tuesday after Martin Luther King Day: weekend plus holiday"),
    ("2027-01-19 08:00", "20270114", True, "the same, a cache one session behind"),
    ("2026-10-22 09:00", "20261021", False, "a fresh cache on an ordinary morning"),
    ("2026-10-22 20:00", "20261022", False, "a cache newer than the last session"),
]


class Clock:
    """Freezes the clock at an ET wall time for _today_et (datetime.datetime.now) and for date.today in forecast."""

    def __init__(self, et):
        self.utc = REAL_DT.fromisoformat(et).replace(tzinfo=NY).astimezone(timezone.utc)

    def __enter__(self):
        utc = self.utc

        class FakeDT(REAL_DT):
            @classmethod
            def now(cls, tz=None):
                return utc.astimezone(tz) if tz else utc.astimezone(MYT).replace(tzinfo=None)

            @classmethod
            def today(cls):
                return cls.now()

            @classmethod
            def utcnow(cls):
                return utc.replace(tzinfo=None)

        class FakeDate(date):
            @classmethod
            def today(cls):
                return utc.astimezone(MYT).date()

        self.saved = dtmod.datetime, forecast.date
        dtmod.datetime, forecast.date = FakeDT, FakeDate
        return self

    def __exit__(self, *exc):
        dtmod.datetime, forecast.date = self.saved


class NoGW:
    """No gateway: every attribute is missing (hasattr works; any use would fail)."""

    def __getattr__(self, name):
        raise AttributeError(f"offline test: no gateway ({name})")


def head(as_of, tmp):
    """The first line of forecast.run's result with a one-row cache whose close is as_of."""
    forecast.OUT = tmp / as_of
    forecast.OUT.mkdir(parents=True, exist_ok=True)
    row = {"close": 100.0, "range1_pct": 2.0, "range1_lo": 1.5, "range1_hi": 3.0, "absr1_pct": 1.2, "vol20_pct": 25.0,
           "note": "", "spread": None}
    market = {"spy_vol20_pct": 15.0, "spy_vol20_10_90_pct": [12, 19], "spy_vol5_pct": 14.0, "typical_spy_vol_pct": 16.0,
              "size_scale_vs_typical": 1.07}
    (forecast.OUT / f"fc_{as_of}.json").write_text(json.dumps({"as_of": as_of, "market": market, "forecasts": {"ZZ": row}}),
                                                   encoding="utf-8")
    txt = forecast.run(NoGW(), {"symbols": ["ZZ"]})
    assert "ZZ 100 1(0.75-1.5) 1.2 25 na na na na" in txt, txt
    return txt.splitlines()[0]


def main():
    import live                                  # _prev_session drops every holiday silently if this import fails
    assert {"2026-11-26", "2026-12-25", "2027-01-18"} <= live.HOLIDAYS, sorted(live.HOLIDAYS)
    forecast._session = lambda gw, syms, iv_hint=None: ("session ranges from 09:30 ET", {})
    events.live = lambda gw, syms, base=None, **kw: {"rows": {}}
    macro_feats.macro_today = lambda *a, **k: "macro: (stub)"
    tmp = Path(tempfile.mkdtemp())
    bad = []
    for et, as_of, want, why in CASES:
        with Clock(et):
            h = head(as_of, tmp)
            today, prev = forecast._today_et(), forecast._prev_session(forecast._today_et())
            machine = dtmod.datetime.now().date()
        got = STALE in h
        ok = got == want and h.startswith(f"as of the close of {as_of[:4]}-{as_of[4:6]}-{as_of[6:]}")
        ok &= today == REAL_DT.fromisoformat(et).date()                    # the New York date, whatever the machine's
        print(f"{'ok  ' if ok else 'FAIL'} {et} ET (machine {machine}) close {as_of} last session {prev} "
              f"{'stale' if got else 'fresh'}: {why}")
        if not ok:
            bad.append((et, as_of, h))
    with Clock("2026-10-21 13:00"):              # the mock reproduces the 21 Oct bug with the pre-fix _today_et
        fixed, forecast._today_et = forecast._today_et, lambda: forecast.date.today()
        try:
            assert STALE in head("20261020", tmp), "the mocked clock does not reproduce the 21 Oct bug"
        finally:
            forecast._today_et = fixed
    assert not bad, bad
    print("ok")


if __name__ == "__main__":
    main()
