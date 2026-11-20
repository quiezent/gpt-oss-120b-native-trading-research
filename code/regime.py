"""The market-facts line of the Week 6 state, for sleeves with "regime_line": true in sleeves.json (candidate 2 and its
ML twin; Clayton, 14 Nov 2026: candidate 2 is "like W6-E v1 ... but with VIX"). One line of facts directly after the
state's "Rates:" line: VIX against the prior close, QQQ's open against the prior close and its move since. No advice, no
rule, no book figures.

The text is the Trading Model Researcher's: regime_line and insert_after_rates below are copied verbatim from its
runs/ml/research/week6_prep/c2/regime/regime_line_final.py (models-w3, 14 Nov 2026), because candidate 2's training rows
are rendered with that function byte for byte. Don't edit them here; regime_test.py compares the two files.

Three forms, by what the look can see (facts() picks one from the data, not from the clock alone):
  before the open (the 09:00 look takes no quotes):   Market: VIX closed Mon at 18.26. QQQ closed Mon at 679.35.
  the delayed QQQ last is still a pre-market print:    Market: VIX 19.10, +0.84 from Mon's close of 18.26. QQQ 673.00 before the open, ...
  the session's open is visible (10:00 to 15:30):      Market: VIX 19.46, ... QQQ opened at 672.10, ...; now 667.97, -0.61% since the open.
"""
from datetime import date, datetime
from datetime import time as dtime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
VIX = {"symbol": "VIX", "secType": "IND", "exchange": "CBOE"}     # the index, delayed, through the framework's own quote and bars
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


# ---- copied verbatim from the Researcher's regime_line_final.py: keep byte for byte ----
def _pct(a, b):
    return f"{100 * (a / b - 1):+.2f}%"


def regime_line(vix_last, vix_prev_close, qqq_prev_close, qqq_open, qqq_last, prev_day):
    if vix_last is None or qqq_last is None:
        return f"Market: VIX closed {prev_day} at {vix_prev_close:.2f}. QQQ closed {prev_day} at {qqq_prev_close:.2f}."
    v = f"VIX {vix_last:.2f}, {vix_last - vix_prev_close:+.2f} from {prev_day}'s close of {vix_prev_close:.2f}."
    if qqq_open is None:
        q = (f"QQQ {qqq_last:.2f} before the open, {_pct(qqq_last, qqq_prev_close)} from {prev_day}'s close of "
             f"{qqq_prev_close:.2f}.")
    else:
        q = (f"QQQ opened at {qqq_open:.2f}, {_pct(qqq_open, qqq_prev_close)} from {prev_day}'s close of "
             f"{qqq_prev_close:.2f}; now {qqq_last:.2f}, {_pct(qqq_last, qqq_open)} since the open.")
    return f"Market: {v} {q}"


def insert_after_rates(state_text, line):
    """Put the regime line directly after the state's "Rates:" line. Raises if there is not exactly one such line."""
    lines = state_text.split("\n")
    hits = [i for i, s in enumerate(lines) if s.startswith("Rates:")]
    if len(hits) != 1:
        raise ValueError(f"expected one Rates line, found {len(hits)}")
    return "\n".join(lines[:hits[0] + 1] + [line] + lines[hits[0] + 1:])


# ---- end of the copy ----


BARS_TIMEOUT, QUOTE_TIMEOUT = 10.0, 8.0     # seconds; the requests answer in about 1-3 s (14 Nov). Every sleeve's look waits
MAX_FAILS = 2                               # for them, so a request that failed twice today is not asked again today
_day = {}    # per ET date: {"vix_prev": (YYYYMMDD, close), "qqq_open": float, "fails": {request: count}}


def _today(today):
    if _day.get("date") != today:
        _day.clear()
        _day.update(date=today, fails={})
    return _day


def _ask(c, what, call):
    """call() unless this request already failed MAX_FAILS times today; a failure (an error or a timeout, the slow
    cases) is counted and raised. A quick answer that lacks the bar is not a failure and is asked again at the next look."""
    if c["fails"].get(what, 0) >= MAX_FAILS:
        raise RuntimeError(f"{what}: failed {MAX_FAILS} times today, not asked again")
    try:
        return call()
    except Exception:
        c["fails"][what] = c["fails"].get(what, 0) + 1
        raise


def vix_prev_close(gw, today, want=None):
    """(YYYYMMDD, close) of the last VIX daily bar before `today` (a date): the prior session's official close. The
    snapshot's "close" field is not used for it: after a session ends it still shows the session before (on Sat 14 Nov
    it read Thursday's 18.70 beside Friday's last of 19.30), and what it shows before the open is unverified.
    want: the session (YYYYMMDD) it must be of; another session's close is returned but not kept, so the next look asks
    again (a daily bar that arrives late is then picked up)."""
    c = _today(today)
    if "vix_prev" in c:
        return c["vix_prev"]
    day = today.strftime("%Y%m%d")
    rows = _ask(c, "VIX daily bars", lambda: gw.bars(VIX, duration="10 D", bar_size="1 day", what="TRADES", timeout=BARS_TIMEOUT))
    bars = sorted((str(b.get("date", ""))[:8], float(b["close"])) for b in rows if b.get("close") and str(b.get("date", ""))[:8] < day)
    if not bars:
        raise ValueError("no VIX daily bar before today")
    if want is None or bars[-1][0] == want:
        c["vix_prev"] = bars[-1]
    return bars[-1]


def qqq_open(gw, today):
    """Today's regular-hours open of QQQ: the open of the 09:30 five-minute bar, or None while the (delayed) bars don't
    have it yet."""
    c = _today(today)
    if "qqq_open" not in c:
        stamp = today.strftime("%Y%m%d") + " 09:30:00"
        rows = _ask(c, "QQQ 5-minute bars", lambda: gw.bars("QQQ", duration="1 D", bar_size="5 mins", what="TRADES", rth=True,
                                                             timeout=BARS_TIMEOUT))
        for b in rows:
            if str(b.get("date", "")).startswith(stamp) and b.get("open"):
                c["qqq_open"] = float(b["open"])
                break
    return c.get("qqq_open")


def _et(stamp):
    """A framework quote's last_time ("2026-11-14T00:59:59Z") as an aware ET datetime, or None."""
    try:
        return datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).astimezone(ET) if stamp else None
    except ValueError:
        return None


def _open_visible(now, qqq_quote):
    """True once the delayed QQQ last is a print of today's regular session. From the quote's own time when it has
    one; else from the clock, 16 minutes after the open (the data is delayed about 15)."""
    start = datetime.combine(now.date(), dtime(9, 30), tzinfo=ET)
    t = _et((qqq_quote or {}).get("last_time"))
    return t >= start if t else (now - start).total_seconds() >= 16 * 60


def facts(gw, now, session, qqq_prev_close, prev_date, qqq_quote=None, log=lambda m: None):
    """The line for this look, the same for every sleeve that carries it. now: the look's time (ET, aware); session:
    True from 09:30, when the look has delayed quotes; qqq_prev_close and prev_date ("YYYY-MM-DD"): the market table's
    close for QQQ and the session it is from; qqq_quote: the framework's quote row for QQQ, whose "last" is the price the
    market table shows and whose "last_time" says whether that print is of today's session.

    Raises when the prior closes can't be had or are from different sessions: the caller then leaves the line out. With
    the closes in hand, anything else that is missing, stale or fails gives the closes-only form, which is true at every
    look, and a log line."""
    today = now.date()
    vday, vprev = vix_prev_close(gw, today, prev_date.replace("-", ""))
    if vday != prev_date.replace("-", ""):
        raise ValueError(f"VIX's prior close is of {vday}, the table's QQQ close of {prev_date}")
    prev_day = DAYS[date.fromisoformat(prev_date).weekday()]
    closes = regime_line(None, vprev, qqq_prev_close, None, None, prev_day)
    if not session:
        return closes
    c = _today(today)
    qlast = (qqq_quote or {}).get("last")                      # the print itself: the table's mid or close fallback is not "now"
    try:
        q = _ask(c, "VIX snapshot", lambda: gw.quotes([VIX], timeout=QUOTE_TIMEOUT)[0])
    except Exception as e:                                     # noqa: BLE001
        log(f"regime line: closes only (VIX snapshot: {type(e).__name__}: {str(e)[:120]})")
        return closes
    vlast, vtime = (None if q.get("error") else q.get("last")), _et(q.get("last_time"))
    if q.get("close") and abs(q["close"] - vprev) > 0.005:
        log(f"regime line: the VIX snapshot's close {q['close']} is not the prior session's {vprev} ({vday}); the daily bar is used")
    if vlast and (vtime is None or vtime.date() != today):     # a value left from an earlier session is not today's VIX
        log(f"regime line: closes only (the VIX snapshot's last {vlast} is of {q.get('last_time')!r}, not of today)")
        return closes
    if not vlast or not qlast:
        log(f"regime line: closes only (VIX last {vlast!r}, QQQ last {qlast!r})")
        return closes
    if not _open_visible(now, qqq_quote):
        return regime_line(float(vlast), vprev, qqq_prev_close, None, float(qlast), prev_day)
    try:
        opened = qqq_open(gw, today)
    except Exception as e:                                     # noqa: BLE001
        log(f"regime line: closes only (QQQ's open: {type(e).__name__}: {str(e)[:120]})")
        return closes
    if not opened:
        log("regime line: closes only (QQQ's 09:30 bar is not in the bars yet)")
        return closes
    return regime_line(float(vlast), vprev, qqq_prev_close, opened, float(qlast), prev_day)


def shown(user_text):
    """The Market line of a look's user message, or None if it has none: the first line starting "Market: " before the
    model's own notes (which may hold such a line of their own). For the look record."""
    head = user_text.split(chr(10) + "Your notes from your last look", 1)[0]
    return next((l for l in head.splitlines() if l.startswith("Market: ")), None)
