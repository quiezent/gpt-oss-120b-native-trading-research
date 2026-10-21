"""Minimal FRED/ALFRED client. The key comes from ml_research.keys and is never printed, logged or written.
Errors are re-raised with the key redacted."""
import time
import requests
from ml_research import keys

BASE = "https://api.stlouisfed.org/fred/"
_last = [0.0]
_s = requests.Session()


def get(path, **params):
    key = keys.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY not set")
    params = dict(params, api_key=key, file_type="json")
    for attempt in range(4):
        wait = 0.55 - (time.time() - _last[0])  # FRED allows 120 req/min; stay well under
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = _s.get(BASE + path, params=params, timeout=30)
        except requests.RequestException as e:
            err = str(e).replace(key, "REDACTED")
            if attempt == 3:
                raise RuntimeError("FRED request failed: " + err) from None
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code != 200:
            raise RuntimeError(f"FRED {path} HTTP {r.status_code}: {r.text[:300].replace(key, 'REDACTED')}")
        return r.json()
    raise RuntimeError(f"FRED {path}: retries exhausted")
