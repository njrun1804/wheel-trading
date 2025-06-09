# Databento Unity Integration Guide

## 0. Prerequisites

```bash
pip install --upgrade databento pandas duckdb
export DATABENTO_API_KEY="db-…"
```

All code below imports the key from that env var.
If you're running inside Docker/K8s, inject it as a secret at runtime.

---

## 1. Which Databento datasets matter for a Unity wheel bot?

| Purpose                                    | Dataset                     | Typical schema(s)               | Why this one                                                                                               |
| ------------------------------------------ | --------------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Unity option definitions + NBBO/trades** | `OPRA.PILLAR`               | `definition`, `mbp-1`, `trades` | Consolidated US-options feed (covers all 17 venues) and carries the OCC contract IDs. ([databento.com][1]) |
| **Unity top-of-book equity quotes**        | `EQUUS.MINI`                | `mbp-1`                         | Composite "mini NBBO" for >7,000 US stocks—no exchange licence fees. ([databento.com][2])                  |
| **Venue-level equity depth** *(optional)*  | e.g. `NYSE_AMER.INTEGRATED` | `mbp-10`, `trades`              | Only if you need per-venue depth; Unity lists on NYSE American.                                            |

Rate limits: 100 concurrent streams **and** 100 req/s per IP for Historical and Reference APIs. ([databento.com][3])

---

## 2. Symbology that actually works

* **All Unity options (parent):** `U.OPT`   *(stype = PARENT)* ([databento.com][4])
* **One specific contract (raw/OSI):** `U250613P00022500` (put, $22.50, exp 2025-06-13).
* **Underlying equity (raw):** `U` in equities datasets, or `U.EQ` if you prefer parent.

The helper snippets below convert between them automatically.

---

## 3. One-shot "hello world" pull (Unity options + spot)

```python
from datetime import datetime, timedelta, timezone
import databento as db
from databento import Schema, SType
import pandas as pd

API  = db.Historical()                 # picks up env var
UTC  = timezone.utc
NOW  = datetime.now(UTC)

# --- 1. Resolve Unity's entire option chain  ----------------------
defs = API.timeseries.get_range(
    dataset   = "OPRA.PILLAR",
    schema    = Schema.DEFINITION,
    start     = "2025-06-01",
    end       = "2025-06-14",
    stype_in  = SType.PARENT,
    symbols   = ["U.OPT"],
)
chain = defs.to_df()

# --- 2. Filter to ~30-40 DTE & ±15 % moneyness ---------------------
spot  = 24.9   # real-time spot comes in §4
sel   = chain.assign(
            dte = (chain["expiration"] - NOW).dt.days,
            mny = (chain["strike"] / spot) - 1
       ).query("25 <= dte <= 60 and abs(mny) <= 0.15")
raw_syms = sel["raw_symbol"].unique().tolist()

# --- 3. Pull top-of-book (NBBO) for those symbols ------------------
qts = API.timeseries.get_range(
    dataset  = "OPRA.PILLAR",
    schema   = "mbp-1",
    start    = NOW - timedelta(days=2),
    end      = NOW,
    stype_in = SType.RAW_SYMBOL,
    symbols  = raw_syms,
).to_df()

# --- 4. Pull Unity equity NBBO (cheap) -----------------------------
u_bbo = API.timeseries.get_range(
    dataset  = "EQUUS.MINI",
    schema   = "mbp-1",
    start    = NOW - timedelta(days=2),
    end      = NOW,
    stype_in = SType.RAW_SYMBOL,
    symbols  = ["U"],
).to_df()

# --- 5. Inspect / persist -----------------------------------------
print(chain.head(3))
print(qts.groupby("raw_symbol")["ask_px"].head())
u_bbo.to_parquet("unity_nbbo.parquet")
```

---

## 4. Turn it into a reusable loader (drop-in for `wheel_trading/`)

```python
# wheel_trading/utils/databento_unity.py
from __future__ import annotations
from functools import lru_cache
from datetime import datetime, timedelta, timezone
import databento as db
from databento import Schema, SType
import pandas as pd

API = db.Historical()            # env var handles auth
UTC = timezone.utc

@lru_cache
def chain(start: str, end: str) -> pd.DataFrame:
    """Return option definitions for Unity between two ISO dates."""
    return API.timeseries.get_range(
        dataset  = "OPRA.PILLAR",
        schema   = Schema.DEFINITION,
        stype_in = SType.PARENT,
        symbols  = ["U.OPT"],
        start    = start,
        end      = end,
    ).to_df()

def quotes(raw_symbols: list[str], start: datetime, end: datetime,
           schema: str = "mbp-1") -> pd.DataFrame:
    return API.timeseries.get_range(
        dataset  = "OPRA.PILLAR",
        schema   = schema,
        stype_in = SType.RAW_SYMBOL,
        symbols  = raw_symbols,
        start    = start,
        end      = end,
    ).to_df()

def spot(days_back: int = 2) -> pd.DataFrame:
    now = datetime.now(UTC)
    return API.timeseries.get_range(
        dataset  = "EQUUS.MINI",
        schema   = "mbp-1",
        stype_in = SType.RAW_SYMBOL,
        symbols  = ["U"],
        start    = now - timedelta(days=days_back),
        end      = now,
    ).to_df()
```

### How wheel-logic can call it

```python
from utils.databento_unity import chain, quotes, spot
from datetime import datetime, timezone

defs = chain("2025-06-01", "2025-06-14")

# pick strikes ±10 % & 30-45 DTE
now  = datetime.now(timezone.utc)
spot_px = spot(1)["ask_px"].iloc[-1]
sel = defs.query(
    "25 <= (expiration - @now).dt.days <= 45 and abs(strike / @spot_px - 1) <= 0.10"
)
qbbo = quotes(sel["raw_symbol"].tolist(),
              start=now - pd.Timedelta(days=1),
              end=now)
#  feed qbbo into premium/delta scoring
```

---

## 5. Cost-control & rate-limit hygiene

```python
est = API.metadata.get_cost(
    dataset="OPRA.PILLAR",
    schema="mbp-1",
    symbols=["U.OPT"],
    start="2025-06-01",
    end="2025-06-30",
)
print(f"Bytes: {est.bytes:,}  Cost: ${est.cost:,.2f}")
```

*Historical pulls are metered by **uncompressed bytes**; MBP-1 for U's option chain across 30 days ≈ 30–50 MB (≈ $0.60–$1).
*Respect 100 req/s & 100 streams/IP; if you parallelise, use `asyncio` and an `asyncio.Semaphore(90)` guard. ([databento.com][3])

---

## 6. Common gotchas

| Symptom                                  | Fix                                                                                  |
| ---------------------------------------- | ------------------------------------------------------------------------------------ |
| `Could not resolve smart symbols: U.OPT` | You passed `stype_in = RAW_SYMBOL`; use `PARENT` when you send `U.OPT`.              |
| `data_end_after_available_end`           | End date is T+0; OPRA historical is *T+1* after US close. Pull through yesterday.    |
| Empty dataframe for equity BBO           | Unity is NYSE American—if you're pulling `NASDAQ.INTEGRATED` you'll get nothing.     |
| Big invoices                             | Call `metadata.get_cost` first, or narrow to DTE & moneyness before you hit `mbp-1`. |

---

## 7. Delete / refactor checklist

* ❌ `mock_option_chain*.py`, `mock_option_quote*.json` fixtures
* ❌ Yahoo Finance fallbacks (replace with `EQUUS.MINI`)
* ✅ Rename every lingering `DatentoClient` → `DatabentoClient`.

Once those are gone, the tests that were failing to "load" should pass without any monkey-patching or mock data.

---

### 🔑 TL;DR for busy reviewers

* **Unity options are 100 % present in Databento** via `OPRA.PILLAR` → `U.OPT`.
* Underlying spot: use `EQUUS.MINI` (`U`).
* Filter chain definitions in Python, then only pull NBBO for the ~20–40 contracts your wheel bot cares about.
* Budget: ≈ $1 per month of MBP-1 per underlier if you pre-filter wisely.

Ship it.

[1]: https://databento.com/docs/venues-and-datasets/opra-pillar?utm_source=chatgpt.com "OPRA Pillar Binary: Data feed specifications - Databento"
[2]: https://databento.com/docs/venues-and-datasets/equs-mini?utm_source=chatgpt.com "Databento US Equities Mini"
[3]: https://databento.com/docs/api-reference-reference?utm_source=chatgpt.com "Databento API documentation - Reference"
[4]: https://databento.com/docs/standards-and-conventions/symbology?utm_source=chatgpt.com "Symbology | Databento standards & conventions"
