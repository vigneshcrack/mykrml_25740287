"""Daily cryptocurrency market data from the Kraken public OHLC API (AT3).

Kraken's ``/0/public/OHLC`` endpoint is public (no API key) and returns, for
``interval=1440``, up to the 720 most recent daily candles. The final candle in
every response is the current, still-open day, so it is dropped by default:
features computed from a half-finished day would differ from the training data.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import pandas as pd
import requests

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
DAILY_INTERVAL_MINUTES = 1440

SUPPORTED_TOKENS: tuple[str, ...] = ("BTC", "ETH", "XRP", "SOL", "DOGE", "XLM", "TRX", "XMR")

# Kraken uses XBT for Bitcoin and XDG for Dogecoin (pairs verified online 20 Oct 2026).
KRAKEN_PAIRS: dict[str, str] = {
    "BTC": "XBTUSD",
    "ETH": "ETHUSD",
    "XRP": "XRPUSD",
    "SOL": "SOLUSD",
    "DOGE": "XDGUSD",
    "XLM": "XLMUSD",
    "TRX": "TRXUSD",
    "XMR": "XMRUSD",
}

OHLC_COLUMNS = ["time", "open", "high", "low", "close", "vwap", "volume", "trades"]

# Kraken error prefixes worth retrying (rate limiting / temporary unavailability).
_RETRYABLE_KRAKEN_ERRORS = ("EAPI:Rate limit", "EGeneral:Too many requests", "EService:")


class UnsupportedTokenError(ValueError):
    """Raised when a token is not one of ``SUPPORTED_TOKENS``."""


class MarketDataError(RuntimeError):
    """Raised when market data cannot be retrieved or is invalid."""


class _RetryableError(Exception):
    """Internal: a temporary failure that is worth retrying."""


def normalise_token(token: str) -> str:
    """Return the upper-case token symbol, or raise ``UnsupportedTokenError``.

    Parameters
    ----------
    token : str
        Token symbol such as ``"btc"`` or ``"BTC"``.

    Returns
    -------
    str
        Normalised symbol, e.g. ``"BTC"``.
    """
    symbol = str(token).strip().upper()
    if symbol not in SUPPORTED_TOKENS:
        raise UnsupportedTokenError(
            f"Unsupported cryptocurrency {token!r}; supported tokens: {list(SUPPORTED_TOKENS)}"
        )
    return symbol


def parse_kraken_ohlc(payload: dict, drop_incomplete: bool = True) -> pd.DataFrame:
    """Convert a Kraken OHLC JSON payload into a tidy daily DataFrame.

    Parameters
    ----------
    payload : dict
        Decoded JSON returned by the Kraken OHLC endpoint.
    drop_incomplete : bool, default True
        Drop the final row, which Kraken documents as the current, not yet
        committed candle.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``date`` (timezone-naive UTC dates, ascending) with float
        columns ``open, high, low, close, vwap, volume`` and int ``trades``.
    """
    errors = payload.get("error") or []
    if errors:
        raise MarketDataError(f"Kraken API error: {errors}")

    result = payload.get("result") or {}
    pair_keys = [key for key in result if key != "last"]
    if not pair_keys:
        raise MarketDataError("Kraken response contained no OHLC data")

    rows = result[pair_keys[0]]
    df = pd.DataFrame(rows, columns=OHLC_COLUMNS)
    if drop_incomplete and len(df) > 0:
        df = df.iloc[:-1]

    for column in ["open", "high", "low", "close", "vwap", "volume"]:
        df[column] = df[column].astype(float)
    df["trades"] = df["trades"].astype(int)
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None).dt.normalize()

    df = df.drop(columns="time").set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_kraken_daily_ohlcv(
    token: str,
    session: requests.Session | None = None,
    timeout: float = 15.0,
    max_retries: int = 2,
    backoff_seconds: float = 1.0,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Fetch daily OHLCV candles for one token from Kraken.

    Retries on network errors, HTTP 429/5xx and Kraken rate-limit errors with
    exponential back-off (``backoff_seconds * 2**attempt``).

    Parameters
    ----------
    token : str
        One of ``SUPPORTED_TOKENS``.
    session : requests.Session, optional
        Re-usable HTTP session (useful for connection pooling and tests).
    timeout : float, default 15.0
        Per-request timeout in seconds.
    max_retries : int, default 2
        Number of retries after the first attempt.
    backoff_seconds : float, default 1.0
        Base delay for exponential back-off.
    drop_incomplete : bool, default True
        Drop the current, still-open daily candle.

    Returns
    -------
    pandas.DataFrame
        Daily candles indexed by date (see ``parse_kraken_ohlc``).

    Raises
    ------
    UnsupportedTokenError
        If ``token`` is not supported.
    MarketDataError
        If data cannot be retrieved after all retries.
    """
    symbol = normalise_token(token)
    params = {"pair": KRAKEN_PAIRS[symbol], "interval": DAILY_INTERVAL_MINUTES}
    http = session or requests.Session()
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = http.get(KRAKEN_OHLC_URL, params=params, timeout=timeout)
            if response.status_code == 429 or response.status_code >= 500:
                raise _RetryableError(f"Kraken HTTP {response.status_code}")
            response.raise_for_status()
            payload = response.json()
            errors = payload.get("error") or []
            if any(str(e).startswith(_RETRYABLE_KRAKEN_ERRORS) for e in errors):
                raise _RetryableError(f"Kraken temporary error: {errors}")
            return parse_kraken_ohlc(payload, drop_incomplete=drop_incomplete)
        except (requests.RequestException, _RetryableError) as exc:
            # temporary problem: wait and try again (unless out of retries)
            last_error = exc
            if attempt < max_retries:
                time.sleep(backoff_seconds * (2**attempt))

    raise MarketDataError(f"Could not retrieve {symbol} data from Kraken: {last_error}")


def fetch_many_daily_ohlcv(tokens: Iterable[str] = SUPPORTED_TOKENS, **kwargs) -> pd.DataFrame:
    """Fetch daily candles for several tokens into one long (panel) DataFrame.

    Parameters
    ----------
    tokens : iterable of str, default all ``SUPPORTED_TOKENS``
    **kwargs
        Passed to ``fetch_kraken_daily_ohlcv``.

    Returns
    -------
    pandas.DataFrame
        Columns ``date, token, open, high, low, close, vwap, volume, trades``,
        sorted by token then date.
    """
    frames = []
    for token in tokens:
        df = fetch_kraken_daily_ohlcv(token, **kwargs).reset_index()
        df.insert(1, "token", normalise_token(token))
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values(["token", "date"]).reset_index(drop=True)
