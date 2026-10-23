from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

from mykrml_25740287.data.crypto import (
    KRAKEN_PAIRS,
    MarketDataError,
    UnsupportedTokenError,
    fetch_kraken_daily_ohlcv,
    normalise_token,
    parse_kraken_ohlc,
)


def _payload(n=3, key="XXBTZUSD"):
    day = 86_400
    start = 1_760_000_000 - (1_760_000_000 % day)
    rows = [
        [start + i * day, "100.0", "110.0", "90.0", f"{105 + i}.0", "101.0", "12.5", 42]
        for i in range(n)
    ]
    return {"error": [], "result": {key: rows, "last": start + (n - 1) * day}}


def _response(payload, status=200):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def test_normalise_token_accepts_lower_case():
    assert normalise_token(" btc ") == "BTC"


def test_unsupported_token_raises():
    with pytest.raises(UnsupportedTokenError):
        normalise_token("ADA")


def test_parse_drops_incomplete_last_candle_and_casts_types():
    df = parse_kraken_ohlc(_payload(n=3))
    assert len(df) == 2  # the 3rd (current, still-open) candle is dropped
    assert df.index.name == "date"
    assert df["close"].tolist() == [105.0, 106.0]
    assert df["volume"].dtype == float
    assert df.index.is_monotonic_increasing


def test_parse_can_keep_incomplete_candle():
    assert len(parse_kraken_ohlc(_payload(n=3), drop_incomplete=False)) == 3


def test_parse_raises_on_kraken_error():
    with pytest.raises(MarketDataError):
        parse_kraken_ohlc({"error": ["EQuery:Unknown asset pair"], "result": {}})


def test_fetch_uses_kraken_pair_and_daily_interval():
    session = MagicMock()
    session.get.return_value = _response(_payload(key="XDGUSD"))
    df = fetch_kraken_daily_ohlcv("DOGE", session=session)
    params = session.get.call_args.kwargs["params"]
    assert params == {"pair": KRAKEN_PAIRS["DOGE"], "interval": 1440}
    assert isinstance(df, pd.DataFrame) and len(df) == 2


def test_fetch_retries_on_rate_limit_then_succeeds():
    session = MagicMock()
    session.get.side_effect = [_response({}, status=429), _response(_payload())]
    df = fetch_kraken_daily_ohlcv("BTC", session=session, backoff_seconds=0)
    assert session.get.call_count == 2
    assert len(df) == 2


def test_fetch_raises_after_all_retries_fail():
    session = MagicMock()
    session.get.side_effect = requests.ConnectionError("network down")
    with pytest.raises(MarketDataError):
        fetch_kraken_daily_ohlcv("BTC", session=session, max_retries=2, backoff_seconds=0)
    assert session.get.call_count == 3
