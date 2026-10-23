# mykrml_25740287

Reusable machine learning utilities developed by Vignesh Selvam (25740287) for UTS 36120
Advanced Machine Learning Application: data sourcing, feature engineering, target
construction, leakage-aware validation and performance metrics.

## Installation

The package is published on TestPyPI:

```bash
uv add mykrml-25740287==0.3.0 --index https://test.pypi.org/simple/ --index-strategy unsafe-best-match
# or
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mykrml-25740287==0.3.0
```

## Usage (AT3 – cryptocurrency forecasting)

```python
from mykrml_25740287.data.crypto import fetch_kraken_daily_ohlcv
from mykrml_25740287.features.crypto import (
    compute_features, compute_market_risk_index, forward_target,
)

candles = fetch_kraken_daily_ohlcv("BTC")          # daily OHLCV, incomplete day dropped
features = compute_features(candles)               # 13 contract features
mri_today = compute_market_risk_index(features, candles["close"])
mri_in_7_days = forward_target(mri_today, horizon=7)   # training target
```

| Module | Contents |
|---|---|
| `data.crypto` | Kraken OHLC client, supported tokens and pair mapping (AT3) |
| `data.weather` | Open-Meteo historical weather fetch and daily aggregation (AT2) |
| `data.sets` | `pop_target`, `save_sets`, `load_sets`, competition loader (AT1) |
| `features.crypto` | AT3 features, Market Risk Index and the other three targets |
| `features.targets` | AT2 Climate Comfort and Weather Hazard targets |
| `features.basketball` | AT1 basketball features |
| `models.performance` | Regression, binary and multiclass metrics; skill vs baseline |
| `models.validation` | Grouped, chronological and purged chronological splits |

## Development

```bash
uv sync
uv run pytest        # 48 tests
uv build
uv publish --index testpypi --token <TESTPYPI_TOKEN>
```

## License

MIT. Created with the `py-pkgs-cookiecutter` template.
