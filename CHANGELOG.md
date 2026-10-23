# Changelog

<!--next-version-placeholder-->

## v0.3.0 (20/10/2026) – AT3

- **New** `data.crypto`: Kraken public OHLC client (`fetch_kraken_daily_ohlcv`,
  `fetch_many_daily_ohlcv`, `parse_kraken_ohlc`) with token validation, pair mapping,
  retries with exponential back-off, and removal of the incomplete current-day candle.
- **New** `features.crypto`: the 13 AT3 contract features (`compute_features`,
  `compute_features_panel`), the four targets (Asset Performance Index, Market Risk Index
  and components, Trend Category, Trading Opportunity Rating) and `forward_target`
  (token-aware t+7 shift).
- **New** `models.validation.chronological_split_with_gap`: purged chronological split for
  panel time series.
- **New** `models.performance.skill_vs_baseline`; `regression_scores` now reports R²;
  `multiclass_classification_scores` now reports macro-F1 and balanced accuracy.
- **New** `data.sets.load_sets`; `save_sets` now also accepts NumPy arrays.
- **Fixed** dependency ranges to match the AT3 required stack (pandas 2.2.x,
  scikit-learn 1.6.x, Python 3.12.13); removed the broken `mykrml-25740287` console script.
- 28 new unit tests (48 in total), including boundary tests for the class labels and
  hand-calculated index values.

## v0.2.0 (09/2026) – AT2

- Open-Meteo weather fetching and daily aggregation; Climate Comfort Index, Weather Hazard
  Index/Category and `shift_target`; regression and multiclass metrics; chronological split.

## v0.1.0 (23/08/2026) – AT1

- First release of `mykrml_25740287`!
