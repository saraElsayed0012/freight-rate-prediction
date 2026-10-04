# Freight Rate Prediction

Predicts the posted rate (USD) for freight loads. Final predictions for the 12,000 validation loads are in
`validation_predictions.csv`; the fixed December 2025 predictions are in `december_predictions.csv`.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# put train-test.csv, validation.csv, december-chart-inputs.csv next to train.py (or in data/)
python train.py
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

`train.py` prints the holdout error, then writes `validation_predictions.csv` and `december_predictions.csv`.
`score.py` validates both files and writes `scorer_results/candidate_december.png`.
Tested with Python 3.12.

## Files

| File | Purpose |
|---|---|
| `train.py` | Full pipeline: features, time-based validation, final model, predictions |
| `code.ipynb` | Step-by-step exploration (data quality, feature checks, experiments) |
| `score.py` | Provided scorer: validates files and draws the December chart |
| `validation_predictions.csv` | Final predictions (`load_id,predicted_rate`) |
| `december_predictions.csv` | Completed December chart inputs |
| `scorer_results/candidate_december.png` | December chart |
| `requirements.txt` | Dependencies |

## Approach

**Model.** One LightGBM regressor predicts *rate per mile*, which is multiplied by distance to give dollars
(rate per mile is much more stable than the dollar amount). Huber loss limits the effect of a few extreme
labels ($12-14 per mile).

**Features.** `distance`, `log(distance)`, `|weight|`, equipment type, pickup/delivery coordinates, day of week.

**Validation.** `validation.csv` covers Nov-Dec 2025, after all labeled data (Jan-Oct). A random split would
leak market conditions, so the split is by time: train on Jan-Aug, test on Sep-Oct. Holdout error:
**MAE about $97.7 (MAPE 4.3%)**. The final model is then trained on all 48,000 labeled rows.

## Data-quality findings

- 292 training rows have negative weights (sign errors); absolute value is used.
- 300 missing weights and 374 missing `market_index` values; LightGBM handles missing values natively.
- A few rates of $12-14 per mile look like bad labels; handled with Huber loss.
- `quote_signal` follows the true rate in some months, runs backwards in others (Apr, May, Jul, Oct) and is
  noise in August. In the validation months it has no relationship with distance. It is excluded.
- `market_index` tracks rates Jan-Aug but fails in Sep-Oct (it raised holdout error from about $98 to $170).
  It is excluded, and it is not available for the December chart anyway.

## What was tried

| Experiment | Holdout MAE |
|---|---|
| distance + equipment | $132 |
| + weight | $115 |
| + coordinates | $100 |
| + weekday | **$97.7** |
| + `market_index` | $169 (worse) |
| equipment x distance, geographic zones | no gain |
| lane-average rate per mile | worse |
| CatBoost / LightGBM blend | no consistent gain across time folds |

## December chart

The chart input is one fixed load (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb) for every day
of December, so only the date changes. The only date effect that repeats into December is the weekday
(mid-week loads cost about 2% more than Sunday/Monday), which gives the weekly pattern in the chart.
# freight-rate-prediction
