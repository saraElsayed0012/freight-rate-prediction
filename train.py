"""Freight rate model. Usage: python train.py
Reads train-test.csv, validation.csv, december-chart-inputs.csv (next to this file or in data/).
Writes validation_predictions.csv and december_predictions.csv."""
import os
import numpy as np, pandas as pd, lightgbm as lgb


def find(*names):
    for d in ('', ''):
        for n in names:
            if os.path.exists(d + n):
                return d + n
    raise FileNotFoundError(f'Could not find any of {names}. Put the CSV files next to train.py or in data/.')


EQ = {'Dry Van': 0, 'Reefer': 1, 'Flatbed': 2}


def feat(d):
    X = pd.DataFrame(index=d.index)
    X['dist'] = d.distance
    X['logd'] = np.log(d.distance)
    X['w'] = d.weight.abs()                      # 292 train rows have sign-flipped weights; NaNs left to LightGBM
    X['eq'] = d.equipment.map(EQ)
    for k in ['pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon']:
        X[k] = d[k]
    X['dow'] = d.date.dt.dayofweek               # weekday: the only date effect that repeats into December
    return X                                     # market_index / quote_signal deliberately left out (unstable)


def model():
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=60,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             objective='huber', alpha=0.5, verbose=-1)   # Huber: robust to the $12-14/mile outliers


def fit(tr):
    return model().fit(feat(tr), tr.rpm)          # predict rate per mile


def predict(m, d):
    return m.predict(feat(d)) * d.distance        # back to dollars


if __name__ == '__main__':
    t = pd.read_csv(find('train-test.csv', 'train_test.csv'), parse_dates=['date'])
    v = pd.read_csv(find('validation.csv'), parse_dates=['date'])
    c = pd.read_csv(find('december-chart-inputs.csv', 'december_chart_inputs.csv'), parse_dates=['date'])
    t['rpm'] = t.posted_rate / t.distance

    # Validation design: validation.csv is the future (Nov-Dec) vs train (Jan-Oct), so hold out the LAST two
    # months (Sep-Oct) of train and fit on Jan-Aug. No random split: it would leak market conditions.
    a, b = t[t.date < '2025-09-01'], t[t.date >= '2025-09-01']
    err = (predict(fit(a), b) - b.posted_rate).abs()
    print(f'time holdout (Sep-Oct)  MAE ${err.mean():.1f}  MAPE {(err / b.posted_rate).mean() * 100:.2f}%')

    m = fit(t)                                    # final model on all labeled data
    v['predicted_rate'] = predict(m, v).round(2)
    v[['load_id', 'predicted_rate']].to_csv('validation_predictions.csv', index=False)

    city = {}                                     # coordinates for the chart lane, looked up by city name
    for d in (t, v):
        for p, la, lo in zip(d.pickup, d.pickup_lat, d.pickup_lon): city[p] = (la, lo)
        for p, la, lo in zip(d.delivery, d.delivery_lat, d.delivery_lon): city[p] = (la, lo)
    c['pickup_lat'], c['pickup_lon'] = zip(*c.pickup.map(city))
    c['delivery_lat'], c['delivery_lon'] = zip(*c.delivery.map(city))
    c['predicted_rate'] = predict(m, c).round(2)
    c[['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']].to_csv(
        'december_predictions.csv', index=False)
    print(c.groupby(c.date.dt.day_name()).predicted_rate.mean().round(1).to_string())