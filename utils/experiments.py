from utils.metrics import rmse, smape, mase
from utils.preprocessing import *
import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from catboost import CatBoostRegressor

def process_series(series_df, h=18, scaler=None, use_boxcox=False):
  """"Функция для обучения и инференса статистических моделей"""
  series_df = series_df.copy()
  series_df = series_df.sort_values('ds')
  train_df = series_df.iloc[:-h]
  test_df = series_df.iloc[-h:]
  train_values = train_df['y'].values
  test_values = test_df['y'].values
  train_scaled, test_scaled, sc = apply_scaler_series(train_values, test_values, scaler)

  if use_boxcox:
    train_used, lam, shift = apply_boxcox(train_scaled, test_scaled)
  else:
    train_used = train_scaled
    lam, shift = None, None


  train_sf = pd.DataFrame({
      'unique_id': 'series',
      'ds': train_df['ds'],
      'y': train_used
  })

  models = [
      AutoARIMA(),
      AutoETS(),
      AutoTheta(),
  ]

  sf = StatsForecast(models=models, freq=12, n_jobs=1)
  sf.fit(train_sf)

  forecast = sf.predict(h=h).reset_index()
  if use_boxcox:
    for col in forecast.columns:
      if col not in ['unique_id', 'ds']:
        forecast[col] = invert_boxcox(forecast[col].values, lam, shift)

  if sc is not None:
    for col in forecast.columns:
      if col not in ['unique_id', 'ds']:
        vals = forecast[col].values.reshape(-1,1)
        forecast[col] = sc.inverse_transform(vals).flatten()
  results = []

  for model in ['AutoARIMA', 'AutoETS', 'AutoTheta']:
    y_pred = forecast[model].values

    results.append({
        'model': model,
        'sMAPE': smape(test_values, y_pred),
        'RMSE': rmse(test_values, y_pred),
        'MASE': mase(test_values, y_pred, train_values),

    })

  return results, forecast, train_df, test_df

def run_patchtst(train_df, h=18):
  """Эксперимент для PatchTST"""
  model = PatchTST(
      h=h,
      input_size=36,
      max_steps=200,
      scaler_type=None
  )

  nf = NeuralForecast(models=[model], freq=1)
  nf.fit(df=train_df)
  forecast = nf.predict()
  return forecast


def run_catboost_single(train_df, test_df, lags=[1,2,3,6,12]):
  """Эксперимент для CatBoost"""
  df_all = pd.concat([train_df, test_df]).copy()
  df_all = create_lag_features(df_all, lags)
  train_feat = df_all.iloc[:len(train_df)].dropna()
  X_train = train_feat[[f'lag_{l}' for l in lags]]
  y_train = train_feat['y']
  model = CatBoostRegressor(verbose=0)
  model.fit(X_train, y_train)
  history = train_df['y'].values.tolist()
  preds = []
  for _ in range(len(test_df)):
    row = [history[-l] for l in lags]
    pred = model.predict([row])[0]
    preds.append(pred)
    history.append(pred)
  return np.array(preds)