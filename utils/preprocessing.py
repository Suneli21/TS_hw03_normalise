# здесь лежат все дополнительные функции по подготовке данных
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import pandas as pd


def add_ds(df):
  """Добавление колонки с timestamp"""
  df = df.copy()
  df['ds'] = df.groupby('unique_id').cumcount()
  return df

def train_test_split(df, horizon=18):
  """Разделение на трейн и тест"""
  df = add_ds(df)
  df = df.drop(columns='t')
  train = []
  test = []
  for uid, group in df.groupby('unique_id'):
      train.append(group.iloc[:-horizon])
      test.append(group.iloc[-horizon:])

  return pd.concat(train).reset_index(drop=True), pd.concat(test).reset_index(drop=True)

def apply_scaler_series(train, test, scaler):
  """Нормализация данных"""
  if scaler is None:
    return train.copy(), test.copy(), None

  sc = scaler.__class__()
  train_scaled = sc.fit_transform(train.reshape(-1,1)).flatten()
  test_scaled = sc.transform(test.reshape(-1,1)).flatten()
  return train_scaled, test_scaled, sc


def apply_boxcox(train, test):
  """Применение Box-Cox преобразования к ряду"""
  shift = 0
  if (train <= 0).any():
      shift = abs(train.min()) + 1

  train_shifted = train + shift
  train_bc, lam = boxcox(train_shifted)

  return train_bc, lam, shift


def invert_boxcox(preds, lam, shift):
  """Применение обратного Box-Cox преобразования к ряду"""
  return inv_boxcox(preds, lam) - shift


def apply_scaling_train(train, test, scaler):
  """Применение нормализации"""
  if scaler is None:
      return train.copy(), test.copy(), {}

  train_scaled = []
  test_scaled = []
  scalers_dict = {}

  for uid in train['unique_id'].unique():
    tr = train[train['unique_id'] == uid].copy()
    te = test[test['unique_id'] == uid].copy()

    sc = scaler.__class__()

    tr['y'] = sc.fit_transform(tr[['y']])
    te['y'] = sc.transform(te[['y']])

    train_scaled.append(tr)
    test_scaled.append(te)
    scalers_dict[uid] = sc

  return pd.concat(train_scaled), pd.concat(test_scaled), scalers_dict

def inverse_scaling(forecast, scalers_dict):
  """"Обратное преобразование к нормализованным прогнозам"""
  df_inv = forecast.copy()
  for uid, sc in scalers_dict.items():
    mask = df_inv['unique_id'] == uid
    vals = df_inv.loc[mask, ['PatchTST']].values
    temp_df = pd.DataFrame(vals, columns=['y'])
    inv = sc.inverse_transform(temp_df)
    df_inv.loc[mask, 'PatchTST'] = inv.flatten()

  return df_inv

def create_lag_features(df, lags=[1,2,3,6,12]):
  """Лаговые признаки для CatBoost"""
  df = df.copy()
  for lag in lags:
    df[f'lag_{lag}'] = df.groupby('unique_id')['y'].shift(lag)
  return df