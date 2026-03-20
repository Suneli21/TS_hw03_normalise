import numpy as np

def smape(y_true, y_pred):
  return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def mase(y_true, y_pred, y_train):
  naive = np.mean(np.abs(np.diff(y_train)))
  return np.mean(np.abs(y_true - y_pred)) / naive

def rmse(y_true, y_pred):
  return np.sqrt(np.mean((y_true - y_pred)**2))