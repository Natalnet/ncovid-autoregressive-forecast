# grid search simple forecasts
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from numpy import array
from ast import literal_eval
import datetime

# split a univariate dataset into train/test sets
def split_dataset(data, n_test, n_days):
  # makes dataset multiple of 7
  data = data[len(data) % n_days:]
  # make test set multiple of 7
  n_test -= n_test % n_days
  # split into standard weeks
  train, test = data[:-n_test], data[-n_test:]
  # restructure into windows of weekly data
  train = array(split(train, len(train)/n_days))
  test = array(split(test, len(test)/n_days))
  return train, test

# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
  scores = list()
  # calculate an RMSE score for each day
  for i in range(actual.shape[1]):
    # calculate mse
    mse = mean_squared_error(actual[:, i], predicted[:, i])
    # calculate rmse
    rmse = sqrt(mse)
    # store
    scores.append(rmse)
  # calculate overall RMSE
  s = 0
  for row in range(actual.shape[0]):
    for col in range(actual.shape[1]):
      s += (actual[row, col] - predicted[row, col])**2
  score = sqrt(s / (actual.shape[0] * actual.shape[1]))
  return score, scores

# summarize scores
def summarize_scores(name, score, scores):
  s_scores = ',' .join(['%.1f' % s for s in scores])
  print('%s: [%.3f] %s' % (name, score, s_scores))

# evaluate a single model
def evaluate_model(data, n_test, n_days, model_func, cfg, forecast=False):
  # walk-forward validation over each week
  predictions = list()
  # split dataset
  train, test = split_dataset(data, n_test, n_days)
  # history is a list of weekly data
  history = [x for x in train]
  for i in range(len(test)):
    # predict the week
    yhat_sequence = model_func(history, cfg)
    # store the predictions
    predictions.append(yhat_sequence)
    # get real observation and add to history for predicting the next week
    if forecast:
      history.append(yhat_sequence)
    else:
      history.append(test[i, :])
  predictions = array(predictions)
  # evaluate predictions days for each week
  score, scores = evaluate_forecasts(test[:, :], predictions)
  return score

# evaluate a single model
def evaluate_model_prediction(data, n_test, n_days, model_func, cfg, forecast=False):
  # walk-forward validation over each week
  predictions = list()
  # split dataset
  train, test = split_dataset(data, n_test, n_days)
  # history is a list of weekly data
  history = [x for x in train]
  for i in range(len(test)):
    # predict the week
    yhat_sequence = model_func(history, cfg)
    # store the predictions
    predictions.append(yhat_sequence)
    # get real observation and add to history for predicting the next week
    if forecast:
      history.append(yhat_sequence)
    else:
      history.append(test[i, :])
  predictions = array(predictions)
  # evaluate predictions days for each week
  score, scores = evaluate_forecasts(test[:, :], predictions)
  return score, scores, predictions

# score a model, return None on failure
def score_model(data, n_test, n_days, func, cfg, debug=False):
  result = None
  # convert config to a key
  key = str(cfg)
  # show all warnings and fail on exception if debugging
  if debug:
    result = evaluate_model(data, n_test, n_days, func, cfg)
  else:
    # one failure during model validation suggests an unstable config
    try:
      # never show warnings when grid searching, too noisy
      with catch_warnings():
        filterwarnings("ignore")
        result = evaluate_model(data, n_test, n_days, func, cfg)
    except:
      error = None
  # check for an interesting result
  if result is not None:
    print('> Model[%s] %.3f' % (key, result))
  return (key, result)

  # grid search configs
def grid_search(func, data, cfg_list, n_test, n_days, parallel=True):
  scores = None
  # split into train and test
  if parallel:
    # execute configs in parallel
    executor = Parallel(n_jobs=cpu_count(), backend= 'multiprocessing')
    tasks = (delayed(score_model)(data, n_test, n_days, func, cfg) for cfg in cfg_list)
    scores = executor(tasks)
  else:
    scores = [score_model(data, n_test, n_days, func, cfg) for cfg in cfg_list]
  # remove empty results
  scores = [r for r in scores if r[1] != None]
  # sort configs by error, asc
  scores.sort(key=lambda tup: tup[1])
  return scores

# weekly persistence model
def weekly_persistence(history):
  # get the data for the prior week
  last_week = history[-1]
  return last_week[:]

# week one year ago persistence model
def week_one_year_ago_persistence(history):
  # get the data for the prior week
  last_week = history[-52]
  return last_week[:]

# daily persistence model
def daily_persistence(history):
  # get the data for the prior week
  last_week = history[-1]
  # get the total active power for the last day
  value = last_week[-1]
  # prepare 7 day forecast
  forecast = [value for _ in range(7)]
  return forecast

# convert windows of weekly multivariate data into a series of total power
def to_series(data):
  # extract just the total power from each week
  series = [week[:] for week in data]
  # flatten into a single series
  series = array(series).flatten()
  return series

# arima forecast
def arima_forecast(history, cfg, model_return=False):
  order = cfg
  # convert history into a univariate series
  series = to_series(history)
  # define the model
  model = ARIMA(series, order=cfg)
  # fit the model
  model_fit = model.fit(disp=False)
  # make forecast
  yhat = model_fit.predict(len(series), len(series)+6)
  return model_fit if model_return else yhat

# create a set of sarima configs to try
def arima_configs(params=None):
  if not 'p_params' in params or params==None:
    p_params = [7]
    d_params = [0]
    q_params = [0]
  else:
    p_params = params["p_params"]
    q_params = params["q_params"]
    d_params = params["d_params"]

  models = list()
  # create config instances
  for p in p_params:
    for d in d_params:
      for q in q_params:
          cfg = (p, d, q)
          models.append(cfg)
  return models

# one-step sarima forecast
def sarima_forecast(history, config, model_return=False):
  order, sorder, trend = config
  series = to_series(history)
  # define model
  model = SARIMAX(series, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
  # fit model
  model_fit = model.fit(disp=False)
  # make one step forecast
  yhat = model_fit.predict(len(series), len(series)+6)
  return model_fit if model_return else yhat

# create a set of sarima configs to try
def sarima_configs(params=None):
  if not 'p_params' in params or params==None:
    # define config lists
    p_params = [7]
    d_params = [0]
    q_params = [0]
    # t_params = ['n', 'c', 't', 'ct']
    t_params = ['n']
    P_params = [0]
    D_params = [0]
    Q_params = [0]
    m_params = [0]
  else:
    p_params = params["p_params"]
    q_params = params["q_params"]
    d_params = params["d_params"]
    t_params = params["t_params"]
    P_params = params["P_params"]
    D_params = params["D_params"]
    Q_params = params["Q_params"]
    m_params = params["m_params"]

  models = list()
  # create config instances
  for p in p_params:
    for d in d_params:
      for q in q_params:
        for t in t_params:
          for P in P_params:
            for D in D_params:
              for Q in Q_params:
                for m in m_params:
                  cfg = [(p,d,q), (P,D,Q,m), t]
                  models.append(cfg)
  return models

# create a set of exponential smoothing configs to try
def exp_smoothing_configs(params=None):
    # # define config lists
    # t_params = ['add', 'mul', None]
    # d_params = [True, False]
    # s_params = ['add', 'mul', None]
    # p_params = seasonal
    # b_params = [True, False]
    # r_params = [True, False]
    if not 'p_params' in params or params==None:
      # define config lists
      t_params = [None]
      d_params = [False]
      s_params = [None]
      p_params = [None]
      b_params = [False]
      r_params = [False]
    else:
      t_params = params["t_params"]
      d_params = params["d_params"]
      s_params = params["s_params"]
      p_params = params["p_params"]
      b_params = params["b_params"]
      r_params = params["r_params"]

    
    models = list()
    # create config instances
    for t in t_params:
        for d in d_params:
            for s in s_params:
                for p in p_params:
                    for b in b_params:
                        for r in r_params:
                            cfg = [t,d,s,p,b,r]
                            models.append(cfg)
    return models

# one-step Holt Winters Exponential Smoothing forecast
def exp_smoothing_forecast(history, config, model_return=False):
    t,d,s,p,b,r = config
    # define model
    series = to_series(history)
    series = array(series)
    model = ExponentialSmoothing(series, trend=t, damped=d, seasonal=s, seasonal_periods=p)
    # fit model
    model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
    # make one step forecast
    yhat = model_fit.predict(len(series), len(series)+6)
    return model_fit if model_return else yhat

def predictions_to_weboutput_all(y_hat, begin, end):
  period = pd.date_range(begin, end)
  returned_dictionary = list()
  for date, value in zip(period, y_hat):
      returned_dictionary.append(
          {
              "date": datetime.datetime.strftime(date, "%Y-%m-%d"),
              "prediction": str(value),
          }
      )

  return returned_dictionary