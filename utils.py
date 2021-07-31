from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error


def adf_test(dataframe) -> None:
    adf, pvalue, _, _, critical_values, _ = adfuller(dataframe)
    is_stationary = pvalue <= 0.05
    test_stat_p_value = round(critical_values['5%'], 2)
    pvalue = round(pvalue, 2)
    adf = round(adf, 2)
    print(
        f'Is the time series stationary? {is_stationary}\n'
        f'test statistic value = {adf}\n'
        f'p value = {pvalue}\n'
        f'test critical values (5%) = {test_stat_p_value}'
    )


def forecast_with_interval(result, forecast):
    res = result.copy(deep=True)
    res['min'] = pd.Series(dtype=float)
    res['max'] = pd.Series(dtype=float)
    for i in range(forecast.shape[0]):
        res['passengers'][forecast.index[i]] = forecast.iloc[i]['passengers']
        res['min'][forecast.index[i]] = forecast.iloc[i]['mean_ci_lower']
        res['max'][forecast.index[i]] = forecast.iloc[i]['mean_ci_upper']
    res['log_out'] = res['log_passengers'].shift(1) + res['passengers']
    res['log_min'] = res['log_passengers'].shift(1) + res['min']
    res['log_max'] = res['log_passengers'].shift(1) + res['max']
    res['scale_out'] = np.exp(res['log_out'])
    res['scale_min'] = np.exp(res['log_min'])
    res['scale_max'] = np.exp(res['log_max'])

    return res


def plot_forecast_interval(forecast_result):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    ax1.plot(forecast_result['original_data'], label='original data')
    ax1.plot(forecast_result['scale_out']
             ['1958-01-01': '1958-12-01'], label='forecasted data')
    ax1.fill_between(forecast_result['1958-01-01': '1958-12-01'].index,
                     forecast_result['scale_min']['1958-01-01': '1958-12-01'],
                     forecast_result['scale_max']['1958-01-01': '1958-12-01'],
                     color='k', alpha=0.1, label="90% confidence interva")
    ax1.axvline(x=datetime(1957, 12, 1), c='red', linestyle='--')
    ax2.plot(forecast_result['original_data']
             ['1957-10-01': '1958-12-01'], label='original data')
    ax2.axvline(x=datetime(1957, 12, 1), c='red', linestyle='--')
    ax2.plot(forecast_result['scale_out']
             ['1957-12-01': '1958-12-01'], label='forecasted data')
    ax2.fill_between(forecast_result['1958-01-01': '1958-12-01'].index,
                     forecast_result['scale_min']['1958-01-01': '1958-12-01'],
                     forecast_result['scale_max']['1958-01-01': '1958-12-01'],
                     color='k', alpha=0.1, label="90% confidence interva")
    ax1.set_title('Forecast')
    ax2.set_title('Zoom on forecast')
    ax1.legend()
    ax2.legend(loc=2)
    plt.show()


def expanding_window(window: int, order_ARIMA: tuple, X_train, X_test):
    nb_forecast = X_test.shape[0]
    nb_expanding_window = int(np.ceil(nb_forecast / window))
    X_train_exp = X_train.copy()
    original_model = ARIMA(endog=X_train_exp, order=order_ARIMA)
    fitted_model = original_model.fit()
    forecast = fitted_model.predict(
        start=X_test.index[0], end=X_test.index[window - 1])

    forecast_list = list(forecast.iloc[: window])
    for i in tqdm(range(nb_expanding_window - 1)):
        X_train_exp = pd.concat([X_train_exp, X_test[i*window: (i+1)*window]])
        exp_model = ARIMA(endog=X_train_exp, order=order_ARIMA)
        updated_fitted = exp_model.fit()
        forecast = updated_fitted.predict(start=X_test.index[(
            i+1)*window], end=X_test.index[min(nb_forecast - 1, (i+2)*window - 1)])
        if window == 1:
            forecast_list.append(forecast.iloc[0])
        else:
            forecast_list.extend(forecast)
    return forecast_list


def plot_result(X_test, preds, title):
    fig, ax = plt.subplots(1, 2, figsize=(15, 8), sharey=True)

    ax[0].plot(preds, label='predictions')
    ax[0].plot(X_test, label='original data')
    ax[1].plot(X_test - preds, label='residuals', c='g')
    fig.suptitle(title)
    ax[0].legend()
    ax[1].legend()
    ax[0].tick_params(axis='x', labelrotation=60)
    ax[1].tick_params(axis='x', labelrotation=60)
    ax[1].tick_params(labelleft=True)
    plt.show()


def score_and_plot(X_test, predictions, title):
    pred_test = X_test.copy()
    pred_test['passengers'] = predictions
    mape_score = mean_absolute_percentage_error(X_test, pred_test)
    print(f'\nMAPE score: {mape_score:.4f}')
    plot_result(X_test, pred_test, title + f' - MAPE score: {mape_score:.4f}')
    return mape_score


def add_forecast_data(result, forecast):
    res = result.copy(deep=True)
    for i in range(forecast.shape[0]):
        res['passengers'][forecast.index[i]] = forecast.iloc[i]['passengers']
    res['log_out'] = res['log_passengers'].shift(1) + res['passengers']
    res['scale_out'] = np.exp(res['log_out'])
    return res


def plot_forecast(forecast_result):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    ax1.plot(forecast_result['original_data'], label='original data')
    ax1.plot(forecast_result['scale_out']
             ['1958-01-01': '1958-12-01'], label='forecasted data')
    ax1.axvline(x=datetime(1957, 12, 1), c='red', linestyle='--')
    ax2.plot(forecast_result['original_data']
             ['1957-10-01': '1958-12-01'], label='original data')
    ax2.axvline(x=datetime(1957, 12, 1), c='red', linestyle='--')
    ax2.plot(forecast_result['scale_out']
             ['1957-12-01': '1958-12-01'], label='forecasted data')
    ax1.set_title('Forecast')
    ax2.set_title('Zoom on forecast')
    ax1.legend()
    ax2.legend()
    plt.show()
