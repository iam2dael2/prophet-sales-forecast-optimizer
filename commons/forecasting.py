import logging

# Set logging level to suppress informational messages
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

# Disable specific warning regarding to prophet plot
logging.getLogger("prophet.plot").disabled = True

import pandas as pd
import itertools
from prophet import Prophet
from prophet.diagnostics import cross_validation
from dateutil.relativedelta import relativedelta
from prophet.diagnostics import performance_metrics

def show_progress(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"{func.__name__}: Done")
        return result
    return wrapper

@show_progress
def forecast_using_prophet(
    data_to_fit: pd.DataFrame, 
    periods: int, 
    freq: str = 'MS',
    model_params: dict = None,
    **extra_regressor_kwargs
) -> pd.DataFrame:
    """
    Forecast values for next observation by using Prophet model

    Parameters
    ------------
        data_to_fit: pd.DataFrame
            specified data to fit

        periods: int
            number of forecast's periods

        freq: str
            specified frequency of period

        model_params: dict
            specified parameters of Prophet model
    
    Returns
    ------------
        forecast: pd.DataFrame
            resulting forecast with more additional information
    """
    # Initialize prophet model
    extra_regressor_name: str = extra_regressor_kwargs.get('extra_regressor_str')
    extra_regressor_mode: str = extra_regressor_kwargs.get('extra_regressor_mode')

    model = Prophet(**model_params) if model_params is not None else Prophet()
    model.add_regressor(
        name=extra_regressor_name, 
        prior_scale=0.5,
        mode=extra_regressor_mode
    ) if None not in (extra_regressor_name, extra_regressor_mode) else None
    model.fit(df=data_to_fit)

    # Specify which period to be forecasted
    future = model.make_future_dataframe(periods=periods, freq=freq) \
             if extra_regressor_kwargs.get('future') is None \
             else extra_regressor_kwargs.get('future') 

    # Resulting forecasts
    forecasts = model.predict(future)

    return model, forecasts

@show_progress
def fit_cv(
    data_to_fit: pd.DataFrame,
    periods: int,
    horizon: str = '90 days',
    extra_regressor: str = None
) -> pd.DataFrame:
    """
    Return listing of model's parameters with corresponding error

    Parameters
    ------------
        data_to_fit: pd.DataFrame
            specified data to fit

        periods: int
            number of periods to be forecasted

        horizon: str
            size of interval (pd.timeDelta)

        extra_regressor: str
            name of regressor contained in `data_to_fit`

    Returns
    ------------
        tuning_results: pd.DataFrame
            Listing of parameters with corresponding errors
    """
    param_grid = {  
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative'],
        'mode': ['additive', 'multiplicative'],
    }

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here

    # Use cross validation to evaluate all parameters
    for params in all_params:
        
        # Initialize parameters for model
        model_params_init = {param: value for param, value in params.items() if param != 'mode'}

        # Initialize parameter for extra regressor
        extra_param = {param: value for param, value in params.items() if param == 'mode'}

        # Fit model with given params
        model = Prophet(**model_params_init)  
        model.add_regressor(extra_regressor, prior_scale=0.5, **extra_param) if extra_regressor is not None else None
        model.fit(data_to_fit)

        # Cross validation
        size = len(data_to_fit) - (periods * 3)

        # ==> Assign cutoffs
        start_dt = data_to_fit.loc[size, "ds"]
        end_dt = data_to_fit.iloc[-1]['ds']
        cutoffs = [date for date in pd.date_range(start=start_dt, end=end_dt, freq=f'{periods-1}MS') if date < (end_dt - relativedelta(months=periods))]

        df_cv = cross_validation(model, cutoffs=cutoffs, horizon=horizon, parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    # Find the best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses

    return tuning_results