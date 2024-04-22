import logging

# Disable specific warning regarding to prophet plot
logging.getLogger("prophet.plot").disabled = True

import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
from prophet import Prophet

def show_progress(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"{func.__name__}: Done")
        return result
    return wrapper

@show_progress
def structuring_data(
        df: pd.DataFrame, 
        date_col: str, 
        first_order_col: str, 
        second_order_col: str,
    ) -> dict :

    """
    Serialize the specified dataframe into dictionary of two-level keys

    Parameters
    ------------
        df: specified DataFrame
        date_col: date-type column contained in df
        first_order_col: first column contained in df
        second_order_col: second column contained in df

    Return
    ------------
        data: two-level dictionary of df
    """
    data = {}

    for feature_1 in df[first_order_col].unique():

        # Filtering to column `item`
        sub_df = df[df[first_order_col] == feature_1]
        data[feature_1] = {}

        for feature_2 in sub_df[second_order_col].unique():

            # Resample index to be monthly date
            sub_data = sub_df[sub_df[second_order_col] == feature_2].set_index(date_col).resample("1MS").agg({"quantity": "sum", "unit_price": "mean"})
            
            # Add this data in dictionary
            data[feature_1][feature_2] = sub_data.reset_index()

    return data

@show_progress
def plot_2_ts_data(
    date_index: pd.DatetimeIndex, 
    data1: pd.Series, 
    data2: pd.Series, 
    **kwargs
) -> None:
    """
    Plot two time series data in 1 figure, sharing same x-axis

    Parameters
    ------------
        date_index: date-type column or indexes
        data1: first time series data
        data2: second time series data
    """

    # Figure 1
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:red"
    ax1.set_xlabel("Monthly Date")
    ax1.set_ylabel(kwargs.get("label_1"), color=color)
    ax1.plot(date_index, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Figure 2
    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel(kwargs.get("label_2"), color=color)
    ax2.plot(date_index, data2, color=color, linestyle='dotted')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(kwargs.get("title"))
    plt.show()

@show_progress
def prepare_data(
    data: Union[pd.DataFrame, pd.Series], 
    ds: str, 
    y: str,
    ds_format: str = '%m'
) -> Union[pd.DataFrame, pd.Series]:
    """
    Preprocess data to be able to continue on forecasting by Prophet

    Parameters
    ------------
        data: pandas DataFrame or Series
            specified data containing `date` and `observed value` columns

        ds: str
            name of column representing `date` of data

        y: str
            name of column representing `observed value` of data

        ds_format: str
            specified format of date (based on `strftime documentation`)

    Returns
    ------------
        modified_data: pandas DataFrame or Series
            resulting data
    """
    modified_data = data.rename(columns={ds: "ds", y: "y"})
    modified_data['ds'] = pd.to_datetime(modified_data['ds'], format=ds_format)
    return modified_data

@show_progress
def plot_forecast(
    model: Prophet,
    forecasts: pd.DataFrame,
    x_label: str = 'Date',
    y_label: str = 'Quantity of Interest',
    is_saved: bool = False,
    file_path_to_save: str = None
) -> None:
    """
    Plot the forecast within specified prophet model

    Parameters
    ------------

    """
    model.plot(forecasts)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{x_label} vs {y_label}')
    plt.show()

    plt.savefig(file_path_to_save, bbox_inches='tight') \
    if is_saved and file_path_to_save is not None \
    else None