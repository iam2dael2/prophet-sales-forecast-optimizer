import os
import numpy as np
import pandas as pd
import warnings
from scipy.stats import pearsonr
from commons import utils, forecasting
from prophet.serialize import model_to_json, model_from_json

warnings.filterwarnings('ignore')

# Import data
df_qty: pd.DataFrame = pd.read_csv("datasets/1_target_ts.csv")
df_price: pd.DataFrame = pd.read_csv("datasets/2_related_ts.csv", header=None, names=["item", "org", "date", "unit_price"])

# Combine both data into one
df_combined: pd.DataFrame = pd.merge(left=df_qty, right=df_price, on=["item", "org", "date"])

# Change data-type for column `date`
df_combined['date'] = pd.to_datetime(df_combined['date'], format='%Y-%m-%d')

# Separate data for distinct item and org
data: dict = utils.structuring_data(
    df_combined, 
    date_col="date", 
    first_order_col="org", 
    second_order_col="item"
)

count = 0
# Iterate for each item and organization to forecast
for item in data.keys():
    for org in data[item].keys():
        
        count += 1
        print(f"Iteration {count}\n\n")
        if count > 10:
            break

        # Indexing data based on item and organization
        chosen_data = data[1617388][3994423].copy()

        # Calculate coefficient of correlation between quantity and unit price
        corr_coef, _ = pearsonr(chosen_data.quantity, chosen_data.unit_price)
        print(f"[Item: {item} | Organization: {org}]\nCorrelation of Quantity vs Price: {corr_coef}")

        # Modify data to be able to be run in FBProphet
        df_modified = utils.prepare_data(
            data=chosen_data,
            ds="date",
            y="quantity"
        )

        # Prepare `unit_price` time series data to be forecasted
        df_modified_price = utils.prepare_data(
            data=chosen_data,
            ds="date",
            y="unit_price"
        ).drop("quantity", axis=1)

        # Forecast `unit_price` for specified item
        price_model, price_forecasts = forecasting.forecast_using_prophet(
            data_to_fit=df_modified_price,
            periods=12
        )

        # Cross validate on quantity data
        qty_model_params_cv = forecasting.fit_cv(
            data_to_fit=df_modified,
            periods=3,
            extra_regressor="unit_price"
        )

        # Choose the best model among probable parameters
        params_sorted_by_rmse = qty_model_params_cv.sort_values(by="rmse", ascending=True)
        best_params = params_sorted_by_rmse.drop('rmse', axis=1).iloc[0].to_dict()

        best_model_params = {param: value for param, value in best_params.items() if param != 'mode'}
        extra_regressor_mode = best_params['mode']

        # Forecast `quantity` for specified item
        qty_model, qty_forecasts = forecasting.forecast_using_prophet(
            data_to_fit=df_modified,
            periods=12,
            model_params=best_model_params,
            extra_regressor_name='unit_price',
            extra_regressor_mode=extra_regressor_mode,
            future=price_forecasts[['ds', 'yhat']].rename(columns={'yhat': 'unit_price'})
        )

        # Save the Prophet model, both unit_price and qty. 
        destination_file_path = f'models/org_{org}/item_{item}'

        # Make an empty directory, if not existed
        destination_file_path_splits = destination_file_path.split("/")
        for i in range(len(destination_file_path_splits)):
            sub_path = "/".join(destination_file_path_splits[:i+1])
            if not os.path.exists(sub_path):
                os.mkdir(sub_path)

        # NOTE: Uncomment this, if you want to save the model
        # with open(f'{destination_file_path}/price_model.json', 'w') as fout:
        #     fout.write(model_to_json(price_model))

        # with open(f'{destination_file_path}/qty_model.json', 'w') as fout: 
        #     fout.write(model_to_json(qty_model))

        # Save the forecast plot in image
        utils.plot_forecast(
            model=qty_model,
            forecasts=qty_forecasts,
            is_saved=True,
            file_path_to_save=f"{destination_file_path}/forecast.png"
        )