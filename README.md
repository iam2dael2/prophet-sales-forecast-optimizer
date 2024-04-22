## Prophet Sales Forecast Optimizer

The "Prophet Sales Forecast Optimizer" is a powerful tool designed to enhance sales forecasting accuracy by leveraging hyperparameter tuning techniques. It employs the Prophet forecasting algorithm, which is an open-source tool developed by Facebook for time series forecasting tasks.
Here's a breakdown of its key components and functionalities:

* <b>Prophet Algorithm</b><br>
   The optimizer utilizes the Prophet algorithm, which is specifically designed to handle time series data with strong seasonal patterns and multiple seasonality. It automatically detects seasonality in the data and adjusts forecasts accordingly.
* <b>Hyperparameter Tuning</b><br>
   Hyperparameter tuning is a process of optimizing the parameters of a model to achieve the best performance. The optimizer employs techniques such as grid search, random search, or Bayesian optimization to systematically search through the hyperparameter space and find the optimal set of parameters for the Prophet algorithm.
* <b>Forecast Accuracy</b><br>
   By fine-tuning the model parameters, the optimizer aims to maximize forecast accuracy. It takes into account various factors such as trend, seasonality, holidays, and special events to generate precise forecasts.
* <b>Automated-best parameters</b><br>
   The optimizer offers best parameters by automated approach to tailor the forecasting model to specific business needs. Users obtain adjusted parameters such as seasonality prior scale, seasonality mode, and changepoint prior scale to fine-tune the model's behavior.
* <b>Performance Evaluation</b><br>
   After hyperparameter tuning, the optimizer evaluates the performance of the optimized model using metrics such as Root Mean Squared Error (RMSE). This helps users assess the effectiveness of the forecasting model and make informed decisions.

## Getting Started

Follow these steps to get started with the Prophet Sales Forecast Optimizer:

1. Install dependencies by running
   
   ```
   pip install -r requirements.txt
   ```
2. Run `main.py`

   ```
   python3 main.py
   ```

## Acknowledgements

* The Prophet forecasting algorithm developed by Facebook.
* Hyperparameter tuning techniques inspired by recent advancements in machine learning optimization.
