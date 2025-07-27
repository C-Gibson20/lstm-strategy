# LSTM Strategy

<p>This project implements a robust framework for forecasting stock prices using LSTM neural networks, enhanced with feature engineering, feature selection via random forests, and walk-forward validation. The system is designed for deployment within the QuantConnect platform using a trained Keras model for live trading.</p>
<br>

## Features

* **LSTM Neural Network**: Two-layer LSTM model with dropout for regularization.
* **Feature Engineering**: Generates technical indicators such as EMA, SMA, MACD, and rolling statistics.
* **Feature Selection**: RandomForestRegressor used to identify the most informative features.
* **Walk-Forward Validation**: Robust backtesting using expanding windows and step-wise retraining.
* **Evaluation Metrics**: RMSE, MAE, and MAPE tracked across all validation iterations.
* **QuantConnect Integration**: Trained LSTM model is serialized and used for daily trading decisions.

<br>

## Implementation

### Data Processing

* Historical price data is fetched using QuantConnect’s `QuantBook`.
* Features engineered include:

  * Exponential Moving Averages (EMA10, EMA30)
  * Simple Moving Average (SMA50)
  * MACD
  * Lag features (1-day, 5-day)
  * Rolling mean and standard deviation (20-day window)

### Feature Selection

* Uses `RandomForestRegressor` to rank feature importance.
* Top N features are retained for model training to reduce noise and dimensionality.

### Model Architecture

```python
Sequential([
    LSTM(128, return_sequences=True),
    Dropout(0.4),
    LSTM(64),
    Dropout(0.4),
    Dense(1)
])
```

* Optimizer: `Adam` with configurable learning rate
* Loss: Mean Squared Error (MSE)

### Model Training

* Input sequences of shape `(time_step, n_features)`
* Early stopping and learning rate reduction callbacks
* Training, validation, and testing split manually for final evaluation

### Walk-Forward Validation

* Incrementally expands the training window
* Applies stride-based window shifts
* Retrains model at each step and evaluates performance
* Scoring metrics per iteration: RMSE, MAE, MAPE, validation loss, and learning rate

<br>

## Deployment: QuantConnect Live Trading

* Uses the `LSTMPrediction` class derived from `QCAlgorithm`
* Daily trade logic compares consecutive model predictions:

  * **Buy** when next prediction > last
  * **Sell** when next prediction ≤ last
* Model retraining scheduled at month-end
* RollingWindow keeps 4000 historical data points for real-time inference
