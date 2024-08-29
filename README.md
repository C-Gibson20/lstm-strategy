# LSTM Strategy

<p>This project aims to predict stock prices using a Long Short-Term Memory (LSTM) neural network model. The model is trained using historical stock data to generate future predictions. The implementation includes both a research environment for model development and a live trading algorithm to execute trades based on the model's predictions.</p>

### Features
**Data Loading and Preprocessing**: Historical stock data is loaded and preprocessed, including feature engineering with various technical indicators such as EMA, SMA, MACD, and rolling statistics.<br>
**Feature Selection**: A Random Forest Regressor is used to select the most important features for the LSTM model.<br>
**LSTM Model**: The model is built using TensorFlow/Keras with configurable parameters such as LSTM units, dropout rates, and learning rates. It is trained using walk-forward validation to ensure robust performance.<br>
**Walk-Forward Validation**: The model's performance is evaluated iteratively over multiple time windows, with metrics like RMSE, MAE, and MAPE.<br>
**Live Trading Algorithm**: The QuantConnect-based algorithm trades based on the predictions from the trained LSTM model. It monitors the stock price and executes buy/sell orders based on the predicted price movements.<br>
