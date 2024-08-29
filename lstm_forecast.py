# region imports
from AlgorithmImports import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import json
# endregion

CONFIG = {
    'time_step': 3,
    'lstm_units': [128, 64],
    'dropout_rate': 0.25,
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'n_features': 1,
    'test_size': 0.2,
    'val_size': 0.1
}


def load_data(data): 
    data = data.reset_index()['close']
    # data = add_features(data)
    data.dropna(inplace=True)
    return data


def add_features(data):
    df = pd.DataFrame(data, columns=['close'])
    df['Lag_1'] = df['close'].shift(1)
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_30'] = df['Close'].ewm(span=30, adjust=False).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['MACD'] = df['EMA_10'] - df['EMA_30']   
    df['Lag_5'] = df['Close'].shift(5)
    df['Rolling_Mean_20'] = df['Close'].rolling(window=20).mean()
    df['Rolling_Std_20'] = df['Close'].rolling(window=20).std()
    return df


def feature_selection(data, n_features):
    X = data.drop(columns=['close'])
    y = data['close']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    top_features = X.columns[np.argsort(model.feature_importances_)[::-1][:n_features-1]]
    X_selected = X[top_features]
    X_selected = X_selected.copy()
    X_selected['close'] = y          

    return X_selected


def normalize_features(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def build_lstm_model(input_shape, lstm_units, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=lstm_units[0], return_sequences=True)) 
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=lstm_units[1])) 
    model.add(Dropout(dropout_rate))
    model.add(Dense(1)) 

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def get_callbacks():
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        min_delta=1e-4,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    return [early_stopping, reduce_lr]


def create_sequences(data, time_step):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, :])
        y.append(data[i, -1])
    return np.array(X), np.array(y)


def retrain_model(algorithm, data, model_name):
    data = load_data(data)
    data = np.array(data).reshape(-1, 1)
    scaled_data, scaler = normalize_features(data)
    X, y = create_sequences(scaled_data, CONFIG['time_step'])
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    n_data_points = X.shape[0]
    val_size = int(0.2 * n_data_points)

    X_train, X_val = X[: - val_size], X[- val_size:]
    y_train, y_val = y[: - val_size], y[- val_size:]

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), CONFIG['lstm_units'], CONFIG['dropout_rate'], CONFIG['learning_rate'])
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=get_callbacks(),
        verbose=1
    )

    json_config = model.to_json()
    algorithm.ObjectStore.Save(model_name, json_config)


def forecast(algorithm, data, model):
    data = load_data(data)
    data = np.array(data).reshape(-1, 1)
    last_sequence = data[-CONFIG['time_step']:, :]
    scaled_last_sequence, scaler = normalize_features(last_sequence) 
    scaled_last_sequence = scaled_last_sequence.reshape(1, CONFIG['time_step'], CONFIG['n_features'])
    prediction = model.predict(scaled_last_sequence)
    return scaler.inverse_transform(prediction) 

    # new_row = np.zeros((1, CONFIG['n_features']))  
    # new_row[0, 0] = scaled_last_sequence[0, -1, 0]
    # new_row[0, -1] = prediction
    # scaled_last_sequence = np.concatenate([scaled_last_sequence[0, 1:, :], new_row], axis=0)
    
    # last_sequence_inversed = scaler.inverse_transform(scaled_last_sequence)
    # return last_sequence_inversed[-1, -1]
