"""
Processing the data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def process_data(train, test, lags):
    """Process data
    Reshape and split train\test data.

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag.
    # Returns
        X_train: ndarray (flow data).
        X_train_time: ndarray (time features).
        y_train: ndarray.
        X_test: ndarray (flow data).
        X_test_time: ndarray (time features).
        y_test: ndarray.
        scaler: MinMaxScaler.
    """
    attr = 'Lane 1 Flow (Veh/5 Minutes)'
    df1 = pd.read_csv(train, encoding='utf-8', parse_dates=['5 Minutes']).fillna(0)
    df2 = pd.read_csv(test, encoding='utf-8', parse_dates=['5 Minutes']).fillna(0)

    # Add time-based features
    for df in [df1, df2]:
        df['hour'] = df['5 Minutes'].dt.hour
        df['day_of_week'] = df['5 Minutes'].dt.dayofweek
        df['month'] = df['5 Minutes'].dt.month
        df['is_weekend'] = df['5 Minutes'].dt.dayofweek.isin([5, 6]).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)

    scaler = MinMaxScaler(feature_range=(0, 1))
    flow1 = scaler.fit_transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    train, train_time, test, test_time = [], [], [], []
    for i in range(lags, len(flow1)):
        train.append(flow1[i - lags: i + 1])
        train_time.append([
            df1['hour'].iloc[i],
            df1['day_of_week'].iloc[i],
            df1['month'].iloc[i],
            df1['is_weekend'].iloc[i],
            df1['hour_sin'].iloc[i],
            df1['hour_cos'].iloc[i]
        ])
    for i in range(lags, len(flow2)):
        test.append(flow2[i - lags: i + 1])
        test_time.append([
            df2['hour'].iloc[i],
            df2['day_of_week'].iloc[i],
            df2['month'].iloc[i],
            df2['is_weekend'].iloc[i],
            df2['hour_sin'].iloc[i],
            df2['hour_cos'].iloc[i]
        ])

    train = np.array(train)
    train_time = np.array(train_time)
    test = np.array(test)
    test_time = np.array(test_time)

    # Shuffle the training data
    shuffle_index = np.random.permutation(len(train))
    train = train[shuffle_index]
    train_time = train_time[shuffle_index]

    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    return X_train, train_time, y_train, X_test, test_time, y_test, scaler