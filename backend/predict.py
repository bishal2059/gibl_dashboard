import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import tensorflow as tf
from pandas.tseries.offsets import MonthBegin
from tensorflow.keras.models import load_model

model_dir = './modelsEachBranch'

# Load the CSV files
branch_data = pd.read_csv("./data/global_ime_enhanced_branch_data.csv")
monthly_performance = pd.read_csv("./data/global_ime_monthly_performance_enhanced.csv")
customer_segments = pd.read_csv("./data/global_ime_customer_segments.csv")

# Merge all three DataFrames on 'branch_name'
merged_df = branch_data.merge(monthly_performance, on='branch_id', how='inner')
merged_df = merged_df.merge(customer_segments, on='branch_id', how='inner')

# Save the combined DataFrame to a new CSV file
merged_df.to_csv("global_ime_combined.csv", index=False)

# Assuming merged_df is already loaded as a pandas DataFrame
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def preprocess_data(df):
    df = df.copy()
    df['month_dt'] = pd.to_datetime(df['month'], format='%Y-%m')
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != 'month']
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    df = df.sort_values(['branch_id', 'month_dt'])
    return df, label_encoders


def create_sequences(group, feature_cols, n_timesteps=6, target_col='revenue'):
    X, y, months = [], [], []
    data = group.copy()
    values = data[feature_cols + [target_col]].values
    for i in range(len(values) - n_timesteps):
        X.append(values[i:i + n_timesteps, :-1])
        y.append(values[i + n_timesteps, -1])
        months.append(data['month'].values[i + n_timesteps])
    return np.array(X), np.array(y), months

def generate_future_months(last_dt, periods=12):
    return pd.date_range(start=last_dt + MonthBegin(), periods=periods, freq='MS').strftime('%Y-%m').tolist()

def forecast_next_n_months(df_branch, model_dir, timesteps=6, n_months=12):
    # np.random.seed(seed)
    branch_id = df_branch['branch_id'].iloc[0]

    # Load saved model & preprocessors
    model = load_model(f'{model_dir}/branch_{branch_id}_lstm.h5', compile=False)
    scaler_X = joblib.load(f'{model_dir}/branch_{branch_id}_scaler_X.pkl')
    scaler_y = joblib.load(f'{model_dir}/branch_{branch_id}_scaler_y.pkl')
    seasonal_factors = joblib.load(f'{model_dir}/branch_{branch_id}_seasonal.pkl')
    trend_model = joblib.load(f'{model_dir}/branch_{branch_id}_trend.pkl')
    # noise_std = joblib.load(f'{model_dir}/branch_{branch_id}_noise_std.pkl')
    feature_cols = joblib.load(f'{model_dir}/branch_{branch_id}_features.pkl')

    # Sort and prep
    df_branch = df_branch.sort_values('month_dt').copy()
    df_branch['time_index'] = (df_branch['month_dt'] - df_branch['month_dt'].min()).dt.days
    base_time = df_branch['month_dt'].min()

    last_rows = df_branch[feature_cols + ['revenue']].tail(timesteps).copy()
    window = last_rows.values.copy()
    last_dt = df_branch['month_dt'].iloc[-1]
    # last_dt_minus24 = last_dt - pd.DateOffset(months=24)
    future_months = generate_future_months(last_dt, periods=n_months)

    preds = []
    for m in future_months:
        month_dt = pd.to_datetime(m)
        month_num = month_dt.month
        seasonal_factor = seasonal_factors.get(month_num, 1.0)

        future_index = (month_dt - base_time).days
        current_index = df_branch['time_index'].max()
        future_df = pd.DataFrame({'time_index': [future_index]})
        current_df = pd.DataFrame({'time_index': [current_index]})
        trend_base = trend_model.predict(future_df)[0]
        trend_current = trend_model.predict(current_df)[0]
        # trend_base = trend_model.predict([[future_index]])[0]
        # trend_current = trend_model.predict([[current_index]])[0]
        trend_factor = trend_base / trend_current if abs(trend_current) > 1e-6 else 1.0

        x_input = window[-timesteps:, :-1]
        x_scaled = scaler_X.transform(x_input).reshape(1, timesteps, -1)

        base_prediction_scaled = model.predict(x_scaled, verbose=0)[0][0]
        base_prediction = scaler_y.inverse_transform([[base_prediction_scaled]])[0][0]

        adjusted = base_prediction * seasonal_factor * trend_factor
        # adjusted += np.random.normal(0, noise_std)
        adjusted = max(0, adjusted)  # Clamp to non-negative

        preds.append(adjusted)

        next_features = window[-1, :-1]
        next_row = np.concatenate([next_features, [adjusted]])
        window = np.vstack([window, next_row])

    return future_months, preds


def forecast_next_n_months(df_branch, model_dir, timesteps=6, n_months=12):
    # np.random.seed(seed)
    branch_id = df_branch['branch_id'].iloc[0]

    # Load saved model & preprocessors
    model = load_model(f'{model_dir}/branch_{branch_id}_lstm.h5', compile=False)
    scaler_X = joblib.load(f'{model_dir}/branch_{branch_id}_scaler_X.pkl')
    scaler_y = joblib.load(f'{model_dir}/branch_{branch_id}_scaler_y.pkl')
    seasonal_factors = joblib.load(f'{model_dir}/branch_{branch_id}_seasonal.pkl')
    trend_model = joblib.load(f'{model_dir}/branch_{branch_id}_trend.pkl')
    # noise_std = joblib.load(f'{model_dir}/branch_{branch_id}_noise_std.pkl')
    feature_cols = joblib.load(f'{model_dir}/branch_{branch_id}_features.pkl')

    # Sort and prep
    df_branch = df_branch.sort_values('month_dt').copy()
    df_branch['time_index'] = (df_branch['month_dt'] - df_branch['month_dt'].min()).dt.days
    base_time = df_branch['month_dt'].min()

    last_rows = df_branch[feature_cols + ['revenue']].tail(timesteps).copy()
    window = last_rows.values.copy()
    last_dt = df_branch['month_dt'].iloc[-1]
    # last_dt_minus24 = last_dt - pd.DateOffset(months=24)
    future_months = generate_future_months(last_dt, periods=n_months)

    preds = []
    for m in future_months:
        month_dt = pd.to_datetime(m)
        month_num = month_dt.month
        seasonal_factor = seasonal_factors.get(month_num, 1.0)

        future_index = (month_dt - base_time).days
        current_index = df_branch['time_index'].max()
        future_df = pd.DataFrame({'time_index': [future_index]})
        current_df = pd.DataFrame({'time_index': [current_index]})
        trend_base = trend_model.predict(future_df)[0]
        trend_current = trend_model.predict(current_df)[0]
        # trend_base = trend_model.predict([[future_index]])[0]
        # trend_current = trend_model.predict([[current_index]])[0]
        trend_factor = trend_base / trend_current if abs(trend_current) > 1e-6 else 1.0

        x_input = window[-timesteps:, :-1]
        x_scaled = scaler_X.transform(x_input).reshape(1, timesteps, -1)

        base_prediction_scaled = model.predict(x_scaled, verbose=0)[0][0]
        base_prediction = scaler_y.inverse_transform([[base_prediction_scaled]])[0][0]

        adjusted = base_prediction * seasonal_factor * trend_factor
        # adjusted += np.random.normal(0, noise_std)
        adjusted = max(0, adjusted)  # Clamp to non-negative

        preds.append(adjusted)

        next_features = window[-1, :-1]
        next_row = np.concatenate([next_features, [adjusted]])
        window = np.vstack([window, next_row])

    return future_months, preds

set_seeds(42)

# Preprocess data
df_processed, encoders = preprocess_data(merged_df)
feature_cols = [c for c in df_processed.columns if c not in ['revenue', 'month', 'month_dt']]


def branch_name_to_id(branch_name):
    # Get corresponding branch_id from the branch_name_x column in merged_df
    return merged_df[merged_df['branch_name_x'] == branch_name]['branch_id'].values[0]

def get_prediction_of_branch_n(branch_name, n_months=6):
    model_dir = 'modelsEachBranch'
    timesteps = 6
    n_context = 120
    branch_id = branch_name_to_id(branch_name)
    df_branch = df_processed[df_processed['branch_id'] == branch_id].copy()
    df_branch = df_branch.sort_values(["month_dt"])
    future_months, future_preds = forecast_next_n_months(df_branch, model_dir, timesteps, n_months=n_months)

    months_hist = df_branch['month'].tolist()[-n_context:]
    actual_hist = df_branch['revenue'].tolist()[-n_context:]

    return months_hist, actual_hist, future_months, future_preds


