"""Preprocessing and feature engineering: lags, horizon-shifted targets, scaling."""
import pandas as pd

def train_preparation(df, h):
    df = df.copy()
    df["ipca_target_month"] = df["ipca_mom"].shift(-h)
    df = df.iloc[:-h]

    split_point = int(len(df) * 0.7)

    df_train = df.iloc[:split_point - h+1]
    df_test = df.iloc[split_point:]

    X_train = df_train.drop(columns="ipca_target_month")
    y_train = df_train["ipca_target_month"]

    X_test = df_test.drop(columns="ipca_target_month")
    y_test = df_test["ipca_target_month"]

    return X_train, y_train, X_test, y_test
