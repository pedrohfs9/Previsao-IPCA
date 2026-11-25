"""Evaluation: rolling-window backtest, RMSE/MAE tables and forecast combination."""

import pandas as pd
import sklearn

from ipca_forecast.features import train_preparation
from ipca_forecast.models import make_models, run_csr


def compare_with_focus(df, horizons):
    rows = []

    for h in horizons:
        X_train, y_train, X_test, y_test = train_preparation(df, h)

        mae_lasso, rmse_lasso, _ = make_models(X_train, y_train, X_test, y_test, h)
        mae_csr, rmse_csr, _ = run_csr(X_train, y_train, X_test, y_test)

        focus_predictions = X_test[f"focus_med_t{h + 1}"]
        mae_focus = sklearn.metrics.mean_absolute_error(y_test, focus_predictions)
        rmse_focus = sklearn.metrics.root_mean_squared_error(y_test, focus_predictions)

        rows.append({
            "h": h,
            "mae_focus": mae_focus,
            "mae_lasso": mae_lasso,
            "mae_csr": mae_csr,
            "rmse_focus": rmse_focus,
            "rmse_lasso": rmse_lasso,
            "rmse_csr": rmse_csr,
            "ratio_lasso": rmse_lasso / rmse_focus,
            "ratio_csr": rmse_csr / rmse_focus,
        })

    return pd.DataFrame(rows).set_index("h").round(3)
