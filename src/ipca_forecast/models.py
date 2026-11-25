"""Model definitions: benchmarks (RW, AR) and scikit-learn models (LASSO, RF, factors, CSR)."""

import pandas as pd
import sklearn

def make_models(X_train, y_train, X_test, y_test, h):
    scaler = sklearn.preprocessing.StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    split = sklearn.model_selection.TimeSeriesSplit(n_splits=5, gap=h - 1)

    lassoCV = sklearn.linear_model.LassoCV(cv=split, max_iter=20000)

    lassoCV.fit(X_train_normalized, y_train)

    lassoCV_predictions = lassoCV.predict(X_test_normalized)
    mae_lassoCV = sklearn.metrics.mean_absolute_error(y_test, lassoCV_predictions)
    rmse_lassoCV = sklearn.metrics.root_mean_squared_error(y_test, lassoCV_predictions)

    return mae_lassoCV, rmse_lassoCV, lassoCV_predictions
