from math import sqrt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import xgboost as xgb


def get_regression_scores(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)

    # Calculate R-squared
    r_squared = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared: {r_squared}")


def perform_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    get_regression_scores(y_test, y_pred)


def perform_svr(X_train, y_train, X_test, y_test):
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the accuracy (e.g., using mean squared error)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)

    # Calculate R-squared
    r_squared = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared: {r_squared}")


def perform_xgboost(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(objective='reg:squarederror', seed=42)  # Use 'reg:squarederror' for regression

    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)

    get_regression_scores(y_test, y_pred)
