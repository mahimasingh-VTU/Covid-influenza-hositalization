from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

from helpers import print_separator


def polynomial_regression_hyper_parameter_tuning(X_train, y_train):
    model = make_pipeline(PolynomialFeatures(), Ridge())

    # Define the hyperparameters to tune
    param_grid = {
        'polynomialfeatures__degree': [1, 2, 3, 4, 5],  # Specify the degrees to be considered
        'ridge__alpha': [0.1, 1, 10]  # Specify the alpha values for Ridge regression
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)  # X is the input features, y is the target variable

    # Get the best hyperparameters
    best_degree = grid_search.best_params_['polynomialfeatures__degree']
    best_alpha = grid_search.best_params_['ridge__alpha']

    return best_degree, best_alpha


def xgboost_hyper_parameter_tuning(X_train, y_train):
    print_separator('XGBoost Hyper Parameter Tuning')
    # Define the parameters grid
    param_grid = {
        'learning_rate': [0.1, 0.01, 0.001],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300]
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror', seed=42),
                               param_grid=param_grid,
                               cv=5,
                               scoring='neg_mean_squared_error',
                               verbose=1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    return best_params
