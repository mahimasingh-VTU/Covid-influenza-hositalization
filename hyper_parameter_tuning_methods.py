from sklearn.model_selection import GridSearchCV
import xgboost as xgb

from helpers import print_separator


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
