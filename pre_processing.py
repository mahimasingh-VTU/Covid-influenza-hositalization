import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

from helpers import print_separator


def clean_data_set(df):
    X_date = splitting_date_into_blocks(df)
    X_removed_null = drop_col_with_10per_null(X_date)
    X_one, one_hot_encoded_columns, not_one_hot_encoded_columns = one_hot_encoding(X_removed_null)

    # Filling the null values with mean
    X_one.fillna(X_one.mean(), inplace=True)

    X_std = standardization_data(X_one, one_hot_encoded_columns, not_one_hot_encoded_columns)

    return X_std, one_hot_encoded_columns, not_one_hot_encoded_columns


def splitting_date_into_blocks(X):
    X['day'] = pd.to_datetime(X['date']).dt.day
    X['month'] = pd.to_datetime(X['date']).dt.month
    X['year'] = pd.to_datetime(X['date']).dt.year
    # Since we already have day, month and year, we can drop the initial invoice_date now
    X = X.drop(['date'], axis=1)

    return X


def drop_col_with_coverage(df):
    columns_to_drop = [col for col in df.columns if '_coverage' in col]
    columns_to_drop.append('geocoded_state')
    columns_to_drop.append('total_patients_hospitalized_confirmed_influenza')
    # columns_to_drop.append('total_patients_hospitalized_confirmed_influenza_and_covid')
    df.drop(columns=columns_to_drop, inplace=True)

    return df


def drop_col_with_10per_null(X):
    col_10per_null = []
    total_rows = X.shape[0]

    percen_to_drop = 10

    for col in X.columns:
        if X[col].isnull().sum() > (total_rows / percen_to_drop):
            col_10per_null.append(col)

    X.drop(columns=col_10per_null, inplace=True)

    return X


def one_hot_encoding(df):
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']

    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    not_one_hot_encoded_columns = [col for col in df.columns if col not in categorical_cols]

    one_hot_encoded_columns = [col for col in df_encoded.columns if col not in not_one_hot_encoded_columns]

    return df_encoded, one_hot_encoded_columns, not_one_hot_encoded_columns


def standardization_data(X, one_hot_encoded_columns, not_one_hot_encoded_columns):
    # print_separator('Scaling the Data set')
    # Scaling the columns that are not one hot encoded
    scaler_x = StandardScaler()
    # Drop the one hot encoded columns and then scale the dataset
    X_std = scaler_x.fit_transform(X.drop(one_hot_encoded_columns, axis=1))
    # Converting to dataframe
    X_std = pd.DataFrame(X_std, columns=not_one_hot_encoded_columns)
    # Adding the one hot encoded columns back to the dataset
    X_std[one_hot_encoded_columns] = X[one_hot_encoded_columns]

    return X_std


def rf_feat_selection_reg(X, y, threshold=0.1, show_importance=False, print_selected_features=False):
    print_separator('Doing Random Forest Feature Selection')

    random_state = 1007

    # Doing Random Forest Analysis to check for feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf.fit(X, y)
    # Get the feature importance
    importances = rf.feature_importances_

    # Sort the features by importance
    indices = importances.argsort()[::-1]
    selected_features = []

    # Getting the feature ranking based on threshold which is default to 0.1
    for f in range(X.shape[1]):
        if show_importance:
            print(f"{X.columns[indices[f]]}: {importances[indices[f]]:.2f}")
        if importances[indices[f]] >= threshold:
            selected_features.append(X.columns[indices[f]])

    # if show_importance:
    #     plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    #     plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    #     plt.xlabel('Relative Importance')
    #     plt.tight_layout()
    #     plt.title("Feature importance using Random Forest for Regression")
    #     plt.show()

    if print_selected_features:
        print_separator('Selected Features after random forest feature selection')
        print(selected_features)

    return selected_features


def pca_analysis(X):
    print_separator('Performing PCA analysis')
    pca = PCA()
    pca.fit_transform(X)
    ex_var_ratio = pca.explained_variance_ratio_
    print(np.round(np.cumsum(ex_var_ratio * 100), 2))

    plt.plot(np.arange(1, len(np.cumsum(ex_var_ratio)) + 1, 1),
             np.cumsum(ex_var_ratio))
    plt.xticks(np.arange(1, len(np.cumsum(ex_var_ratio)) + 1, 1))
    plt.grid()
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title('Elbow plot for PCA')
    plt.show()
