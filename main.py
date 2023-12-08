import pandas as pd
from sklearn.model_selection import train_test_split

from pre_processing import drop_col_with_coverage, clean_data_set, pca_analysis, rf_feat_selection_reg
from regression_methods import perform_linear_regression, perform_svr, perform_xgboost

df = pd.read_csv('./data/data-6Dec.csv')
target_name = 'total_patients_hospitalized_confirmed_influenza_and_covid'
initial_null_value_target = df[target_name].isnull().sum()

df.dropna(subset=[target_name], inplace=True)
df = drop_col_with_coverage(df)
target = df[target_name]
X = df.drop([target_name], axis=1)
y = pd.DataFrame(target)

X_std, one_hot_encoded_columns, not_one_hot_encoded_columns = clean_data_set(X)

state_columns = [col for col in X_std.columns if col.startswith('state')]

# null_values = X_std[state_columns].isnull().any()
# for col in state_columns:
#     X_std[col].fillna(False, inplace=True)

X_std.drop(columns=state_columns, inplace=True)
print(X_std.shape)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, shuffle=True, random_state=1)

# selected_features = rf_feat_selection_reg(X_train, y_train.values.ravel(), 0.01, True, True)

selected_features = ['critical_staffing_shortage_today_no', 'previous_day_admission_adult_covid_confirmed_70-79', 'critical_staffing_shortage_anticipated_within_week_no', 'previous_day_deaths_covid_and_influenza', 'previous_day_admission_adult_covid_confirmed', 'critical_staffing_shortage_today_yes', 'previous_day_admission_adult_covid_confirmed_60-69', 'hospital_onset_covid', 'previous_day_admission_adult_covid_confirmed_80+', 'deaths_covid', 'critical_staffing_shortage_anticipated_within_week_yes']

X_train = X_train[selected_features]
X_test = X_test[selected_features]

perform_linear_regression(X_train, y_train, X_test, y_test)

perform_xgboost(X_train, y_train, X_test, y_test)