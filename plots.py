import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.express as px

from hyper_parameter_tuning_methods import xgboost_hyper_parameter_tuning, polynomial_regression_hyper_parameter_tuning
from pre_processing import drop_cols, clean_data_set, pca_analysis, rf_feat_selection_reg
from regression_methods import perform_linear_regression, perform_xgboost, perform_polynomial_regression

df = pd.read_csv('./data/data-6Dec.csv')
target_name = 'total_patients_hospitalized_confirmed_influenza_and_covid'
initial_null_value_target = df[target_name].isnull().sum()


df.dropna(subset=[target_name], inplace=True)
df = drop_cols(df)
target = df[target_name]
X = df.drop([target_name], axis=1)
y = pd.DataFrame(target)

X_std, one_hot_encoded_columns, not_one_hot_encoded_columns = clean_data_set(X)
df.set_index('date', inplace=True)
# Dropped columns with state because they came as not important in the feature selection
# But where taking a lot of time to run the feature selection
# Needs to be checked again when submitting to show the work
state_columns = [col for col in X_std.columns if col.startswith('state')]

# null_values = X_std[state_columns].isnull().any()
# for col in state_columns:
#     X_std[col].fillna(False, inplace=True)

X_std.drop(columns=state_columns, inplace=True)
print(X_std.shape)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, shuffle=True, random_state=1)

selected_features = rf_feat_selection_reg(X_train, y_train.values.ravel(), 0.01, True, True)
print("selected_features",)
#selected_features = ['critical_staffing_shortage_today_no', 'previous_day_admission_adult_covid_confirmed_70-79', 'critical_staffing_shortage_anticipated_within_week_no', 'previous_day_deaths_covid_and_influenza', 'previous_day_admission_adult_covid_confirmed', 'critical_staffing_shortage_today_yes', 'previous_day_admission_adult_covid_confirmed_60-69', 'hospital_onset_covid', 'previous_day_admission_adult_covid_confirmed_80+', 'deaths_covid', 'critical_staffing_shortage_anticipated_within_week_yes']


# Scatter plot of specific variables against the target variable
sns.scatterplot(x='inpatient_beds_used_covid', y='deaths_covid', data=df)
plt.title('Scatter Plot: Inpatient Beds Used (COVID) vs. Deaths (COVID)')
plt.show()


plt.figure(figsize=(12, 8))
# Bar plot of categorical variables
sns.barplot(x='state', y='deaths_covid', data=df)
plt.title('Bar Plot: Deaths (COVID) by State')
plt.xticks(rotation=45, ha="right")
plt.show()

# Line plot for time series data
df_time_series = df.reset_index()
sns.lineplot(x='index', y='deaths_covid', data=df_time_series)
plt.title('Time Series Line Plot: Deaths (COVID) over Time')
plt.xlabel('Date')
plt.ylabel('Deaths (COVID)')
plt.show()


# Extract relevant columns for mobility patterns (adjust as needed)
mobility_columns = ['hospital_onset_covid', 'inpatient_beds_utilization', 'inpatient_beds_utilization_numerator', 'inpatient_beds_utilization_denominator']
mobility_df = df[mobility_columns]

# Plot mobility patterns over time
plt.figure(figsize=(12, 8))
for column in mobility_df.columns:
    sns.lineplot(data=mobility_df[column], label=column)

plt.title('Mobility Patterns Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Extract relevant columns for time series plots (adjust as needed)
time_series_columns = ['previous_day_admission_adult_covid_confirmed', 'previous_day_admission_pediatric_covid_confirmed']

# Plot time series of hospitalization rates over time
fig = px.line(df, x=df.index, y=time_series_columns, title='Hospitalization Rates Over Time')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Hospitalization Rates')
fig.show()

# Create a new dataframe with selected features
selected_features_df = X_train[selected_features]

# Concatenate the target variable to the selected features dataframe
correlation_df = pd.concat([selected_features_df, y_train], axis=1)

# Calculate and plot the correlation matrix
correlation_matrix = correlation_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

# Plot the utilization of therapeutic supplies
therapeutic_columns = [
    'on_hand_supply_therapeutic_a_casirivimab_imdevimab_courses',
    'on_hand_supply_therapeutic_b_bamlanivimab_courses',
    'on_hand_supply_therapeutic_c_bamlanivimab_etesevimab_courses',
    'previous_week_therapeutic_a_casirivimab_imdevimab_courses_used',
    'previous_week_therapeutic_b_bamlanivimab_courses_used',
    'previous_week_therapeutic_c_bamlanivimab_etesevimab_courses_used'
]

therapeutic_df = df[therapeutic_columns]
plt.figure(figsize=(12, 8))
sns.lineplot(data=therapeutic_df, markers=True)
plt.title('Utilization of Therapeutic Supplies Over Time')
plt.xlabel('Date')
plt.ylabel('Supply Quantity')
plt.legend(therapeutic_columns)
plt.show()

# Explore the occupancy rates of pediatric and adult ICU beds
icu_columns = [
    'staffed_adult_icu_bed_occupancy',
    'staffed_icu_pediatric_patients_confirmed_covid',
    'staffed_pediatric_icu_bed_occupancy'
]

icu_df = df[icu_columns]
plt.figure(figsize=(12, 8))
sns.lineplot(data=icu_df, markers=True)
plt.title('ICU Bed Occupancy Over Time')
plt.xlabel('Date')
plt.ylabel('Occupancy Rate')
plt.legend(icu_columns)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
df = pd.read_csv('./data/data-6Dec.csv')

# Age Group Analysis
age_group_columns = [
    'previous_day_admission_adult_covid_confirmed_18-19',
    'previous_day_admission_adult_covid_confirmed_20-29',
    'previous_day_admission_adult_covid_confirmed_30-39',
    'previous_day_admission_adult_covid_confirmed_40-49',
    'previous_day_admission_adult_covid_confirmed_50-59',
    'previous_day_admission_adult_covid_confirmed_60-69',
    'previous_day_admission_adult_covid_confirmed_70-79',
    'previous_day_admission_adult_covid_confirmed_80+',
]

age_group_df = df[age_group_columns]
plt.figure(figsize=(12, 8))
sns.lineplot(data=age_group_df, markers=True)
plt.title('COVID-19 Cases and Hospitalizations by Age Group Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend(age_group_columns)
plt.show()

# Pediatric Analysis
pediatric_columns = [
    'previous_day_admission_pediatric_covid_confirmed_0_4',
    'previous_day_admission_pediatric_covid_confirmed_5_11',
    'previous_day_admission_pediatric_covid_confirmed_12_17',
]

pediatric_df = df[pediatric_columns]
plt.figure(figsize=(12, 8))
sns.lineplot(data=pediatric_df, markers=True)
plt.title('COVID-19 Cases and Hospitalizations in Pediatric Populations Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend(pediatric_columns)
plt.show()

# Comparison Across States
state_columns = ['state', 'deaths_covid', 'total_patients_hospitalized_confirmed_influenza_and_covid']
state_df = df[state_columns].groupby('state').mean().reset_index()

plt.figure(figsize=(12, 8))
sns.barplot(x='state', y='total_patients_hospitalized_confirmed_influenza_and_covid', data=state_df)
plt.title('Comparison of Hospitalizations Across States')
plt.xlabel('State')
plt.ylabel('Average Hospitalizations')
plt.show()
