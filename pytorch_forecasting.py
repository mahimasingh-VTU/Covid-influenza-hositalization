import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from hyper_parameter_tuning_methods import xgboost_hyper_parameter_tuning, polynomial_regression_hyper_parameter_tuning
from pre_processing import drop_cols, clean_data_set, pca_analysis, rf_feat_selection_reg, clean_data_set_PTF
from regression_methods import perform_linear_regression, perform_xgboost, perform_polynomial_regression

df = pd.read_csv('./data/data-6Dec.csv')
target_name = 'total_patients_hospitalized_confirmed_influenza_and_covid'
initial_null_value_target = df[target_name].isnull().sum()

df.dropna(subset=[target_name], inplace=True)
df = drop_cols(df)
target = df[target_name]
X = df.drop([target_name], axis=1)
y = pd.DataFrame(target)

X_std, one_hot_encoded_columns, not_one_hot_encoded_columns = clean_data_set_PTF(X)
state_columns = [col for col in X_std.columns if col.startswith('state')]

X_std.drop(columns=state_columns, inplace=True)

# Sort DataFrame by the 'date' column in ascending order
df.sort_values(by='date', inplace=True)

# Check for and handle duplicates
duplicates = df[df.duplicated(subset=['date'], keep=False)]

if not duplicates.empty:
    # If there are duplicates, you may choose to aggregate or average values for those dates
    # For example, assuming 'target_name' is the column you want to aggregate, you can use mean():
    df = df.groupby('date').mean().reset_index()

# Set the 'date' column as the index
df.set_index('date', inplace=True)

# Feature Selection
selected_features = ['critical_staffing_shortage_today_yes', 'critical_staffing_shortage_today_no',
                     'hospital_onset_covid', 'inpatient_beds', 'inpatient_beds_used',
                     'previous_day_admission_adult_covid_confirmed',
                     'previous_day_admission_adult_covid_suspected',
                     'icu_patients_confirmed_influenza',
                     'previous_day_admission_influenza_confirmed',
                     'total_patients_hospitalized_confirmed_influenza_and_covid']

# selected_features =['critical_staffing_shortage_today_no', 'previous_day_admission_adult_covid_confirmed_70-79',
#  'critical_staffing_shortage_today_yes', 'critical_staffing_shortage_anticipated_within_week_no',
#  'previous_day_admission_adult_covid_confirmed', 'previous_day_deaths_covid_and_influenza', 'hospital_onset_covid',
#  'previous_day_admission_adult_covid_confirmed_80+', 'critical_staffing_shortage_anticipated_within_week_yes',
#  'previous_day_admission_adult_covid_confirmed_60-69', 'icu_patients_confirmed_influenza', 'deaths_covid',
#  'previous_day_admission_pediatric_covid_suspected', 'adult_icu_bed_covid_utilization_numerator',
#  'inpatient_bed_covid_utilization_numerator']

df_selected = df[selected_features]
#
# # Handle missing values (if any)
# df_selected.fillna(0, inplace=True)

# Train-Test Split
train_size = int(len(df_selected) * 0.8)
train_data, test_data = df_selected.iloc[:train_size], df_selected.iloc[train_size:]

# Normalization
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)


# Prepare PyTorch DataLoader
def create_dataset(data, seq_length):
    sequences = []
    target = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        target.append(label)
    return torch.tensor(sequences), torch.tensor(target)


seq_length = 7  # You can adjust this based on the context of your data
X_train, y_train = create_dataset(train_data_scaled, seq_length)
X_test, y_test = create_dataset(test_data_scaled, seq_length)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# PyTorch Model
# class SimpleLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])
#         return out

# input_size = len(selected_features)
# hidden_size = 64
# output_size = 1

import torch.nn.functional as F
class ImprovedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(ImprovedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Apply ReLU activation to the output of LSTM
        out = F.relu(out)
        out = self.fc(out[:, -1, :])
        return out


# Adjust hyperparameters
input_size = len(selected_features)
hidden_size = 128  # Increase hidden size
output_size = 1
num_layers = 3  # Increase the number of layers
dropout = 0.3

# model = SimpleLSTM(input_size, hidden_size, output_size)
model = ImprovedLSTM(input_size, hidden_size, output_size, num_layers,dropout)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training
num_epochs = 1000

for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X.float())
        loss = criterion(output, batch_y.float())
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Evaluation
model.eval()
predictions = []
with torch.no_grad():
    for X_test_batch, _ in test_loader:
        prediction = model(X_test_batch.float())
        # predictions.append(prediction.item())
        predictions.append(prediction.squeeze().numpy())

# Inverse transform using StandardScaler
predictions = np.array(predictions).squeeze()  # Remove singleton dimensions

# Reshape the predictions to match the shape of the training data
predictions = predictions.reshape(-1, 1)  # Assuming there's one target variable

# Create a new scaler for the target variable
target_scaler = StandardScaler()
target_scaler.fit_transform(y_train.numpy().reshape(-1, 1))  # Fit on the training target variable

# Inverse transform using the target variable's scaler
predictions = target_scaler.inverse_transform(predictions)

# Adjust the shape of y_test to match predictions
y_test_adjusted = y_test[:, 0].reshape(-1, 1)  # Assuming y_test is of shape (263, 10)

# Evaluate the model
mse = mean_squared_error(y_test_adjusted.numpy(), predictions)
print(f'Mean Squared Error: {mse}')
mae = mean_absolute_error(y_test_adjusted.numpy(), predictions)
rmse = np.sqrt(mean_squared_error(y_test_adjusted.numpy(), predictions))
r2 = r2_score(y_test_adjusted.numpy(), predictions)

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared (R2): {r2}')

# Plot the results
plt.figure(figsize=(15, 6))
plt.plot(y_test_adjusted, label='True Values')
plt.plot(predictions, label='Predictions')
plt.title('LSTM Model Prediction vs True Values')
plt.xlabel('Index')
plt.ylabel('Total Patients')
plt.legend()
plt.show()
