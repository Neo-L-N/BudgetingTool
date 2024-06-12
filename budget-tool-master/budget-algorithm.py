import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# Load the dataset
data_path = 'home/your/data/input'
df = pd.read_csv(data_path, index_col=0, parse_dates=True)  # Ensure the date is the index

# Split the data into features and labels
X = df[['Gas', 'Food', 'Entertainment', 'Total Expenses', 'Monthly Income']].values
y = df['Savings'].values.reshape(-1, 1)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features and the target
feature_scaler = StandardScaler()
target_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)
y_train_scaled = target_scaler.fit_transform(y_train)
y_test_scaled = target_scaler.transform(y_test)

# Define the DNN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, y_train_scaled, epochs=100, validation_split=0.2)

# Evaluate the model on scaled test data
model.evaluate(X_test_scaled, y_test_scaled)

# Calculate the average of the last 6 months for future predictions
last_six_months_avg = df[-6:].mean(axis=0)
future_features = np.tile(last_six_months_avg[['Gas', 'Food', 'Entertainment', 'Total Expenses', 'Monthly Income']].values, (12, 1))

# Scale the future features
future_features_scaled = feature_scaler.transform(future_features)

# Predict future savings scaled
future_savings_scaled = model.predict(future_features_scaled).flatten()
future_savings = target_scaler.inverse_transform(future_savings_scaled.reshape(-1, 1)).flatten()

# Calculate accumulated savings
all_savings = np.concatenate((df['Savings'].values, future_savings))
cumulative_savings = np.cumsum(all_savings)

# Calculate the average prediction for the next year
average_future_savings = np.mean(future_savings)
print(f"Average predicted savings for the next year: ${average_future_savings:.2f}")

# Plot historical and predicted cumulative savings
plt.figure(figsize=(14, 7))
months = pd.date_range(start=df.index[0], periods=len(cumulative_savings), freq='M')
plt.plot(months[:len(df)], cumulative_savings[:len(df)], label='Historical Accumulated Savings')
plt.plot(months[len(df):], cumulative_savings[len(df):], label='Predicted Accumulated Savings', linestyle='--')
plt.title('Accumulated Savings Over Time')
plt.xlabel('Month')
plt.ylabel('Cumulative Savings ($)')
plt.legend()
plt.grid(True)
plt.savefig('/home/your/savings/.png')
plt.show()

# Plotting a breakdown of all costs per month
plt.figure(figsize=(14, 8))
plt.plot(df.index, df['Gas'], label='Gas', marker='o', linestyle='-')
plt.plot(df.index, df['Food'], label='Food', marker='o', linestyle='-')
plt.plot(df.index, df['Entertainment'], label='Entertainment', marker='o', linestyle='-')
plt.plot(df.index, df['Total Expenses'], label='Total Expenses', marker='o', linestyle='-', linewidth=2)
plt.plot(df.index, df['Monthly Income'], label='Monthly Income', marker='o', linestyle='-', linewidth=2, alpha=0.7)
plt.title('Monthly Breakdown of Expenses and Income')
plt.xlabel('Month')
plt.ylabel('Amount ($)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('home/your/data/output')
plt.show()
