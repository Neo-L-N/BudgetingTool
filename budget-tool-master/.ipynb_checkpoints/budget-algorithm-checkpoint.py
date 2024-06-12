import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load the dataset
data_path = '/home/samurai/budget-tool-master/financial_data_adjusted.csv'
df = pd.read_csv(data_path, index_col=0, parse_dates=True)  # Ensure the date is the index

# Split the data into features and labels
X = df[['Gas', 'Food', 'Entertainment', 'Total Expenses', 'Monthly Income']].values
y = df['Savings'].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)

# Evaluate the model
model.evaluate(X_test_scaled, y_test)

# Plotting the progression of Monthly Savings
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Savings'], marker='o', linestyle='-', color='blue')
plt.title('Progression of Monthly Savings')
plt.xlabel('Month')
plt.ylabel('Savings ($)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the figure
plot_path = '/home/samurai/budget-tool-master/monthly_savings_progression.png'
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

# Optionally, attempt to display the plot (may not work in non-interactive environments)
plt.show()

