import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your data from a CSV file or other source
data = pd.read_csv('data.csv')  # Replace 'your_dataset.csv' with your data file

# Split the data into features (X) and target (y)
X = data[['Year', 'TV', 'Newspaper']]
y = data['Turnover']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared (R2) Score: {r2}')

# Create Graphs
plt.figure(figsize=(12, 5))

# Graph 1: Turnover vs. Years
plt.subplot(131)
plt.scatter(data['Year'], data['Turnover'], color='blue', label='Actual Turnover')
plt.plot(data['Year'], model.predict(X), color='red', label='Predicted Turnover')
plt.title('Turnover vs. Years')
plt.xlabel('Year')
plt.ylabel('Turnover')
plt.legend()

# Graph 2: TV vs. Advertising Cost
plt.subplot(132)
plt.scatter(data['TV'], data['Turnover'], color='blue', label='Actual Turnover')
plt.title('TV vs. Advertising Cost')
plt.xlabel('TV Advertising Cost')
plt.ylabel('Turnover')
plt.legend()

# Graph 3: Newspaper vs. Advertising Cost
plt.subplot(133)
plt.scatter(data['Newspaper'], data['Turnover'], color='blue', label='Actual Turnover')
plt.title('Newspaper vs. Advertising Cost')
plt.xlabel('Newspaper Advertising Cost')
plt.ylabel('Turnover')
plt.legend()

plt.tight_layout()
plt.show()


# Check data types
print(X_train.dtypes)
print(y_train.dtypes)

# Remove non-numeric characters (e.g., commas) from 'y_train'
y_train = y_train.str.replace(',', '').astype(float)
