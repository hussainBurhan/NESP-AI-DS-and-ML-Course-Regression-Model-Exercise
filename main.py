# Import necessary libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(1)

# Read the Boston housing dataset from a CSV file
boston = pd.read_csv('boston.csv')

# Separate features (x) and target variable (y)
x = boston.drop('MEDV', axis=1)
y = boston['MEDV']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Initialize and train a RandomForestRegressor model
Regression_Model = RandomForestRegressor().fit(x_train, y_train)

# Make predictions on the test set
predicted_y = Regression_Model.predict(x_test)

# Print the predicted values
print(f'predicted MEDV: {predicted_y}')

# Import mean_absolute_error function and calculate MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predicted_y)

# Print the Mean Absolute Error (MAE)
print(f'means absolute error: {mae}')
