import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate example data
np.random.seed(0)
X1 = np.random.rand(100, 1) * 10   # First independent variable
X2 = np.random.rand(100, 1) * 5    # Second independent variable

y = 2.5 * X1 + 3.0 * X2 + np.random.randn(100, 1) * 2  # Target variable

# Stack X1 and X2 to form a 2D feature matrix
X = np.hstack((X1, X2))  

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the results
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")

# Visualization (for X1 vs. y, since X2 is also influencing y)
plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual')
plt.scatter(X_test[:, 0], y_pred, color='red', label='Predicted', alpha=0.7)
plt.xlabel("X1 (Feature 1)")
plt.ylabel("y (Target)")
plt.legend()
plt.show()
