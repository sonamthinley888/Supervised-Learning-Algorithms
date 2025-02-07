import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic data
house_sizes = np.random.randint(120, 400, 100)  # House size in square meters
bedrooms = np.random.randint(2, 6, 100)         # Number of bedrooms

# Cost per square meter range ($1,940 to $3,950)
cost_per_sqm = np.random.uniform(1940, 3950, 100)

# Base house prices using real Canberra cost estimates
prices = house_sizes * cost_per_sqm  

# Adjust prices based on bedroom count using given cost ranges
for i in range(len(prices)):
    if bedrooms[i] == 2:
        prices[i] = np.clip(prices[i], 234000, 474000)
    elif bedrooms[i] == 3:
        prices[i] = np.clip(prices[i], 339500, 691250)
    elif bedrooms[i] == 4:
        prices[i] = np.clip(prices[i], 582000, 1185000)
    elif bedrooms[i] == 5:
        prices[i] = np.clip(prices[i], 585000, 1185000)

# Create DataFrame
df = pd.DataFrame({"Size (sqm)": house_sizes, "Bedrooms": bedrooms, "Price ($)": prices})

# Features (X) and target variable (y)
X = df[["Size (sqm)", "Bedrooms"]]
y = df["Price ($)"]

# Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Visualizing house size vs. price
plt.figure(figsize=(10, 6))
plt.scatter(df["Size (sqm)"], df["Price ($)"], color='blue', label="Actual Prices")
plt.xlabel("Size (sqm)")
plt.ylabel("Price ($)")
plt.title("House Size vs Price in Canberra")
plt.legend()
plt.show()

# Visualizing bedrooms vs. price
plt.figure(figsize=(8, 5))
plt.scatter(df["Bedrooms"], df["Price ($)"], color='green', label="Actual Prices")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Price ($)")
plt.title("Bedrooms vs Price in Canberra")
plt.legend()
plt.show()
