import numpy as np
from sklearn.linear_model import LinearRegression

# Example angles in degrees
angle_x = np.array([10, 20, 30, 40, 350])
angle_y = np.array([15, 25, 35, 45, 5])

# Convert angles to radians
angle_x_rad = np.radians(angle_x)
angle_y_rad = np.radians(angle_y)

# Transform angles into sine and cosine components
X = np.column_stack((np.sin(angle_x_rad), np.cos(angle_x_rad)))
y = np.sin(angle_y_rad)  # Example: predicting sine of angle_y

# Perform regression
model = LinearRegression()
model.fit(X, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
