from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression # Import your LinearRegression

# Load the iris dataset
iris = load_iris()
X = iris.data[:, [2, 3]] # Petal length and petal width for input features
y = iris.data[:, 1] # Sepal width for output

# Standardize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape y to be a column vector
y = y.reshape(-1, 1)

# Split the data intp Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size= 0.2, random_state= 42)

#Create and train the Linear Regression model without regularization
model_no_reg = LinearRegression(batch_size = 32, regularization=0, max_epochs = 100, patience =3)
model_no_reg.fit(X_train, y_train)
loss_no_reg = model_no_reg.get_loss_history()

#   Save the model parameters for the model without regularization
model_no_reg.save_model('model_no_regularization.json')

# Plot the loss for the model withput regularization

plt.plot(loss_no_reg)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model 2 Training Loss - No Regularization')
plt.savefig('Model 2 loss_no_regularization.png')
plt.show()

# Create and train the Linearregression model with regularization
model_with_reg = LinearRegression(batch_size=32, regularization=0.001, max_epochs=100, patience =3)
model_with_reg.fit(X_train, y_train)
loss_with_reg = model_with_reg.get_loss_history()

# Save the model parameters for the model with regularization
model_with_reg.save_model('model_with_regularization.json')

# Plot the loss for the model with regularization
plt.plot(loss_with_reg)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model 2 Training Loss - With Regularization')
plt.savefig('Model 2 loss_with_regularization.png')
plt.show()
