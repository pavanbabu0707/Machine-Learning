from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression # Import your LinearRegression

# Load the iris dataset
iris = load_iris()
X = iris.data[:, [0, 1]] # Sepal length and sepal width for input
y = iris.data[:, [2, 3]] # Petal length and petal width for output

# Standardize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data intp Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size= 0.2, random_state= 42)

#Create and train the Linear Regression model
model_multi_reg = LinearRegression(batch_size = 32, regularization=0, max_epochs = 100, patience =3)
model_multi_reg.fit(X_train, y_train)
loss_multi_reg = model_multi_reg.get_loss_history()

#   Save the model parameters for the model with all features
model_multi_reg.save_model('model_parameters_multiple_output.json')

# Plot the loss for the model with regularization
plt.plot(loss_multi_reg)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Multi Reg Model Training Loss')
plt.savefig('Model Multi Reg loss.png')
plt.show()
