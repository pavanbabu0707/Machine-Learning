from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from linear_regression import LinearRegression

# Load the iris dataset

def load_data():
    iris = load_iris()
    X = iris.data[:, [2,1]] # Petal length and Sepal width for input
    y = iris.data[:, 3] # Petal width for output
    return X, y

# Standardize features

def preprocess_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Train and evaluate model

def train_and_evaluate(X_train, X_test, y_train, y_test, regularization, model_params_file):
    model = LinearRegression(batch_size = 32, regularization=regularization, max_epochs = 100, patience =3)
    model.fit(X_train, y_train) # Train the model
    mse = model.score(X_test, y_test)
    return mse

# Main function

def main():
    # Load and preprocess the data
    X, y = load_data()
    X_scaled = preprocess_data(X)

    # Reshape y to be a column vector
    y = y.reshape(-1, 1)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size= 0.2, random_state= 42)

    # Train and evaluate without regularization
    mse_no_reg = train_and_evaluate(X_train, X_test, y_train, y_test, regularization= 0, model_params_file= 'model_parameters1.json')
    print("Mean Squared Error Without Regularization:", mse_no_reg)

    # Train and evaluate with regularization
    mse_with_reg = train_and_evaluate(X_train, X_test, y_train, y_test, regularization= 0.001, model_params_file='model_parameters1_regularization.json')
    print("Mean Squared Error with Regularization:", mse_with_reg)

if __name__ == "__main__":
    main()