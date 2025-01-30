import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

# Function to load data, preprocess, train the model, and evaluate
def logistic_regression_iris():
    # Load the iris dataset and select features (sepal length/width)
    iris = load_iris()
    X = iris.data[:, :2]  # Sepal length and width
    y = iris.target

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features (fit on training data and apply to test data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the logistic regression model
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train_scaled, y_train)

    # Save model parameters to a file
    model.save_model("model2_parameters.json")

    # Test model performance on the test set
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Test set accuracy (sepal length/width): {accuracy:.2%}")

    # Additional performance metrics
    y_pred = model.predict(X_test_scaled)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Run the function
logistic_regression_iris()
