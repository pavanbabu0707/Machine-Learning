import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# Funcion to load data, train logistic regression model and visualize decision boundaries

def logistic_regression_with_feature_subset():
    # Load the dataset and select sepal length/width features
    data = load_iris()
    X = data.data
    y = (data.target == 0).astype(int) # Conver to binary classification (class 1 vs others) 

    # Split the datasets into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

    # Initialize and train logistic regression model
    model = LogisticRegression(learning_rate = 0.01, n_iterations = 1000)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Print accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy on test set using all features: {accuracy * 100:.2f}%")

    # Visulaize decision boundaries
    if X_train.shape[1] == 2:
     plt.figure(figsize=(8, 6))
     plot_decision_regions(X_test, y_test, clf=model)
     plt.title("Logistic Regresion Decision Regions: all features")
     plt.xlabel("Feature 1")
     plt.ylabel("Feature 2")
     plt.show()

# Run the logistic regression process
logistic_regression_with_feature_subset()

