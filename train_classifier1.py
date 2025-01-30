import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# Funcion to load data, train logistic regression model and visualize decision boundaries

def logistic_regression_petal():
    # Load the dataset and select petal length/width features
    data = load_iris()
    X = data.data[:, 2:4]
    y = (data.target == 1).astype(int) # Conver to binary classification (class 1 vs others) 

    # Split the datasets into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

    # Initialize and train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Visulaize decision boundaries
    plt.figure(figsize=(8, 6))
    plot_decision_regions(X_test, y_test, clf=model)
    plt.title("Logistic Regresion Decision Regions: Petal Length/Width")
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.show()

# Run the logistic regression process
logistic_regression_petal()
