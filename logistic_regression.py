import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import json

class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate = 0.01, n_iterations = 1000, batch_size = 32, regularization = 0.01,
                 early_stopping_rounds = 5, random_state = None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.regularization = regularization
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims = True))
        return exp_z / np.sum(exp_z, axis=1, keepdims= True)
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]

        #Initialize weights and bias
        self.coef_ = np.zeros((self.n_classes_, self.n_features_))
        self.intercept_ = np.zeros(self.n_classes_)

        # Split data for early stopping
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size= 0.1, random_state= self.random_state)

        best_loss = np.inf
        no_improvement = 0
        rng = np.random.RandomState(self.random_state)

        for iteration in range(self.n_iterations):
            # Shuffle data
            indices = rng.permutation(len(X_train))
            X_train, y_train = X_train[indices], y_train[indices]

            for start in range(0, len(X_train), self.batch_size):
                end = min(start + self.batch_size, len(X_train))
                X_batch, y_batch = X_train[start:end], y_train[start:end]

                # Compute gradients and update parameters
                y_pred = self.softmax(np.dot(X_batch, self.coef_.T) + self.intercept_)
                for k in range(self.n_classes_):
                    y_true = (y_batch == k). astype(int)
                    error = y_pred[:, k] - y_true
                    self.coef_[k] -= self.learning_rate * (np.dot(X_batch.T, error) / len(X_batch) +
                    self.regularization * self.coef_[k])
                    self.intercept_[k] -= self.learning_rate * np.mean(error)

                # Compute validations loss for early stopping

                val_loss = log_loss(y_val, self.predict_proba(X_val))
                if val_loss < best_loss:
                    best_loss =  val_loss
                    no_improvement = 0
                else:
                    no_improvement += 1
                
                if no_improvement >= self.early_stopping_rounds:
                    break

            return self
        
    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.softmax(np.dot(X, self.coef_.T) + self.intercept_)
        
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.classes_[np.argmax(self.predict_proba(X), axis =1)]
        
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
        
    def save_model(self, file_path):
        check_is_fitted(self)
        model_params = {
                "coef_": self.coef_.tolist(),
                "intercept_": self.intercept_.tolist(),
                "classes_": self.classes_.tolist()
                }
            
        with open(file_path, 'w') as f:
            json.dump(model_params, f)

    def load_model(self, file_path):
        with open(file_path, 'r') as f:
            model_params = json.load(f)
        self.coef_ = np.array(model_params['coef_'])
        self.intercept_ = np.array(model_params['intercept_'])
        self.classes_ = np.array(model_params['classes_'])
        self.n_classes_ = len(self.classes_)
        self.n_features_ = self.coef_.shape[1]
