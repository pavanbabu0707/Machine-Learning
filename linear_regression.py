import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3, learning_rate=0.01):

      """
      Linear Regression using Stochastic Gradient Descent
      :param batch_size: The number of samples per batch
      :param regularization: The Regularization parameter (L2 Regularization)
      :param max_epochs: The maximum number of epochs
      :param patience: The number of epochs to wait before early stopping if validation loss doesn't improve
      :param learning_rate:The learning rate for gradient descent
      """

      self.batch_size = batch_size
      self.regularization = regularization
      self.max_epochs = max_epochs
      self.patience = patience
      self.learning_rate = learning_rate
      self.weights = None
      self.bias = None
      self.loss_history = []

    def _initialize_parameters(self, n_features, n_outputs):
        self.weights = np.random.randn(n_features, n_outputs) * 0.01
        self.bias = np.zeros((1, n_outputs))

    def fit(self, X, y, validation_split = 0.1):
        X_train, X_val, y_train, y_val = train_test_split (X, y, test_size = validation_split, random_state = 42)

        n_samples, n_features = X_train.shape
        n_outputs = y_train.shape[1] if y_train.ndim > 1 else 1

        self._initialize_parameters(n_features, n_outputs)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.max_epochs):
              #Shuffle the training data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for i in range (0, n_samples, self.batch_size):
                X_batch = X_train_shuffled[i:i+self.batch_size]
                y_batch = y_train_shuffled[i:i+self.batch_size]

                y_pred = self.predict(X_batch)

                #Compute gradients
                dw = (1/self.batch_size) * np.dot(X_batch.T, (y_pred - y_batch)) + (self.regularization * self.weights)
                db = (1/self.batch_size) * np.sum(y_pred - y_batch, axis=0, keepdims=True)

                #Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            #Compute Validation loss
            val_loss = self.score(X_val, y_val)
            self.loss_history.append(val_loss)

            #Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights, best_bias = self.weights.copy(), self.bias.copy()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Set the model to the best parameters
        self.weights, self.bias = best_weights, best_bias


    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
          return mean_squared_error(y, self.predict(X))

    def save_model(self, file_path):
          model_params = {
              "weights": self.weights.tolist(),
              "bias": self.bias.tolist(),
              "hyperparameters": {
                  "batch_size": self.batch_size,
                  "regularization": self.regularization,
                  "max_epochs": self.max_epochs,
                  "patience": self.patience,
                  "learning_rate": self.learning_rate
                  }
          }
          with open(file_path, 'w') as f:
              json.dump(model_params, f)

    @classmethod
    def load_model(cls, file_path):
        with open(file_path, 'r') as f:
            model_params = json.load(f)

        model = cls(**model_params['hyperparameters'])
        model.weights = np.array(model_params['weights'])
        model.bias = np.array(model_params['bias'])
        return model

    def get_loss_history(self):
        return self.loss_history





