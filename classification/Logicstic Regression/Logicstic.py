import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Trains the logistic regression model.

        Args:
            X (np.ndarray): Training features (n_samples, n_features).
            y (np.ndarray): Target labels (n_samples,).
        """
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):
            # Calculate linear combination
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid to get predicted probabilities
            y_predicted = self._sigmoid(linear_model)

            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        """
        Predicts probabilities for given features.

        Args:
            X (np.ndarray): Features to predict (n_samples, n_features).

        Returns:
            np.ndarray: Predicted probabilities (n_samples,).
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        Predicts class labels (0 or 1) based on a threshold.

        Args:
            X (np.ndarray): Features to predict (n_samples, n_features).
            threshold (float): Threshold for classifying probabilities.

        Returns:
            np.ndarray: Predicted class labels (n_samples,).
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

# --- Example Usage ---
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score

    # Generate some synthetic data for binary classification
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, random_state=42)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the Logistic Regression model
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nFirst 5 predicted probabilities:", y_pred_proba[:5])
    print("First 5 predicted labels:", y_pred[:5])
    print("First 5 true labels:", y_test[:5])

    # You can also compare with scikit-learn's LogisticRegression
    from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
    skl_model = SklearnLogisticRegression(solver='lbfgs', random_state=42)
    skl_model.fit(X_train, y_train)
    skl_y_pred = skl_model.predict(X_test)
    skl_accuracy = accuracy_score(y_test, skl_y_pred)
    print(f"\nScikit-learn Accuracy: {skl_accuracy:.4f}")
