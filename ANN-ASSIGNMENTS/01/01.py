import numpy as np

class Perceptron:
    def __init__(self, initial_weights, learning_rate=0.1, epochs=100):
        self.weights = np.array(initial_weights, dtype=float)
        self.learning_rate = learning_rate
        self.epochs = epochs


    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                # Add bias term to the input
                xi_augmented = np.insert(xi, 0, 1)
                # Compute prediction
                activation = np.dot(xi_augmented, self.weights)
                prediction = np.sign(activation)
                # Update weights if prediction is incorrect
                if prediction != target:
                    self.weights += self.learning_rate * (target - prediction) * xi_augmented
        return self

    def predict(self, X):
        predictions = []
        for xi in X:
            # Add bias term to the input
            xi_augmented = np.insert(xi, 0, 1)
            activation = np.dot(xi_augmented, self.weights)
            predictions.append(np.sign(activation))
        return np.array(predictions)

# Example usage
if __name__ == "__main__":
    # Training data (3 features, no bias term)
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [-1, -2, -3],
        [-2, -3, -4]
    ])
    y = np.array([1, 1, -1, -1])

    # Initialize perceptron with 4 weights (including bias)
    perceptron = Perceptron(
        initial_weights=[0, 0, 0, 0],
        learning_rate=0.1,
        epochs=100
    )

    # Train the perceptron
    perceptron.fit(X, y)

    # Test predictions
    test_samples = np.array([
        [2, 3, 4],
        [-3, -2, -1]
    ])
    predictions = perceptron.predict(test_samples)

    print("Learned weights:", perceptron.weights)
    print("Predictions for test samples:", predictions)

    