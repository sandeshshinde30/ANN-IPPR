import numpy as np

# Step function as activation
def step_function(net):
    return 1 if net >= 0 else 0

# Perceptron Training Function
def train_perceptron(X, T, learning_rate=1, epochs=10):
    weights = np.zeros(X.shape[1])
    bias = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        for i in range(len(X)):
            x = X[i]
            target = T[i]
            net = np.dot(weights, x) + bias
            output = step_function(net)
            error = target - output

            # Update rule
            weights += learning_rate * error * x
            bias += learning_rate * error

            print(f"  Input: {x}, Target: {target}, Output: {output}, Weights: {weights}, Bias: {bias}")
        print()
    return weights, bias

# Predict function
def predict(X, weights, bias):
    return [step_function(np.dot(weights, x) + bias) for x in X]

# Input vectors for AND/OR gates
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target outputs for AND
T_AND = np.array([0, 0, 0, 1])

# Target outputs for OR
T_OR = np.array([0, 1, 1, 1])

print("🔹 Training Perceptron for AND Gate")
w_and, b_and = train_perceptron(X, T_AND)
print("Predictions (AND):", predict(X, w_and, b_and), "\n")

print("🔹 Training Perceptron for OR Gate")
w_or, b_or = train_perceptron(X, T_OR)
print("Predictions (OR):", predict(X, w_or, b_or))
