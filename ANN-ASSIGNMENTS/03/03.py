import numpy as np

# Initialize parameters
c = 0.1        # Learning rate
lambda_ = 1    # Activation function scaling factor (linear activation)

# Example training data: 3 input patterns with 3 features each
X = np.array([
    [1, 0, 1],    # Input 1
    [0, 1, 0],    # Input 2
    [1, 1, 0]     # Input 3
])

# Target outputs (one for each input pattern)
y = np.array([1, 0, 1])

# Initialize weights (3 features => 3 weights)
weights = np.zeros(3)

# Training loop
epochs = 10
for epoch in range(epochs):
    for i in range(len(X)):
        # Forward pass
        net_input = np.dot(weights, X[i])
        prediction = lambda_ * net_input  # With λ scaling
        
        # Calculate error
        error = y[i] - prediction
        
        # Delta rule weight update
        weights += c * lambda_ * error * X[i]
    
    print(f"Epoch {epoch+1} weights: {weights}")

print("\nFinal trained weights:", weights)