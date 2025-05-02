
## 🧠 Artificial Neural Networks (ANN)

### 🔍 Key Topics to Focus On

#### 1. Perceptron Learning Rule
- Single-layer perceptron architecture
- Binary classification problems
- Weight update rule: `w(new) = w(old) + η * (t - y) * x`
- Convergence properties and limitations

#### 2. Hebbian Learning Rule
- Biological inspiration and Hebb's postulate
- Weight update rule: `w(new) = w(old) + η * x * y`
- Unsupervised learning characteristics
- Applications in pattern association

#### 3. Delta Learning Rule
- Error-correction learning
- Weight update rule: `w(new) = w(old) + η * (t - y) * x`
- Learning rate (λ) and convergence factor (c)
- Relationship to gradient descent

#### 4. Logical Function Implementation
- AND, OR function realization using perceptrons
- Truth table representation
- Network architecture and weight configuration
- Training and testing procedures

#### 5. Deep Learning Applications
- Handwritten digit classification using TensorFlow
- Speech recognition systems
- Expert systems for disease prediction
- Stock market price prediction

### 💻 ANN Assignments

#### Assignment 1: Perceptron Learning Rule
**Problem Statement:**
- Train a network with three different inputs and initial weight vector using Perceptron learning rule in Python

**Sample Implementation:**
```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # Initialize weights with random values
        self.weights = np.random.rand(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
    
    def predict(self, inputs):
        # Add bias input
        inputs_with_bias = np.insert(inputs, 0, 1)
        # Calculate weighted sum
        summation = np.dot(inputs_with_bias, self.weights)
        # Apply activation function (step function)
        return 1 if summation > 0 else 0
    
    def train(self, training_inputs, labels, epochs=100):
        errors = []
        
        for epoch in range(epochs):
            total_error = 0
            
            for inputs, label in zip(training_inputs, labels):
                # Add bias input
                inputs_with_bias = np.insert(inputs, 0, 1)
                
                # Make prediction
                prediction = self.predict(inputs)
                
                # Calculate error
                error = label - prediction
                total_error += abs(error)
                
                # Update weights
                self.weights += self.learning_rate * error * inputs_with_bias
            
            errors.append(total_error)
            
            # Stop if convergence is reached
            if total_error == 0:
                print(f"Converged after {epoch+1} epochs")
                break
                
        return errors

# Example usage
# Define training data (3 inputs)
X = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# Define labels (e.g., for OR function)
y = np.array([0, 1, 1, 1, 1, 1, 1, 1])

# Create and train perceptron
perceptron = Perceptron(input_size=3)
print("Initial weights:", perceptron.weights)
errors = perceptron.train(X, y)
print("Final weights:", perceptron.weights)

# Plot error over epochs
plt.plot(errors)
plt.xlabel('Epochs')
plt.ylabel('Total Error')
plt.title('Perceptron Learning Convergence')
plt.grid(True)
plt.show()

# Test the trained perceptron
print("\nTesting the trained perceptron:")
for inputs, label in zip(X, y):
    prediction = perceptron.predict(inputs)
    print(f"Inputs: {inputs}, Target: {label}, Prediction: {prediction}")
```

#### Assignment 2: Hebbian Learning Rule
**Problem Statement:**
- Train a network with three different inputs and initial weight vector using Hebbian learning rule in Python

**Sample Implementation:**
```python
import numpy as np
import matplotlib.pyplot as plt

class HebbianNetwork:
    def __init__(self, input_size):
        # Initialize weights with zeros
        self.weights = np.zeros(input_size)
    
    def train(self, inputs, learning_rate=1.0, epochs=1):
        weight_history = [self.weights.copy()]
        
        for epoch in range(epochs):
            for x in inputs:
                # Calculate output
                y = np.dot(x, self.weights)
                # Apply Hebbian learning rule
                delta_w = learning_rate * np.outer(x, y).diagonal()
                self.weights += delta_w
                weight_history.append(self.weights.copy())
        
        return weight_history

# Example usage
# Define input patterns (3 inputs)
patterns = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Initialize and train network
network = HebbianNetwork(input_size=3)
print("Initial weights:", network.weights)
weight_history = network.train(patterns)
print("Final weights:", network.weights)

# Plot weight changes over training
plt.figure(figsize=(10, 6))
weight_history = np.array(weight_history)
for i in range(network.weights.size):
    plt.plot(weight_history[:, i], label=f'Weight {i+1}')

plt.xlabel('Training Step')
plt.ylabel('Weight Value')
plt.title('Weight Evolution in Hebbian Learning')
plt.legend()
plt.grid(True)
plt.show()

# Test the trained network
print("\nTesting the trained network:")
for pattern in patterns:
    output = np.dot(pattern, network.weights)
    print(f"Input: {pattern}, Output: {output}")
```

#### Assignment 3: Delta Learning Rule
**Problem Statement:**
- Train a network with three different inputs and initial weight vector using Delta learning rule in Python
- Assume λ = 1 and c = 0.1

**Sample Implementation:**
```python
import numpy as np
import matplotlib.pyplot as plt

class DeltaRuleNetwork:
    def __init__(self, input_size, learning_rate=1.0, convergence_factor=0.1):
        # Initialize weights with random values
        self.weights = np.random.rand(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.convergence_factor = convergence_factor
    
    def activate(self, x):
        # Linear activation function
        return x
    
    def predict(self, inputs):
        # Add bias input
        inputs_with_bias = np.insert(inputs, 0, 1)
        # Calculate weighted sum
        net = np.dot(inputs_with_bias, self.weights)
        # Apply activation function
        return self.activate(net)
    
    def train(self, training_inputs, labels, epochs=100):
        errors = []
        
        for epoch in range(epochs):
            total_error = 0
            
            for inputs, target in zip(training_inputs, labels):
                # Add bias input
                inputs_with_bias = np.insert(inputs, 0, 1)
                
                # Make prediction
                prediction = self.predict(inputs)
                
                # Calculate error
                error = target - prediction
                total_error += error**2
                
                # Update weights using Delta rule
                delta_w = self.learning_rate * error * inputs_with_bias * self.convergence_factor
                self.weights += delta_w
            
            # Store mean squared error
            mse = total_error / len(training_inputs)
            errors.append(mse)
            
            # Check for convergence
            if mse < 0.001:
                print(f"Converged after {epoch+1} epochs")
                break
                
        return errors

# Example usage
# Define training data (3 inputs)
X = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# Define continuous target values
y = np.array([0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8])

# Create and train network with λ=1 and c=0.1
network = DeltaRuleNetwork(input_size=3, learning_rate=1.0, convergence_factor=0.1)
print("Initial weights:", network.weights)
errors = network.train(X, y, epochs=1000)
print("Final weights:", network.weights)

# Plot error over epochs
plt.figure(figsize=(10, 6))
plt.plot(errors)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Delta Rule Learning Convergence')
plt.yscale('log')
plt.grid(True)
plt.show()

# Test the trained network
print("\nTesting the trained network:")
for inputs, target in zip(X, y):
    prediction = network.predict(inputs)
    print(f"Inputs: {inputs}, Target: {target:.4f}, Prediction: {prediction:.4f}")
```

**Problem Statement:**
- Realization of logical AND, OR functions using Perceptron learning rule

**Sample Implementation:**
```python
import numpy as np
import matplotlib.pyplot as plt

class LogicalPerceptron:
    def __init__(self, learning_rate=0.1):
        # Initialize weights with random values (2 inputs + bias)
        self.weights = np.random.rand(3) * 0.1
        self.learning_rate = learning_rate
    
    def predict(self, inputs):
        # Add bias input
        inputs_with_bias = np.insert(inputs, 0, 1)
        # Calculate weighted sum
        summation = np.dot(inputs_with_bias, self.weights)
        # Apply activation function (step function)
        return 1 if summation > 0 else 0
    
    def train(self, training_inputs, labels, epochs=100):
        errors_history = []
        weights_history = [self.weights.copy()]
        
        for epoch in range(epochs):
            total_error = 0
            
            for inputs, label in zip(training_inputs, labels):
                # Add bias input
                inputs_with_bias = np.insert(inputs, 0, 1)
                
                # Make prediction
                prediction = self.predict(inputs)
                
                # Calculate error
                error = label - prediction
                total_error += abs(error)
                
                # Update weights
                self.weights += self.learning_rate * error * inputs_with_bias
                weights_history.append(self.weights.copy())
            
            errors_history.append(total_error)
            
            # Stop if convergence is reached
            if total_error == 0:
                print(f"Converged after {epoch+1} epochs")
                break
                
        return errors_history, np.array(weights_history)

# Define training data for logical functions
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Define labels for AND and OR functions
y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])

# Train AND function
and_perceptron = LogicalPerceptron()
print("Initial weights for AND:", and_perceptron.weights)
and_errors, and_weights = and_perceptron.train(X, y_and)
print("Final weights for AND:", and_perceptron.weights)

# Train OR function
or_perceptron = LogicalPerceptron()
print("Initial weights for OR:", or_perceptron.weights)
or_errors, or_weights = or_perceptron.train(X, y_or)
print("Final weights for OR:", or_perceptron.weights)

# Plot decision boundaries
plt.figure(figsize=(12, 5))

# Plot for AND function
plt.subplot(1, 2, 1)
for i, x in enumerate(X):
    plt.plot(x[0], x[1], 'o' if y_and[i] == 0 else 's', 
             markersize=10, markeredgecolor='black',
             markerfacecolor='white' if y_and[i] == 0 else 'black')

# Plot decision boundary for AND
w = and_perceptron.weights
if w[2] != 0:  # Avoid division by zero
    x1 = np.linspace(-0.5, 1.5, 100)
    x2 = (-w[0] - w[1] * x1) / w[2]
    plt.plot(x1, x2, 'r-')

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('AND Function')
plt.grid(True)

# Plot for OR function
plt.subplot(1, 2, 2)
for i, x in enumerate(X):
    plt.plot(x[0], x[1], 'o' if y_or[i] == 0 else 's', 
             markersize=10, markeredgecolor='black',
             markerfacecolor='white' if y_or[