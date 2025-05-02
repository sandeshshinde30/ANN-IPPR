# 🧠 Hebbian Learning Network Implementation
import numpy as np

# 🎯 Define three different input patterns
inputs = np.array([
    [1, 0, 1],  # Pattern 1
    [0, 1, 1],  # Pattern 2
    [1, 1, 0]   # Pattern 3
], dtype=np.float32)

# 🔢 Initialize weights (small random values)
np.random.seed(42)  # For reproducibility
weights = np.random.uniform(-0.2, 0.2, size=(3,))  # Initial weight vector
print("Initial weights:", weights)

# ⚙️ Learning parameters
learning_rate = 0.1
epochs = 3

# 📚 Hebbian Learning Process
print("\n🚀 Starting Training...")
for epoch in range(epochs):
    print(f"\n🔁 Epoch {epoch+1}/{epochs}")
    for i, x in enumerate(inputs):
        # 🧮 Calculate output (pre-activation)
        y = np.dot(x, weights)
        
        # ✨ Apply Hebbian Rule: Δw = η * x * y
        delta_weights = learning_rate * x * y
        
        # ⬆️ Update weights
        weights += delta_weights
        
        # 📊 Display updates
        print(f"\nPattern {i+1}: {x}")
        print(f"Output before update: {y:.3f}")
        print(f"Weight changes: {np.round(delta_weights, 3)}")
        print(f"Updated weights: {np.round(weights, 3)}")

print("\n🎉 Training Complete!")
print("Final weights:", np.round(weights, 3))