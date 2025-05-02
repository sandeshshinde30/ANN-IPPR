# 📁 digit_classifier.py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 🎯 Step 1: Load and Prepare MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape data for MLP (flatten 28x28 images to 784-dimensional vectors)
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 🔧 Step 2: Build Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 🚀 Step 3: Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 📊 Step 4: Train the Model
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

# 🧪 Step 5: Evaluate on Test Set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")

# 📈 Step 6: Visualize Results
def plot_results(history, sample_preds):
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Sample predictions
    plt.subplot(1, 2, 2)
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        pred = np.argmax(sample_preds[i])
        true = np.argmax(y_test[i])
        plt.title(f"Pred: {pred}\nTrue: {true}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Generate predictions
sample_preds = model.predict(x_test[:6])
plot_results(history, sample_preds)