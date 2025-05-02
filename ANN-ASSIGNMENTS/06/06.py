import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Configuration
DATA_DIR = os.path.abspath("datasets/speech_commands_extracted")
CLASSES = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('_') and not d.startswith('.')]
SAMPLE_RATE = 16000
DURATION = 1  # Seconds
N_MFCC = 40
BATCH_SIZE = 32

# 1. Dataset Preparation
def load_dataset(data_dir):
    # Get list of audio files and labels
    file_paths = []
    labels = []
    
    for label in CLASSES:
        class_dir = os.path.join(data_dir, label)
        if not os.path.exists(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.endswith('.wav'):
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(label)
    
    # Convert string labels to integer indices
    label_to_index = {label: i for i, label in enumerate(CLASSES)}
    labels = [label_to_index[label] for label in labels]
    return tf.data.Dataset.from_tensor_slices((file_paths, labels))

# 2. Audio Preprocessing
def preprocess_audio(file_path, label):
    # Load audio
    audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio)
    audio = tf.squeeze(audio, axis=-1)
    
    # Ensure 1-second duration
    audio = audio[:SAMPLE_RATE * DURATION]
    audio = tf.pad(audio, [[0, SAMPLE_RATE * DURATION - tf.shape(audio)[0]]])
    
    # Convert to MFCC
    stfts = tf.signal.stft(audio, frame_length=256, frame_step=128)
    spectrograms = tf.abs(stfts)
    
    # MFCC calculation
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=20.0,
        upper_edge_hertz=4000.0)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
    
    # Normalize MFCCs
    mfccs = (mfccs - tf.math.reduce_mean(mfccs)) / tf.math.reduce_std(mfccs)
    
    return mfccs, label

# 3. Create Model
def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1)),
        
        # CNN Layers
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D((2,2)),
        
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D((2,2)),
        
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D((2,2)),
        
        # RNN Layers
        tf.keras.layers.Reshape((-1, 128)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
        
        # Classification Head
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 4. Dataset Splitting
def split_dataset(dataset, test_ratio=0.2, val_ratio=0.1):
    dataset = dataset.shuffle(10000)
    total = sum(1 for _ in dataset)
    
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    train_size = total - test_size - val_size
    
    train_ds = dataset.take(train_size)
    remaining = dataset.skip(train_size)
    val_ds = remaining.take(val_size)
    test_ds = remaining.skip(val_size)
    
    return train_ds, val_ds, test_ds

# 5. Training and Evaluation
def train_model():
    # Load and preprocess data
    raw_dataset = load_dataset(DATA_DIR)
    dataset = raw_dataset.map(preprocess_audio, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Split dataset
    train_ds, val_ds, test_ds = split_dataset(dataset)
    
    # Batch and prefetch
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Dynamically determine input shape
    for x_batch, y_batch in train_ds.take(1):
        print("Batch shape:", x_batch.shape)
        input_shape = x_batch.shape[1:]
        break
    
    # Create model
    model = create_model(input_shape, len(CLASSES))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_model()