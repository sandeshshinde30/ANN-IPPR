# 📈 Stock Price Prediction using Feed-Forward Neural Network
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 📁 Load Dataset (Replace with your stock data CSV)
# Dataset should have columns: Date, Open, High, Low, Close, Volume
df = pd.read_csv('stock_prices.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)

# 🔧 Data Preprocessing
def preprocess_data(data, look_back=60):
    # Use only 'Close' price for prediction
    close_prices = data.filter(['Close']).values
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # Create time-series sequences
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

# Parameters
LOOK_BACK = 60  # Use 60 days to predict next day
TEST_SIZE = 0.2  # 20% test data

# Prepare data
X, y, scaler = preprocess_data(df, LOOK_BACK)

# Split dataset
split = int(len(X) * (1 - TEST_SIZE))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 🧠 Build Feed-Forward Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(LOOK_BACK,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Linear activation for regression
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])

# 🚀 Train the Model
early_stop = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=[early_stop],
                    verbose=1)

# 📊 Evaluate and Predict
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1,1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

# 📈 Visualization
plt.figure(figsize=(16,8))
plt.title('Stock Price Prediction')
plt.plot(y_train_actual, label='Actual Train Prices')
plt.plot([None]*len(y_train_actual) + list(y_test_actual), label='Actual Test Prices')
plt.plot(train_predict, label='Training Prediction')
plt.plot([None]*len(y_train_actual) + list(test_predict), label='Testing Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# 🧮 Calculate Metrics
def calculate_metrics(actual, predicted):
    mse = np.mean((actual - predicted)**2)
    mae = np.mean(np.abs(actual - predicted))
    return mse, mae

train_mse, train_mae = calculate_metrics(y_train_actual, train_predict)
test_mse, test_mae = calculate_metrics(y_test_actual, test_predict)

print(f"\nTraining MSE: {train_mse:.4f}, MAE: {train_mae:.4f}")
print(f"Testing MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")