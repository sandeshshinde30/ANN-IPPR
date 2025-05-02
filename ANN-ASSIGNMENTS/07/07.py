# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the Parkinson's disease dataset (download from UCI or load from a local file)
# For this example, we'll use a public dataset from UCI.
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/Parkinsons%20Telemonitoring.zip"

# Load data into a pandas dataframe
data = pd.read_csv(url, compression='zip', header=0, delimiter=',', quotechar='"')

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Separate the features and target variable
X = data.drop(['status'], axis=1)  # Drop the 'status' column (target variable)
y = data['status']  # The 'status' column is our target (0 = No Parkinson's, 1 = Parkinson's)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for many machine learning algorithms)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model performance
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

