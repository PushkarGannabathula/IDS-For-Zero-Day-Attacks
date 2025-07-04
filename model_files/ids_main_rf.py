import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import pickle

# Specify the path to the CSV file (update this to your CSV location)
file_path = "dataset/Wednesday-workingHours.pcap_ISCX.csv"

def load_cicids2017_data(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        if 'Label' not in df.columns:
            raise ValueError("No 'Label' column found in the dataset.")
        label_counts = df['Label'].value_counts()
        labels = label_counts.index
        plt.figure(figsize=(8, 8))
        plt.pie(label_counts, labels=labels, autopct='%1.1f%%', colors=plt.cm.Set3(np.arange(len(labels))))
        plt.title('Dataset Class Distribution (Before Training)')
        plt.show()
        numeric_cols = df.select_dtypes(include=np.number).columns
        if not numeric_cols.any():
            raise ValueError("No numerical columns found in the dataset.")
        X = df[numeric_cols]
        variances = X.var()
        top_features = variances.nlargest(30).index
        X = X[top_features]
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        for col in X.columns:
            percentile_99 = X[col].quantile(0.99)
            X[col] = X[col].clip(upper=percentile_99)
        y = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
        np.save('saved_models/features_used_rf.npy', X.columns.to_numpy())
        with open('saved_models/scaler_rf.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        return X, y.values, scaler
    except Exception as e:
        print(f"Error loading CICIDS2017: {str(e)}")
        print("Using synthetic data instead...")
        return generate_synthetic_data()

def generate_synthetic_data(samples=1000, features=5):
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (samples // 2, features))
    anomalous_data = np.random.normal(0, 1, (samples // 2, features))
    anomalous_data += np.random.uniform(2, 5, (samples // 2, features)) * np.random.randint(0, 2, (samples // 2, features))
    X = np.vstack((normal_data, anomalous_data))
    y = np.array([0] * (samples // 2) + [1] * (samples // 2))
    scaler = MinMaxScaler().fit(X)
    return X, y, scaler

# Load real data from the specified CSV
X, y, scaler = load_cicids2017_data(file_path=file_path)

# Pie chart of dataset class distribution (before training)
class_counts = np.bincount(y)
labels = ['BENIGN', 'Attack']
plt.figure(figsize=(8, 8))
plt.pie(class_counts, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
plt.title('Dataset Class Distribution (Before Training)')
plt.show()

# Split into train (70%) and test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
np.save('saved_models/X_test_rf.npy', X_test)
np.save('saved_models/y_test_rf.npy', y_test)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
accuracy = (y_pred == y_test).mean()
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

# Identify anomalies
anomalies = X_test[y_pred == 1]
print(f"Detected {len(anomalies)} anomalies out of {len(X_test)} test samples.")

# Pie chart of predicted class distribution (after testing)
pred_counts = np.bincount(y_pred)
plt.figure(figsize=(8, 8))
plt.pie(pred_counts, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
plt.title('Predicted Class Distribution (After Testing)')
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Save the model
joblib.dump(rf, 'rf_intrusion_detection_model.joblib')
print("Model saved as 'rf_intrusion_detection_model.joblib'") 