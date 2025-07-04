import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import pickle

# Specify the path to the CSV file (update this to your CSV location)
# Example absolute path (Windows); replace with your path
file_path = "dataset/Wednesday-workingHours.pcap_ISCX.csv"

# Function to load and preprocess a single CICIDS2017 CSV file
def load_cicids2017_data(file_path, timesteps=20):
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        
        print(f"Loading {file_path}...")
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Verify 'Label' column exists
        if 'Label' not in df.columns:
            raise ValueError("No 'Label' column found in the dataset.")

        # Pie chart of dataset class distribution with displaying different attacks (before training)
        label_counts = df['Label'].value_counts()
        labels = label_counts.index
        plt.figure(figsize=(8, 8))
        plt.pie(label_counts, labels=labels, autopct='%1.1f%%', colors=plt.cm.Set3(np.arange(len(labels))))
        plt.title('Dataset Class Distribution (Before Training)')
        plt.show()
        
        # Select numerical features
        numeric_cols = df.select_dtypes(include=np.number).columns
        if not numeric_cols.any():
            raise ValueError("No numerical columns found in the dataset.")
        X = df[numeric_cols]
        
        # Feature selection: Keep top 30 features by variance
        variances = X.var()
        top_features = variances.nlargest(30).index
        X = X[top_features]
        
        # Replace NaNs and infinities with 0
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Clip large values to 99th percentile
        for col in X.columns:
            percentile_99 = X[col].quantile(0.99)
            X[col] = X[col].clip(upper=percentile_99)
        
        # Map labels: BENIGN to 0, others to 1
        y = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        
        # Normalize features
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        
        # Reshape to 3D for CNN: [samples, timesteps, features]
        features = X.shape[1]
        samples = X.shape[0]
        X_3d = []
        for i in range(0, samples - timesteps + 1):
            X_3d.append(X[i:i + timesteps])
        X_3d = np.array(X_3d)
        y = y[timesteps - 1:]  # Align labels with sequences
        
        print(f"Loaded dataset: {X_3d.shape[0]} samples, {timesteps} timesteps, {features} features")
        np.save('saved_models/features_used_cnn.npy', X.columns.to_numpy())
        with open('saved_models/scaler_cnn.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        return X_3d, y, scaler
    
    except Exception as e:
        print(f"Error loading CICIDS2017: {str(e)}")
        print("Using synthetic data instead...")
        return generate_synthetic_data(timesteps=timesteps)

# Synthetic data fallback (fixed to return scaler)
def generate_synthetic_data(samples=1000, timesteps=20, features=5):
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (samples // 2, timesteps, features))
    anomalous_data = np.random.normal(0, 1, (samples // 2, timesteps, features))
    anomalous_data += np.random.uniform(2, 5, (samples // 2, timesteps, features)) * np.random.randint(0, 2, (samples // 2, timesteps, features))
    X = np.vstack((normal_data, anomalous_data))
    y = np.array([0] * (samples // 2) + [1] * (samples // 2))
    scaler = MinMaxScaler().fit(X.reshape(-1, features))  # Dummy scaler
    return X, y, scaler

# Parameters
TIMESTEPS = 20  # Reduced for efficiency
BATCH_SIZE = 64  # Increased for faster training
EPOCHS = 20
PATIENCE = 5

# Load real data from the specified CSV
X, y, scaler = load_cicids2017_data(file_path=file_path, timesteps=TIMESTEPS)
FEATURES = X.shape[2]  # Number of features from the CSV

# Pie chart of dataset class distribution (before training)
class_counts = np.bincount(y)
labels = ['BENIGN', 'Attack']
plt.figure(figsize=(8, 8))
plt.pie(class_counts, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
plt.title('Dataset Class Distribution (Before Training)')
plt.show()

# Split into train (70%) and test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
np.save('saved_models/X_test_cnn.npy', X_test)
np.save('saved_models/y_test_cnn.npy', y_test)

# Convert to torch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class CNN1D(nn.Module):
    def __init__(self, timesteps, features):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x: (batch, timesteps, features) -> (batch, features, timesteps)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x.squeeze(-1)

model = CNN1D(TIMESTEPS, FEATURES).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Early stopping variables
best_val_loss = float('inf')
best_model_state = None
patience_counter = 0
train_acc_hist, val_acc_hist = [], []
train_loss_hist, val_loss_hist = [], []

for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
        preds = (outputs > 0.5).float()
        train_correct += (preds == yb).sum().item()
        train_total += xb.size(0)
    train_loss /= train_total
    train_acc = train_correct / train_total
    train_acc_hist.append(train_acc)
    train_loss_hist.append(train_loss)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            val_loss += loss.item() * xb.size(0)
            preds = (outputs > 0.5).float()
            val_correct += (preds == yb).sum().item()
            val_total += xb.size(0)
    val_loss /= val_total
    val_acc = val_correct / val_total
    val_acc_hist.append(val_acc)
    val_loss_hist.append(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Evaluate on test data
model.eval()
with torch.no_grad():
    y_pred_prob = []
    for xb, _ in test_loader:
        xb = xb.to(device)
        outputs = model(xb)
        y_pred_prob.append(outputs.cpu().numpy())
    y_pred_prob = np.concatenate(y_pred_prob)
    y_pred = (y_pred_prob > 0.5).astype(int)

loss = val_loss_hist[-1]
accuracy = val_acc_hist[-1]
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_acc_hist, label='Train Accuracy')
plt.plot(val_acc_hist, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss_hist, label='Train Loss')
plt.plot(val_loss_hist, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Identify anomalies
anomalies = X_test[y_pred.flatten() == 1]
print(f"Detected {len(anomalies)} anomalies out of {len(X_test)} test samples.")

# Pie chart of predicted class distribution (after testing)
pred_counts = np.bincount(y_pred.flatten())
labels = ['BENIGN', 'Attack']
plt.figure(figsize=(8, 8))
plt.pie(pred_counts, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
plt.title('Predicted Class Distribution (After Testing)')
plt.show()

# Compute metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['BENIGN', 'Attack'], yticklabels=['BENIGN', 'Attack'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

torch.save(model.state_dict(), "cnn_intrusion_detection_model.pt")
print("Model saved as 'cnn_intrusion_detection_model.pt'")