import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import joblib
import pickle

# ==================================================================================================
# --- CONFIGURATION & CONSTANTS ---
# ==================================================================================================

st.set_page_config(page_title="Intrusion Detection System", page_icon="🛡️", layout="wide")

# Model Constants
TIMESTEPS = 20

# Custom CSS
st.markdown("""
<style>
    .main-header { text-align: center; color: #1f77b4; margin-bottom: 2rem; }
    .stPlotlyChart { margin: auto; }
    h3, h4, h5 { text-align: center; }
</style>
""", unsafe_allow_html=True)

# ==================================================================================================
# --- MODEL DEFINITIONS (PYTORCH) ---
# ==================================================================================================

class CNN1D(nn.Module):
    def __init__(self, features):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x).squeeze(-1)

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return self.sigmoid(out).squeeze(-1)

# ==================================================================================================
# --- HELPER FUNCTIONS ---
# ==================================================================================================

@st.cache_resource(show_spinner="Loading models and utilities...")
def load_assets():
    """Load all models, the scaler, and the required feature list."""
    assets = {"models": {}, "scaler": None, "feature_list": None}
    model_dir = "saved_models"
    
    feature_path = os.path.join(model_dir, "features_used.npy")
    if os.path.exists(feature_path):
        feature_list_raw = np.load(feature_path, allow_pickle=True)
        assets["feature_list"] = [str(f).strip() for f in feature_list_raw]
    else:
        st.error("FATAL: `features_used.npy` not found. This file is required.")
        return None

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            assets["scaler"] = pickle.load(f)
    else:
        st.error("FATAL: `scaler.pkl` not found. Cannot proceed.")
        return None

    model_files = {
        "Random Forest": "rf_intrusion_detection_model.joblib",
        "XGBoost": "xgb_intrusion_detection_model.joblib",
        "CNN": "cnn_intrusion_detection_model.pt",
        "LSTM": "lstm_intrusion_detection_model.pt"
    }

    feature_count = len(assets["feature_list"])
    for name, filename in model_files.items():
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            try:
                if name == "CNN": model = CNN1D(features=feature_count)
                elif name == "LSTM": model = LSTMNet(input_size=feature_count)
                
                if name in ["CNN", "LSTM"]:
                    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                    model.eval()
                else:
                    model = joblib.load(path)
                assets["models"][name] = model
            except Exception as e:
                st.error(f"Could not load the '{name}' model. Error: {e}")
        else:
            st.warning(f"Model file not found, skipping: {path}")
            
    return assets

def preprocess_data(df: pd.DataFrame, scaler: MinMaxScaler, required_features: list, timesteps: int):
    """Cleans, scales, and prepares DataFrame for all model types."""
    if df is None or df.empty: return None, None, None, None

    df_processed = df.copy()
    df_processed.columns = df_processed.columns.str.strip()
    
    if 'Label' not in df_processed.columns:
        st.error("The uploaded CSV must contain a 'Label' column.")
        return None, None, None, None
        
    y = df_processed['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1).values
    X_df = df_processed.drop('Label', axis=1)
    
    X_df.replace([np.inf, -np.inf], 0, inplace=True)
    X_df.fillna(0, inplace=True)
    
    missing_cols = set(required_features) - set(X_df.columns)
    for col in missing_cols:
        X_df[col] = 0
    X_df = X_df[required_features]

    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
    X_df.fillna(0, inplace=True)

    try:
        X_scaled = scaler.transform(X_df)
    except Exception as e:
        st.error(f"Error during data scaling: {e}")
        st.error("This means the columns in your CSV do not match the columns the scaler was trained on. Please check the data source.")
        return None, None, None, None

    def create_sequences(features, labels, time_steps):
        X_s, y_s = [], []
        for i in range(len(features) - time_steps):
            X_s.append(features[i:(i + time_steps)])
            y_s.append(labels[i + time_steps])
        return np.array(X_s), np.array(y_s)

    X_seq, y_seq = (np.array([]), np.array([]))
    if len(X_scaled) > timesteps:
        X_seq, y_seq = create_sequences(X_scaled, y, timesteps)
    
    return X_scaled, y, X_seq, y_seq

def get_model_predictions(X_tab, X_seq, models):
    predictions = {}
    for model_name, model in models.items():
        try:
            if model_name in ["CNN", "LSTM"]:
                if X_seq is not None and len(X_seq) > 0:
                    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
                    with torch.no_grad():
                        preds = (model(X_tensor).numpy() > 0.5).astype(int)
                    predictions[model_name] = preds
            else:
                if X_tab is not None and len(X_tab) > 0:
                    predictions[model_name] = model.predict(X_tab)
        except Exception as e:
            st.warning(f"Could not get predictions for {model_name}: {e}")
    return predictions

# ==================================================================================================
# --- MAIN APPLICATION UI ---
# ==================================================================================================

assets = load_assets()
if not assets: st.stop()

models, scaler, FEATURE_LIST = assets["models"], assets["scaler"], assets["feature_list"]

selected = option_menu(menu_title=None, options=["Test for Zero-Day Attacks", "Model Metrics"], icons=["shield-shaded", "bar-chart-line-fill"], orientation="horizontal")

if selected == "Test for Zero-Day Attacks":
    st.title("🧪 Test with New Network Traffic Data")
    st.markdown("Upload a CSV file. A **'Label'** column is required for performance analysis.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file and models and scaler:
        df_test = pd.read_csv(uploaded_file)
        st.info(f"File '{uploaded_file.name}' loaded: {df_test.shape[0]} rows, {df_test.shape[1]} columns.")
        
        with st.spinner("Processing data and running models..."):
            X_tab, y_tab, X_seq, y_seq = preprocess_data(df_test, scaler, FEATURE_LIST, TIMESTEPS)
            
            if X_tab is None:
                st.error("Data preprocessing failed. Please check the file format and ensure required columns are present.")
            else:
                predictions = get_model_predictions(X_tab, X_seq, models)
                st.markdown("---")
                st.header("📊 Prediction Results")

                col1, col2 = st.columns([1.2, 1])
                with col1:
                    st.subheader("Performance Metrics")
                    results = []
                    for name, y_pred in predictions.items():
                        y_true = y_seq if name in ["CNN", "LSTM"] else y_tab
                        if len(y_pred) > 0 and len(y_true) >= len(y_pred): # Ensure true labels are available for all predictions
                             y_true_subset = y_true[:len(y_pred)] # Align lengths
                             results.append({
                                "Model": name, "Accuracy": accuracy_score(y_true_subset, y_pred),
                                "Precision": precision_score(y_true_subset, y_pred, zero_division=0),
                                "Recall": recall_score(y_true_subset, y_pred, zero_division=0),
                                "F1-Score": f1_score(y_true_subset, y_pred, zero_division=0),
                            })
                    if results:
                        st.dataframe(pd.DataFrame(results).set_index("Model").style.format("{:.2%}"))
                    else:
                        st.warning("No metrics could be calculated.")
                
                with col2:
                    st.subheader("Attack vs. Benign Predictions")
                    if predictions:
                        model_to_show = st.selectbox("Select Model:", options=list(predictions.keys()))
                        if model_to_show:
                            preds = predictions[model_to_show]
                            benign, attack = np.sum(preds == 0), np.sum(preds == 1)
                            fig, ax = plt.subplots(figsize=(5, 5))
                            ax.pie([benign, attack], labels=['Benign', 'Attack'], autopct='%1.1f%%', startangle=90, colors=['#2ca02c', '#d62728'], wedgeprops=dict(width=0.4, edgecolor='w'))
                            st.pyplot(fig)

elif selected == "Model Metrics":
    st.title("📈 Model Metrics Dashboard")
    st.markdown("Performance of each model on its pre-saved test dataset.")
    
    test_data_files = {
        "Random Forest": ("X_test_rf.npy", "y_test_rf.npy"), "XGBoost": ("X_test_xgb.npy", "y_test_xgb.npy"),
        "CNN": ("X_test_cnn.npy", "y_test_cnn.npy"), "LSTM": ("X_test_lstm.npy", "y_test_lstm.npy"),
    }
    
    summary_data = []
    for model_name, (X_file, y_file) in test_data_files.items():
        if model_name in models:
            X_path, y_path = os.path.join("saved_models", X_file), os.path.join("saved_models", y_file)
            
            if os.path.exists(X_path) and os.path.exists(y_path):
                X_test, y_test = np.load(X_path, allow_pickle=True), np.load(y_path, allow_pickle=True)
                
                preds = get_model_predictions(X_test, X_test, {model_name: models[model_name]})
                if model_name in preds:
                    summary_data.append({
                        "Model": model_name, "Accuracy": accuracy_score(y_test, preds[model_name]),
                        "Precision": precision_score(y_test, preds[model_name], zero_division=0),
                        "Recall": recall_score(y_test, preds[model_name], zero_division=0),
                        "F1-Score": f1_score(y_test, preds[model_name], zero_division=0),
                        "_cm": confusion_matrix(y_test, preds[model_name])
                    })

    if summary_data:
        st.markdown("---")
        summary_data = sorted(summary_data, key=lambda x: x['Accuracy'], reverse=True)
        st.subheader("📋 Model Comparison Summary")
        st.dataframe(pd.DataFrame(summary_data).drop(columns=['_cm']).set_index("Model").style.format("{:.4f}"))
        
        st.markdown("---")
        st.subheader("📊 Accuracy Comparison")
        fig_acc, ax_acc = plt.subplots(figsize=(10, 5))
        bar_data = {m["Model"]: m["Accuracy"] for m in summary_data}
        ax_acc.bar(bar_data.keys(), bar_data.values(), color=plt.get_cmap('Greens')(np.linspace(0.4, 0.9, len(bar_data))))
        ax_acc.set_ylim(0, 1.05)
        ax_acc.bar_label(ax_acc.containers[0], fmt='%.4f', padding=3)
        st.pyplot(fig_acc)

        st.markdown("---")
        st.subheader("⚙️ Confusion Matrices")
        cm_cols = st.columns(len(summary_data))
        for i, model_data in enumerate(summary_data):
            with cm_cols[i]:
                st.markdown(f"<h5>{model_data['Model']}</h5>", unsafe_allow_html=True)
                fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
                sns.heatmap(model_data['_cm'], annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False, xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
                st.pyplot(fig_cm)
    else:
        st.warning("Could not load any test data. Check for `_test.npy` files in `saved_models`.")