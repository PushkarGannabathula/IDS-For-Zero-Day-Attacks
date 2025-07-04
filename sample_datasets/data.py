import numpy as np
import pandas as pd
import os

# Common parameters
attack_labels = ["DoS slowloris", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "Heartbleed"]
benign_label = "BENIGN"
all_labels = attack_labels + [benign_label]

# Output configurations
output_configs = {
    "sample_dataset_standard.csv": {"BENIGN": 120, **{label: 6 for label in attack_labels}},
    "sample_dataset_large.csv": {"BENIGN": 400, **{label: 20 for label in attack_labels}},
    "sample_dataset_high_attack.csv": {"BENIGN": 120, **{label: 16 for label in attack_labels}},
    "sample_dataset_benign.csv": {"BENIGN": 100},
}

# Column names (same as your script)
column_names = [
    ' Destination Port', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
    ' Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
    ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
    ' Bwd Packet Length Max', ' Bwd Packet Length Min', ' Bwd Packet Length Mean',
    ' Bwd Packet Length Std', ' Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean',
    ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', ' Fwd IAT Total', ' Fwd IAT Mean',
    ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', ' Bwd IAT Total', ' Bwd IAT Mean',
    ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', ' Fwd PSH Flags', ' Bwd PSH Flags',
    ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length', ' Bwd Header Length',
    ' Fwd Packets/s', ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length',
    ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', ' FIN Flag Count',
    ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count',
    ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size',
    ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', ' Fwd Header Length', ' Fwd Avg Bytes/Bulk',
    ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk',
    ' Bwd Avg Bulk Rate', ' Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets',
    ' Subflow Bwd Bytes', ' Init_Win_bytes_forward', ' Init_Win_bytes_backward',
    ' act_data_pkt_fwd', ' min_seg_size_forward', ' Active Mean', ' Active Std', ' Active Max',
    ' Active Min', ' Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min', ' Label'
]

def generate_rows(n_samples, label):
    np.random.seed(42 + hash(label) % 10000)  # Label-based reproducibility
    data = {}

    data[' Destination Port'] = np.random.choice([80, 443, 22, 389, 88, 49666], size=n_samples)
    data[' Flow Duration'] = np.random.randint(100, 600000, n_samples)
    data[' Total Fwd Packets'] = np.random.randint(1, 150, n_samples)
    data[' Total Backward Packets'] = np.random.randint(0, 150, n_samples)

    data[' Total Length of Fwd Packets'] = np.random.randint(6, 3200, n_samples)
    data[' Total Length of Bwd Packets'] = np.random.randint(6, 3200, n_samples)

    data[' Fwd Packet Length Max'] = np.random.randint(6, 1600, n_samples)
    data[' Fwd Packet Length Min'] = np.random.randint(0, 6, n_samples)
    data[' Fwd Packet Length Mean'] = np.random.uniform(1, 350, n_samples)
    data[' Fwd Packet Length Std'] = np.random.uniform(0, 700, n_samples)

    data[' Bwd Packet Length Max'] = np.random.randint(6, 1600, n_samples)
    data[' Bwd Packet Length Min'] = np.random.randint(0, 6, n_samples)
    data[' Bwd Packet Length Mean'] = np.random.uniform(1, 600, n_samples)
    data[' Bwd Packet Length Std'] = np.random.uniform(0, 800, n_samples)

    data[' Flow Bytes/s'] = np.random.uniform(1e5, 6e6, n_samples)
    data[' Flow Packets/s'] = np.random.uniform(100, 50000, n_samples)

    data[' Flow IAT Mean'] = np.random.uniform(1, 2000, n_samples)
    data[' Flow IAT Std'] = np.random.uniform(0, 5000, n_samples)
    data[' Flow IAT Max'] = np.random.uniform(3, 14000, n_samples)
    data[' Flow IAT Min'] = np.random.randint(0, 10, n_samples)

    data[' Fwd IAT Total'] = np.random.uniform(0, 500000, n_samples)
    data[' Fwd IAT Mean'] = np.random.uniform(1, 1000, n_samples)
    data[' Fwd IAT Std'] = np.random.uniform(0, 5000, n_samples)
    data[' Fwd IAT Max'] = np.random.uniform(3, 15000, n_samples)
    data[' Fwd IAT Min'] = np.random.randint(0, 10, n_samples)

    data[' Bwd IAT Total'] = np.random.uniform(0, 500000, n_samples)
    data[' Bwd IAT Mean'] = np.random.uniform(1, 1000, n_samples)
    data[' Bwd IAT Std'] = np.random.uniform(0, 5000, n_samples)
    data[' Bwd IAT Max'] = np.random.uniform(3, 15000, n_samples)
    data[' Bwd IAT Min'] = np.random.randint(0, 10, n_samples)

    data[' Fwd PSH Flags'] = np.random.choice([0, 1], n_samples)
    data[' Bwd PSH Flags'] = np.random.choice([0, 1], n_samples)
    data[' Fwd URG Flags'] = np.random.choice([0, 1], n_samples)
    data[' Bwd URG Flags'] = np.random.choice([0, 1], n_samples)

    data[' Fwd Header Length'] = np.random.randint(20, 600, n_samples)
    data[' Bwd Header Length'] = np.random.randint(20, 600, n_samples)

    data[' Fwd Packets/s'] = np.random.uniform(0, 1e6, n_samples)
    data[' Bwd Packets/s'] = np.random.uniform(0, 1e6, n_samples)

    data[' Min Packet Length'] = np.random.randint(0, 50, n_samples)
    data[' Max Packet Length'] = np.random.randint(50, 1600, n_samples)
    data[' Packet Length Mean'] = np.random.uniform(0, 1500, n_samples)
    data[' Packet Length Std'] = np.random.uniform(0, 1000, n_samples)
    data[' Packet Length Variance'] = data[' Packet Length Std'] ** 2

    for flag in ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG', 'CWE', 'ECE']:
        data[f'{flag} Flag Count'] = np.random.randint(0, 2, n_samples)

    data[' Down/Up Ratio'] = np.random.uniform(0, 3, n_samples)
    data[' Average Packet Size'] = np.random.uniform(0, 1500, n_samples)
    data[' Avg Fwd Segment Size'] = data[' Fwd Packet Length Mean']
    data[' Avg Bwd Segment Size'] = data[' Bwd Packet Length Mean']
    data[' Fwd Header Length.1'] = data[' Fwd Header Length']

    for bulk in ['Fwd', 'Bwd']:
        for name in ['Avg Bytes/Bulk', 'Avg Packets/Bulk', 'Avg Bulk Rate']:
            data[f'{bulk} {name}'] = np.zeros(n_samples)

    data[' Subflow Fwd Packets'] = data[' Total Fwd Packets']
    data[' Subflow Fwd Bytes'] = data[' Total Length of Fwd Packets']
    data[' Subflow Bwd Packets'] = data[' Total Backward Packets']
    data[' Subflow Bwd Bytes'] = data[' Total Length of Bwd Packets']

    data[' Init_Win_bytes_forward'] = np.random.choice([29200, 253, 2081, 237, 260, 0], n_samples)
    data[' Init_Win_bytes_backward'] = np.random.choice([29200, 253, 2081, 237, 260, -1], n_samples)
    data[' act_data_pkt_fwd'] = np.random.choice([0, 1, 2, 3, 4, 10], n_samples)
    data[' min_seg_size_forward'] = np.full(n_samples, 32)

    for col in ['Active Mean', 'Active Std', 'Active Max', 'Active Min',
                'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']:
        data[col] = np.zeros(n_samples)

    data[' Label'] = [label] * n_samples
    return pd.DataFrame(data)

def create_all_datasets():
    for filename, label_counts in output_configs.items():
        frames = [generate_rows(n, label) for label, n in label_counts.items()]
        df_final = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)
        df_final.to_csv(filename, index=False)
        print(f"✅ Created {filename} with {len(df_final)} rows")

if __name__ == "__main__":
    create_all_datasets()
