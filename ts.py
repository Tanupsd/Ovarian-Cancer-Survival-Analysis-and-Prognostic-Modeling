import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from lifelines import KaplanMeierFitter, CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from joblib import dump
from imblearn.over_sampling import SMOTE

# Define Paths
DATA_PATH = "data/clinical_data.csv"
PREPROCESSED_PATH = "data/preprocessed_data.csv"
ENCODED_PATH = "data/encoded_data.csv"
NORMALIZED_PATH = "data/normalized_data.csv"
BALANCED_PATH = "data/balanced_data.csv"
MODELS_PATH = "models"  # Folder for saving models

# Create required directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ----------------- Data Preprocessing -----------------
def preprocess_data():
    print("[INFO] Preprocessing raw data...")
    df = pd.read_csv(DATA_PATH, na_values=['--'])
    df.rename(columns={'days_to_death': 'time', 'vital_status': 'event'}, inplace=True)
    df['event'] = df['event'].map({'Dead': 1, 'Alive': 0})
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    median_time = df.loc[df['time'] >= 0, 'time'].median()
    df['time'] = df['time'].fillna(median_time)
    df.loc[df['time'] < 0, 'time'] = 0
    df.dropna(subset=['event'], inplace=True)
    df.to_csv(PREPROCESSED_PATH, index=False)
    return df

def encode_data():
    print("[INFO] Encoding data...")
    df = pd.read_csv(PREPROCESSED_PATH)
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col].astype(str))
    df.to_csv(ENCODED_PATH, index=False)
    return df

def normalize_data():
    print("[INFO] Normalizing data...")
    df = pd.read_csv(ENCODED_PATH)
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    df.to_csv(NORMALIZED_PATH, index=False)
    return df

def augment_data_with_smote():
    print("[INFO] Applying SMOTE to balance the dataset...")
    df = pd.read_csv(NORMALIZED_PATH)
    if 'event' not in df.columns:
        return df

    X = df.drop(columns=['event', 'time'])
    y = df['event']
    if y.nunique() < 2:
        return df

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled['event'] = y_resampled
    df_resampled['time'] = df['time'].median()

    df_resampled.to_csv(BALANCED_PATH, index=False)
    return df_resampled

# ----------------- Train/Test Split -----------------
def get_train_test_data(test_size=0.2, random_state=42):
    df = pd.read_csv(NORMALIZED_PATH)
    df = df.dropna(subset=['time', 'event'])
    df = df[df['time'] >= 0]

    X = df.drop(columns=['time', 'event'])
    y_time = df['time'].values
    y_event = df['event'].values

    # Split indices, then use them for X, y_time, y_event
    train_idx, test_idx = train_test_split(
        range(len(df)), test_size=test_size, random_state=random_state, stratify=y_event
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_time_train, y_time_test = y_time[train_idx], y_time[test_idx]
    y_event_train, y_event_test = y_event[train_idx], y_event[test_idx]

    return X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test

# ----------------- Models -----------------
def run_kaplan_meier(y_time_test, y_event_test):
    print("[INFO] Running Kaplan-Meier on test set...")
    kmf = KaplanMeierFitter()
    kmf.fit(y_time_test, event_observed=y_event_test)
    kmf.plot_survival_function()
    plt.title("Kaplan-Meier Survival Curve (Test Data)")
    plt.show()
    predictions = np.full_like(y_time_test, fill_value=np.mean(y_time_test), dtype=np.float32)
    return predictions, y_time_test, y_event_test

def run_cox_ph_model(X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test):
    print("[INFO] Running Cox Proportional Hazards model...")

    df_train = pd.DataFrame(X_train, columns=X_train.columns)
    df_train["time"] = y_time_train
    df_train["event"] = y_event_train

    df_test = pd.DataFrame(X_test, columns=X_test.columns)
    df_test["time"] = y_time_test
    df_test["event"] = y_event_test

    # Drop columns with low variance or constant values
    low_variance_cols = [col for col in df_train.columns if col not in ["time", "event"] and df_train[col].nunique() <= 1]
    if low_variance_cols:
        print("[INFO] Dropping low-variance columns:", low_variance_cols)
        df_train = df_train.drop(columns=low_variance_cols)
        df_test = df_test.drop(columns=low_variance_cols, errors="ignore")

    # Drop any rows with NaNs (safety)
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    # Fit CoxPH
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_train, duration_col="time", event_col="event")
    predictions = -cph.predict_partial_hazard(df_test).values.flatten()

    return predictions, df_test["time"].values, df_test["event"].values

def run_random_forest_survival(X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test):
    print("[INFO] Running Random Forest Survival model...")
    y_train_struct = np.array([(e, t) for e, t in zip(y_event_train, y_time_train)],
                              dtype=[('event', '?'), ('time', '<f8')])
    model = RandomSurvivalForest(n_estimators=100, random_state=42)
    model.fit(X_train, y_train_struct)
    dump(model, os.path.join(MODELS_PATH, "random_forest_survival_model.joblib"))

    surv_funcs = model.predict_survival_function(X_test)
    predictions = np.array([sf.y.mean() for sf in surv_funcs])
    return predictions, y_time_test, y_event_test

"""class DeepSurv(nn.Module):
    def __init__(self, input_dim):
        super(DeepSurv, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def run_deepsurv(X_train, X_test, y_event_train, y_event_test, y_time_test):
    print("[INFO] Running DeepSurv model...")
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_event_train, dtype=torch.float32)

    model = DeepSurv(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X_train_t).squeeze()
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), os.path.join(MODELS_PATH, "deepsurv_model.pth"))
    model.eval()

    with torch.no_grad():
        X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
        predictions = -model(X_test_t).squeeze().numpy()

    return predictions, y_time_test, y_event_test
"""

class DeepSurv(nn.Module):
    def __init__(self, input_dim):
        super(DeepSurv, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def cox_ph_loss(risk, times, events, eps=1e-7):
    times = times.view(-1)
    events = events.view(-1)
    risk = risk.view(-1)

    order = torch.argsort(times, descending=True)
    risk_ord = risk[order]
    events_ord = events[order]

    exp_risk = torch.exp(risk_ord)
    cumsum_exp = torch.cumsum(exp_risk, dim=0)
    log_cumsum = torch.log(cumsum_exp + eps)

    diff = risk_ord - log_cumsum
    event_mask = events_ord
    if event_mask.sum() == 0:
        return torch.tensor(0.0, dtype=risk.dtype, device=risk.device)
    neg_log_likelihood = -torch.sum(diff * event_mask) / (event_mask.sum() + eps)
    return neg_log_likelihood

def run_deepsurv(X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test,
                 epochs=150, lr=1e-3, weight_decay=1e-4):
    print("[INFO] Running DeepSurv model with Cox partial likelihood loss...")

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
    y_event_train_t = torch.tensor(y_event_train.astype(float), dtype=torch.float32)
    y_time_train_t = torch.tensor(y_time_train.astype(float), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_t, X_test_t = X_train_t.to(device), X_test_t.to(device)
    y_event_train_t, y_time_train_t = y_event_train_t.to(device), y_time_train_t.to(device)

    model = DeepSurv(X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t).squeeze()
        loss = cox_ph_loss(outputs, y_time_train_t, y_event_train_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"[DeepSurv] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), os.path.join(MODELS_PATH, "deepsurv_model.pth"))
    print("DeepSurv model trained and saved successfully.")

    model.eval()
    with torch.no_grad():
        preds_test = model(X_test_t).squeeze().cpu().numpy()
        predictions = -preds_test  # keep sign consistent

    return predictions, y_time_test, y_event_test
# ----------------- Main -----------------
if __name__ == "__main__":
    preprocess_data()
    encode_data()
    normalize_data()
    augment_data_with_smote()

    X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = get_train_test_data()

    run_kaplan_meier(y_time_test, y_event_test)
    run_cox_ph_model(X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test)
    run_random_forest_survival(X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test)
    run_deepsurv(X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test)