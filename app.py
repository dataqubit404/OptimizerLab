# Step 1: Install & Load Libraries
import os
import certifi

# THE SAFE FIX: Tell Python exactly where to find trusted certificates
os.environ['SSL_CERT_FILE'] = certifi.where()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_diabetes, make_regression, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim

st.set_page_config(page_title="Optimizer Comparison", layout="wide")

st.title("Deep Learning: Gradient Descent vs RMSProp vs Adam")
st.write("A comparative study using various datasets to test Optimizer performance.")

# Step 2: Sidebar UI for Datasets & Hyperparameters
st.sidebar.header("1. Choose Dataset")
dataset_choice = st.sidebar.selectbox(
    "Dataset Source", 
    ["California Housing", "Diabetes", "Synthetic (Custom)", "OpenML (Fetch by ID)"]
)

# Dynamic UI based on dataset choice
synthetic_params = None
openml_id = None

if dataset_choice == "Synthetic (Custom)":
    st.sidebar.write("Generate your own data:")
    s_samples = st.sidebar.slider("Samples", 100, 5000, 1000)
    s_features = st.sidebar.slider("Features", 1, 50, 10)
    s_noise = st.sidebar.slider("Noise Level", 0.0, 50.0, 10.0)
    synthetic_params = {'samples': s_samples, 'features': s_features, 'noise': s_noise}
elif dataset_choice == "OpenML (Fetch by ID)":
    st.sidebar.write("Find Regression IDs at openml.org")
    openml_id = st.sidebar.number_input("OpenML Dataset ID (e.g., 541 for CPU performance)", value=541, step=1)

st.sidebar.header("2. Hyperparameters")
optimizer_choice = st.sidebar.selectbox("Choose Optimizer", ["(SGD vs RMSProp vs Adam)", "SGD", "RMSProp", "Adam"])
lr = st.sidebar.slider("Learning Rate", 0.001, 0.5, 0.1, format="%.3f")
epochs = st.sidebar.slider("Epochs", 50, 500, 200)

# Step 3: Load Dataset Dynamically
@st.cache_data
def load_data(choice, synth_params, oml_id):
    if choice == "California Housing":
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['Target'] = data.target
        return df, data.feature_names
        
    elif choice == "Diabetes":
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['Target'] = data.target
        return df, data.feature_names
        
    elif choice == "Synthetic (Custom)":
        X, y = make_regression(
            n_samples=synth_params['samples'], 
            n_features=synth_params['features'], 
            noise=synth_params['noise'], 
            random_state=42
        )
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['Target'] = y
        return df, feature_names
        
    elif choice == "OpenML (Fetch by ID)":
        data = fetch_openml(data_id=oml_id, as_frame=True, parser='auto')
        df = data.frame.dropna() # Drop NaNs for simplicity
        target_col = data.target_names[0]
        y = df[target_col]
        # Keep only numeric features for this simple PyTorch model
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        df_clean = X.copy()
        df_clean['Target'] = y
        return df_clean, X.columns.tolist()

try:
    df, feature_names = load_data(dataset_choice, synthetic_params, openml_id)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Step 4: Explore Data & Visualize Relationships
with st.expander(f"Explore & Visualize Data: {dataset_choice}"):
    # This line explicitly shows you the row and column count!
    st.markdown(f"**Dataset Shape:** {df.shape[0]} rows (samples) by {df.shape[1]} columns (features + target)")
    
    st.write("Preview of the data (first 5 rows):")
    st.dataframe(df.head()) # st.dataframe looks a bit cleaner than st.write
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df['Target'], bins=50, kde=True, ax=ax)
    ax.set_title("Distribution of Target Variable")
    st.pyplot(fig)

# Step 5: Split Dataset into Training & Testing
X = df.drop('Target', axis=1).values
y = df['Target'].values.reshape(-1, 1)

# Standardization
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# Step 6: Build Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

def train_model(opt_name, X_t, y_t, lr, epochs):
    torch.manual_seed(42)
    model = LinearRegressionModel(X_t.shape[1])
    criterion = nn.MSELoss()
    
    if opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_t)
        loss = criterion(predictions, y_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    return model, losses

if st.button("Run Training"):
    # Step 7 & 8: Train, Predict, and Evaluate
    optimizers_to_run = ["SGD", "RMSProp", "Adam"] if optimizer_choice == "(SGD vs RMSProp vs Adam)" else [optimizer_choice]
    
    results = {}
    models = {}
    
    progress_bar = st.progress(0)
    for i, opt in enumerate(optimizers_to_run):
        model, losses = train_model(opt, X_train_t, y_train_t, lr, epochs)
        results[opt] = losses
        models[opt] = model
        progress_bar.progress((i + 1) / len(optimizers_to_run))
        
    st.success("Training Complete!")

    # Step 9: Visualize Loss & Predicted vs Actual
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Loss Curve Comparison")
        fig_loss, ax_loss = plt.subplots()
        for opt in optimizers_to_run:
            ax_loss.plot(results[opt], label=f"{opt}")
        ax_loss.set_xlabel("Epochs")
        ax_loss.set_ylabel("Mean Squared Error Loss")
        ax_loss.set_title("Convergence Speed")
        ax_loss.legend()
        st.pyplot(fig_loss)
        
    with col2:
        st.subheader("Predicted vs Actual (Test Set)")
        best_opt = optimizers_to_run[-1]
        models[best_opt].eval()
        with torch.no_grad():
            y_pred = models[best_opt](X_test_t).numpy()
            
        fig_pred, ax_pred = plt.subplots()
        ax_pred.scatter(y_test, y_pred, alpha=0.3, color='blue')
        ax_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
        ax_pred.set_xlabel("Actual Values")
        ax_pred.set_ylabel("Predicted Values")
        ax_pred.set_title(f"Model Accuracy ({best_opt})")
        st.pyplot(fig_pred)

    st.subheader("Evaluation Metrics")
    metrics_df = []
    for opt in optimizers_to_run:
        models[opt].eval()
        with torch.no_grad():
            preds = models[opt](X_test_t).numpy()
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            metrics_df.append({"Optimizer": opt, "Test MSE": mse, "R2 Score": r2})
            
    metrics_dataframe = pd.DataFrame(metrics_df)
    st.table(metrics_dataframe)
    
    # Dynamically Determine the Winner
    if len(optimizers_to_run) > 1:
        best_optimizer = metrics_dataframe.loc[metrics_dataframe['Test MSE'].idxmin()]['Optimizer']
        st.success(f"🏆 **Winner for this configuration:** {best_optimizer} achieved the lowest Test MSE!")    # Step 10: Conclusion
    st.markdown("### Conclusion")
    st.markdown("""
    * **SGD** usually takes the longest to converge and might require a very carefully tuned learning rate.
    * **RMSProp** adapts the learning rate based on recent gradient magnitudes, leading to faster and smoother convergence than SGD.
    * **Adam** typically converges the fastest and achieves the lowest loss in the fewest epochs by combining momentum and adaptive learning rates.
    """)