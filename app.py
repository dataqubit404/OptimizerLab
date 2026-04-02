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
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim

st.set_page_config(page_title="Optimizer Comparison", layout="wide")

st.title("Deep Learning: Gradient Descent vs RMSProp vs Adam")
st.write("A comparative study using Linear Regression on the California Housing Dataset.")

# Step 2: Load Dataset
@st.cache_data
def load_data():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target
    return df, housing.feature_names

df, feature_names = load_data()

# Sidebar UI
st.sidebar.header("Hyperparameters")
optimizer_choice = st.sidebar.selectbox("Choose Optimizer", ["(SGD vs RMSProp vs Adam)", "SGD", "RMSProp", "Adam"])
lr = st.sidebar.slider("Learning Rate", 0.001, 0.5, 0.01, format="%.3f")
epochs = st.sidebar.slider("Epochs", 50, 500, 200)

# Step 3 & 4: Explore Data & Visualize Relationships
with st.expander("Explore & Visualize Data"):
    st.write(df.head())
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df['Price'], bins=50, kde=True, ax=ax)
    ax.set_title("Distribution of House Prices")
    st.pyplot(fig)

# Step 5: Split Dataset into Training & Testing
X = df.drop('Price', axis=1).values
y = df['Price'].values.reshape(-1, 1)

# Standardization (Crucial for Neural Networks/Optimizers)
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
    torch.manual_seed(42) # Ensure same starting weights for fair comparison
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
        # Pick the last trained model to show scatter plot
        best_opt = optimizers_to_run[-1]
        models[best_opt].eval()
        with torch.no_grad():
            y_pred = models[best_opt](X_test_t).numpy()
            
        fig_pred, ax_pred = plt.subplots()
        ax_pred.scatter(y_test, y_pred, alpha=0.3, color='blue')
        ax_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
        ax_pred.set_xlabel("Actual Prices")
        ax_pred.set_ylabel("Predicted Prices")
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
            
    st.table(pd.DataFrame(metrics_df))

    # Step 10: Conclusion
    st.markdown("### Conclusion")
    st.markdown("""
    * **SGD** usually takes the longest to converge and might require a very carefully tuned learning rate.
    * **RMSProp** adapts the learning rate based on recent gradient magnitudes, leading to faster and smoother convergence than SGD.
    * **Adam** typically converges the fastest and achieves the lowest loss in the fewest epochs by combining momentum and adaptive learning rates.
    """)