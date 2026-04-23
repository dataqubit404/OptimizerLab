import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = load_diabetes()
X = data.data
y = data.target.reshape(-1, 1)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x)

def train(opt_name, lr, epochs=200):
    torch.manual_seed(42)  # Critical for reproducibility
    model = LinearRegressionModel(X_train_t.shape[1])
    criterion = nn.MSELoss()
    
    if opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        loss.backward()
        optimizer.step()
    
    # Eval on Test Set
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)
        y_pred = model(X_test_t)
        test_loss = criterion(y_pred, y_test_t)
    return test_loss.item()

print("Testing Diabetes Dataset...")
for lr in [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2]:
    print(f"--- LR = {lr} ---")
    s = train("SGD", lr)
    r = train("RMSProp", lr)
    a = train("Adam", lr)
    print(f"SGD: {s:.2f} | RMS: {r:.2f} | Adam: {a:.2f}")
