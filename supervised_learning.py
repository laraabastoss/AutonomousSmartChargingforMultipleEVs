import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


#TODO: Add learning rate decay and early stop mechanism

# ----------------------
# Configuration
# ----------------------
DATA_PATH = "centralized_dataset.npz"
MODEL_PATH = "centralized_ev_policy.pth"
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


# ----------------------
# Model Definition
# ----------------------
class EVPolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class GRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length=168, input_size=1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)  # out shape: (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]       # Take last time step output
        out = self.fc(out)        # Map to output size (1)
        return out


# ----------------------
# Load Data
# ----------------------
data = np.load(DATA_PATH)
X = data['states']
y = data['actions']

# Normalize time
X[:, 0] = X[:, 0] / X[:, 0].max()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)


# ----------------------
# Train Model
# ----------------------
if not os.path.exists(MODEL_PATH):

    model = EVPolicyNet(input_dim=X.shape[1], output_dim=y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    losses = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    # Plot loss
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor)
        test_loss = loss_fn(test_preds, y_test_tensor).item()
        print(f"Test Loss: {test_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

else:
    print(f"Model already exists at {MODEL_PATH}. Skipping training.")


# ----------------------
# DNN Agent Wrapper
# ----------------------
class CentralizedDNNPolicy:
    def __init__(self, model_path, input_dim, output_dim):
        self.model = EVPolicyNet(input_dim, output_dim)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def get_action(self, env):
        t = env.current_step
        price = env.charge_prices[0, t]
        net_load = sum(env.tr_inflexible_loads[i][t] for i in range(len(env.tr_inflexible_loads)))

        socs = []
        for i in range(env.number_of_ports):
            try:
                ev = env.EVs[i] if i < len(env.EVs) else None
                soc = ev.get_soc() if ev is not None else 0.0
            except Exception:
                soc = 0.0
            socs.append(soc)

        state_vec = np.array([t / env.simulation_length, price, net_load] + socs, dtype=np.float32)
        state_tensor = torch.tensor(state_vec).unsqueeze(0)

        with torch.no_grad():
            action_tensor = self.model(state_tensor)

        return action_tensor.squeeze(0).numpy()
