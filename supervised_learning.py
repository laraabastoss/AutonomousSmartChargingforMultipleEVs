import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from generate_sa_pairs import extract_state
from generate_sa_pairs import DATASET_NAME
from torch.utils.data import WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler



#TODO: Add learning rate decay and early stop mechanism

# ----------------------
# Configuration
# ----------------------
DATA_PATH = f"./datasets/{DATASET_NAME}.npz"
MODEL_PATH = f"./models/{DATASET_NAME}_ev_policy.pth"
GRU_DATA_PATH = "gru_consumption_dataset.npz"
GRU_MODEL_PATH = "gru_model.pth"
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


class WeightedMSELoss(nn.Module):
    def __init__(self, alpha=5.0, beta=1.0, threshold=1e-3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.mse = nn.L1Loss(reduction='none')

    def forward(self, predictions, targets):
        loss_per_sample = self.mse(predictions, targets)

        # Calculate mask: 1 if any action > threshold
        is_nonzero = (targets.abs() > self.threshold).any(dim=1).float()

        weights = self.alpha * is_nonzero + self.beta * (1 - is_nonzero)
        weights = weights.view(-1, 1)  # Expand to match shape if needed

        weighted_loss = (loss_per_sample * weights).mean()
        return weighted_loss


# -------------------------------------------------------
# Training Functions
# -------------------------------------------------------
def train_ev_policy_net(
    train_loader,
    X_test_tensor,
    y_test_tensor,
    input_dim,
    output_dim,
    model_path=MODEL_PATH,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    show_plot=True
):
    """
    Train a feed-forward EVPolicyNet on the provided dataloader.
    Saves the trained model to `model_path`.
    """

    model = EVPolicyNet(input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = WeightedMSELoss(alpha=8.0, beta=1.0, threshold=0.01)

    losses = []
    for epoch in range(epochs):
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

    # Optionally plot training loss
    if show_plot:
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("EVPolicyNet Training Loss")
        plt.grid(True)
        plt.show()

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor)
        test_loss = loss_fn(test_preds, y_test_tensor).item()
    print(f"Feed-forward Net Test Loss: {test_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Feed-forward Net model saved to {model_path}")

    return model  # return the trained model if needed


def train_gru_model(
    train_loader,
    X_test_tensor,
    y_test_tensor,
    model_path=GRU_MODEL_PATH,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    show_plot=True
):
    """
    Train a GRU model on the provided dataloader.
    Saves the trained model to `model_path`.
    """

    model = GRU(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
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
            print(f"[GRU] Epoch {epoch}, Loss: {avg_loss:.4f}")

    # Optionally plot training loss
    if show_plot:
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("GRU Training Loss")
        plt.grid(True)
        plt.show()

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor)
        test_loss = loss_fn(test_preds, y_test_tensor).item()
    print(f"GRU Test Loss: {test_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"GRU model saved to {model_path}")

    return model

# ----------------------
# DNN Agent Wrapper
# ----------------------
class CentralizedDNNPolicy:
    def __init__(self, model_path, input_dim, output_dim, gru_path="gru_model.pth", predict_netload = True):
        self.model = EVPolicyNet(input_dim, output_dim)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.gru = GRU()
        self.gru.load_state_dict(torch.load(gru_path))
        self.gru.eval()

        self.predict_netload = predict_netload


    def get_action(self, env):
        t = env.current_step
        price = env.charge_prices[0, t]

        if self.predict_netload == False:
            net_load = sum(env.tr_inflexible_loads[i][t] for i in range(len(env.tr_inflexible_loads)))

        else:
            if t<168:
                net_load = 0.0
            else:
                netloads =[]
                for i in range(t - 168, t):
                    load = sum(env.tr_inflexible_loads[j][i] for j in range(len(env.tr_inflexible_loads)))
                    netloads.append(load)

                netloads = np.array(netloads).reshape(1, 168, 1).astype(np.float32)
                netloads_tensor = torch.tensor(netloads)

                with torch.no_grad():
                    net_load_pred = self.gru(netloads_tensor).item()
                    net_load = net_load_pred
        socs = []
        req_energies = []
        connected_flags = []
        satisfaction_vals = []

        for cs in env.charging_stations:
            for ev in cs.evs_connected:
                if ev is not None:
                    socs.append(ev.get_soc())
                    req_energies.append(ev.required_energy / ev.battery_capacity)
                    connected_flags.append(1)
                    satisfaction_vals.append(ev.get_user_satisfaction())
                else:
                    socs.append(0.0)
                    req_energies.append(0.0)
                    connected_flags.append(0)

        satisfaction = np.mean(satisfaction_vals) if satisfaction_vals else 0.0

        state_vec = np.array(
            [t / env.simulation_length, price, satisfaction, net_load] +
            req_energies +
            socs +
            connected_flags,
            dtype=np.float32
        )
        state_tensor = torch.tensor(state_vec).unsqueeze(0)

        with torch.no_grad():
            action_tensor = self.model(state_tensor)

        return action_tensor.squeeze(0).numpy()


if __name__ == "__main__":
    # 1) Load the feed-forward net data
    data = np.load(DATA_PATH)
    X = data['states']
    y = data['actions']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    threshold = 0.01
    nonzero_mask = (np.abs(y_train) > threshold).any(axis=1)
    weights = np.where(nonzero_mask, 6.0, 1.0)

    weights_tensor = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=BATCH_SIZE,
        sampler=sampler 
    )

    # 2) Train feed-forward net if model doesn't exist
    if not os.path.exists(MODEL_PATH):
        print("\nTraining EVPolicyNet (Feed-forward)...")
        train_ev_policy_net(
            train_loader=train_loader,
            X_test_tensor=X_test_tensor,
            y_test_tensor=y_test_tensor,
            input_dim=X.shape[1],
            output_dim=y.shape[1],
            model_path=MODEL_PATH,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            show_plot=False
        )
    else:
        print(f"Feed-forward model already exists at {MODEL_PATH}. Skipping training.")


    # 3) (Optional) Load & train GRU data
    # if os.path.exists(GRU_DATA_PATH):
    if os.path.exists(GRU_DATA_PATH):
        gru_data = np.load(GRU_DATA_PATH)
        X_gru = gru_data['X']
        y_gru = gru_data['y']

        X_gru_train, X_gru_test, y_gru_train, y_gru_test = train_test_split(
            X_gru, y_gru, test_size=0.1, random_state=42
        )

        X_gru_train_tensor = torch.tensor(X_gru_train, dtype=torch.float32)
        y_gru_train_tensor = torch.tensor(y_gru_train, dtype=torch.float32)
        X_gru_test_tensor = torch.tensor(X_gru_test, dtype=torch.float32)
        y_gru_test_tensor = torch.tensor(y_gru_test, dtype=torch.float32)

        gru_train_loader = DataLoader(
            TensorDataset(X_gru_train_tensor, y_gru_train_tensor),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        if not os.path.exists(GRU_MODEL_PATH):
            print("\nTraining GRU model...")
            train_gru_model(
                train_loader=gru_train_loader,
                X_test_tensor=X_gru_test_tensor,
                y_test_tensor=y_gru_test_tensor,
                model_path=GRU_MODEL_PATH,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE
            )
        else:
            print(f"GRU model already exists at {GRU_MODEL_PATH}. Skipping training.")

    else:
        print(f"GRU data file not found at {GRU_DATA_PATH}, skipping GRU training.")