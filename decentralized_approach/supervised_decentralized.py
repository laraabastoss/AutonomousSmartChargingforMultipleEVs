
import os 
import sys
# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)


import wandb
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from decentralized_approach.generate_decentralized_dataset import get_soc
import joblib



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

class DecentralizedDNNPolicy:
    def __init__(self, model_path, input_dim, output_dim):
        self.model = EVPolicyNet(input_dim, output_dim)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def get_action(self, env):
        t = env.current_step
        price = env.charge_prices[0, t]
        net_load = sum(env.tr_inflexible_loads[i][t] for i in range(len(env.tr_inflexible_loads)))

        actions = []
        total_soc = 0
        for cs in env.charging_stations:
            for ev in cs.evs_connected:
                if ev is not None:
                    soc = ev.get_soc()
                else:
                    soc = 0.0
                total_soc += 1 - soc

        for cs in env.charging_stations:
            for ev in cs.evs_connected:
                if ev is not None:
                    soc = ev.get_soc()
                else:
                    soc = 0.0
                state_single_ev = np.array([t%24, price, net_load] + [total_soc] + [soc], dtype=np.float32)
                state_tensor = torch.tensor(state_single_ev).unsqueeze(0)

            with torch.no_grad():
                action = self.model(state_tensor)
            
            action = max(min(action.item(), 0.5), -0.5)

            action = np.array([action])
            
            actions.append(action)


        return torch.from_numpy(np.array(actions)).squeeze(0).numpy()

if __name__ == '__main__':
    # ----------------------
    # Configuration
    # ----------------------
    DATA_PATH = "decentralized_approach/data/decentralized_dataset.npz"
    MODELS_PATH = 'decentralized_approach/models_decentralized'
    EPOCHS = 200
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3

    # ----------------------
    # Initialize wandb and mlflow
    # ----------------------
    wandb.init(project="ev-policy-training", config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE
    })

    mlflow.set_experiment("ev_policy_training")

    # ----------------------
    # Load Data
    # ----------------------
    data = np.load(DATA_PATH)
    X = data['states']
    y = data['actions']
    mask = y[:, 0] != 0
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)

    # ----------------------
    # Train Model
    # ----------------------
    with mlflow.start_run():
        model = EVPolicyNet(input_dim=X.shape[1], output_dim=y.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.MSELoss()

        losses = []
        for epoch in tqdm(range(EPOCHS)):
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
            wandb.log({"loss": avg_loss, "epoch": epoch})
            mlflow.log_metric("loss", avg_loss, step=epoch)
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test_tensor)
                test_loss = loss_fn(test_preds, y_test_tensor).item()
                wandb.log({"test_loss": test_loss})
                mlflow.log_metric("test_loss", test_loss)
            
            if epoch % 10 == 0:
                # Save model
                path = os.path.join(os.getcwd(), MODELS_PATH, f'epoch_{epoch}_decentralized_ev_policy.pth')
                torch.save(model.state_dict(), path)
                print(f"Model saved to {path}")
                mlflow.pytorch.log_model(model, "model")
                wandb.save(path)
