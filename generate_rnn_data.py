import numpy as np
import pandas as pd
from ev2gym.models.ev2gym_env import EV2Gym
from milp.profit_max import V2GProfitMaxOracleGB
from tqdm import trange

def generate_gru_training_data(consumption_series, seq_length=168):
    """
    Generate input-output pairs for GRU training from energy consumption time series.

    Args:
        consumption_series: 1D numpy array of energy consumption values (hourly).
        seq_length: Length of the input sequence to use (default 168).

    Returns:
        X: numpy array of shape (num_samples, seq_length, 1)
        y: numpy array of shape (num_samples, 1)
    """
    X = []
    y = []
    for i in range(len(consumption_series) - seq_length):
        X.append(consumption_series[i:i+seq_length])
        y.append(consumption_series[i+seq_length])
    X = np.array(X)
    y = np.array(y)

    # reshape for GRU input: (samples, seq_length, input_size=1)
    X = X.reshape(-1, seq_length, 1)
    y = y.reshape(-1, 1)
    return X, y



def generate_gru_dataset(config_file: str, num_episodes: int = 1):
    all_sequences = []
    all_targets = []

    for episode in trange(num_episodes, desc="Generating episodes for GRU"):
        env = EV2Gym(config_file=config_file, verbose=False)
        env.reset()

        # Run the environment for the full simulation_length
        for _ in range(env.simulation_length):
            env.step(np.zeros(env.number_of_ports))  

        # Sum inflexible loads per timestep
        episode_consumption = np.array([
            sum(env.tr_inflexible_loads[j][t] for j in range(len(env.tr_inflexible_loads)))
            for t in range(env.simulation_length)
        ])

        # Generate sequences and targets with seq_length=168
        X, y = generate_gru_training_data(episode_consumption, seq_length=168)

        all_sequences.append(X)
        all_targets.append(y)

    X_all = np.concatenate(all_sequences, axis=0)
    y_all = np.concatenate(all_targets, axis=0)

    np.savez("gru_consumption_dataset.npz", X=X_all, y=y_all)
    
    print(f"GRU dataset generated: {X_all.shape[0]} samples")

if __name__ == "__main__":
    generate_gru_dataset("ev2gym-config/V2GProfitPlusLoadsGenerateData.yaml", num_episodes=100)
