"""
This script generates a dataset using the EV2Gym environment by performing random actions.
"""

from ev2gym.models.ev2gym_env import EV2Gym
import pandas as pd
# from ev2gym.baselines.gurobi_models.profit_max import V2GProfitMaxOracleGB
from milp.profit_max import V2GProfitMaxOracleGB

import numpy as np

def generate_data():
    # Generate a dataset using the EV2Gym environment by just doing random actions
    config_file = "ev2gym-config/GenerateDataset.yaml"

    env = EV2Gym(
        config_file=config_file,
        verbose=False,
        save_replay=True,
    )

    state, _ = env.reset()
    i = 0
    for t in range(env.simulation_length):
        i += 1
        actions = np.ones(env.number_of_ports)

        observation, _, _, _, _ = env.step(actions)

    # Use the replay to generate optimal actions using MILP

    agent = V2GProfitMaxOracleGB(replay_path=env.replay_path, MIPGap=0.0)

    env = EV2Gym(
        config_file=config_file,
        load_from_replay_path=env.replay_path,
        verbose=False,
        save_plots=True,
    )

    _, _ = env.reset()
    
    dataset = pd.DataFrame(columns=['observation', 'reward', 'done', 'truncated', 'stats', 'actions'])

    for t in range(env.simulation_length):
        actions = agent.get_action(env)

        observation, reward, done, truncated, stats = env.step(
            actions, visualize=False
        )
        dataset = pd.concat(
            [
                dataset,
                pd.DataFrame(
                    {
                        'observation': [observation],
                        'reward': [reward],
                        'done': [done],
                        'truncated': [truncated],
                        'stats': [stats],
                        'actions': [actions]
                    }
                )
            ],
            ignore_index=True
        )

        if done:
            print(stats)
            break

    return dataset

if __name__ == "__main__":
    # Adjust episode length and number of episodes as needed
    dataset = pd.DataFrame(columns=['observation', 'reward', 'done', 'truncated', 'stats', 'actions'])
    for i in range(2):
        dataset = pd.concat([dataset, generate_data()])
    print("Dataset generated, saving...")
    dataset.to_csv('dataset.csv', index=False)
