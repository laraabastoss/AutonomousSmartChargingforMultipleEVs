import numpy as np
import pandas as pd
from ev2gym.models.ev2gym_env import EV2Gym
from milp.profit_max import V2GProfitMaxOracleGB
from tqdm import trange
import argparse
import os

DATASET_NAME = "centralized_dataset_25Evs"

def extract_state(env, t):
    
    price = env.charge_prices[0, t]

    socs = []
    satisfaction_vals = []
    connected_flags = []
    req_energies = []

    for cs in env.charging_stations:
        for ev in cs.evs_connected:
            if ev is not None:
                socs.append(ev.get_soc())
                satisfaction_vals.append(ev.get_user_satisfaction())
                req_energies.append(ev.required_energy / ev.battery_capacity)
                connected = 1
                connected_flags.append(connected)
            else:
                socs.append(0.0)
                connected_flags.append(0)
                req_energies.append(0.0)

    satisfaction = np.mean(satisfaction_vals) if satisfaction_vals else 0.0



    return (
        [t % 24, price, satisfaction] +
        req_energies +
        socs +
        connected_flags
    )




def generate_dataset(config_file: str, num_episodes: int = 5):
    all_states = []
    all_actions = []
    idx_offset = 0

    for episode in trange(num_episodes, desc="Generating episodes"):
        
        env = EV2Gym(config_file=config_file, verbose=False, save_replay=True)
        env.reset()

        for t in range(env.simulation_length):
            env.step(np.zeros(env.number_of_ports))

        new_replay_path = f"replay/replay_{env.sim_name}.pkl"
        oracle = V2GProfitMaxOracleGB(replay_path=new_replay_path, MIPGap=0.0, verbose=False)
        env = EV2Gym(config_file=config_file, load_from_replay_path=new_replay_path, verbose=False)
        env.reset()

        

        for t in range(env.simulation_length):

            state = extract_state(env, t)
            actions = oracle.get_action(env)  

            all_states.append(state)
            all_actions.append(actions.tolist())

            _, _, done, _, _ = env.step(actions)

            if done:
                break

    dataset_name = DATASET_NAME
        
    states = np.array(all_states)
    actions = np.array(all_actions)
    np.savez(os.path.join("datasets", f"{dataset_name}.npz"), states=states, actions=actions)

    num_ports = actions.shape[1]
    
    state_cols = (
        ['time', 'price', 'satisfaction'] +
        [f"req_energy_ev{i}" for i in range(num_ports)] +
        [f'soc_{i}' for i in range(num_ports)] +
        [f'connected_flag_{i}' for i in range(num_ports)]
    )


    action_cols = [f'action_{i}' for i in range(num_ports)]
    all_cols = state_cols + action_cols
    df = pd.DataFrame(np.hstack([states, actions]), columns=all_cols)
    df.to_csv(os.path.join("datasets", f"{dataset_name}.csv"), index=False)
    
    print(f"Dataset saved: {states.shape[0]} samples, {states.shape[1]} features")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate (state, action) dataset from EV2Gym environment.")
    parser.add_argument(
        "--num_episodes", "-n",
        type=int,
        default=10,
        help="Number of episodes to generate."
    )
    args = parser.parse_args()

    generate_dataset("ev2gym-config/V2GProfitPlusLoadsGenerateData.yaml", num_episodes=args.num_episodes)
