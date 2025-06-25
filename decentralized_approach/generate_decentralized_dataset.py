import os
import sys

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.getcwd()))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
from ev2gym.models.ev2gym_env import EV2Gym
from milp.profit_max import V2GProfitMaxOracleGB
from tqdm import trange

def get_soc(env, idx):
    ev = env.charging_stations[idx][0]
    if ev is not None:
        soc = ev.get_soc()
    else:
        soc = 0.0
    return soc

def extract_state(env, t, episode):
    price = env.charge_prices[0, t] 
    net_load = 0.0

    socs = []
    total_soc = 0

    for cs in env.charging_stations:
        for ev in cs.evs_connected:
            if ev is not None:
                soc = ev.get_soc()
            else:
                soc = 0.0
            socs.append(soc)
            total_soc += 1 - soc


    states = []
    for cs in env.charging_stations:
        for ev in cs.evs_connected:
            if ev is not None:
                soc = ev.get_soc()
            else:
                soc = 0.0
            state_single_ev = np.array([t%24, price, net_load] + [total_soc] + [soc], dtype=np.float32)
            states.append(state_single_ev)

    return states



def generate_dataset(config_file: str, num_episodes: int = 5):
    all_states = []
    all_actions = []

    for episode in trange(num_episodes, desc="Generating episodes"):
        
        env = EV2Gym(config_file=config_file, verbose=False, save_replay=True, save_plots=True)
        env.reset()

        for t in range(env.simulation_length):
            env.step(np.zeros(env.number_of_ports))

        new_replay_path = f"replay/replay_{env.sim_name}.pkl"
        oracle = V2GProfitMaxOracleGB(replay_path=new_replay_path, MIPGap=0.0, verbose=False)
        env = EV2Gym(config_file=config_file, load_from_replay_path=new_replay_path, verbose=False, save_plots=True)
        env.reset()
    
        for t in range(env.simulation_length):

            states = extract_state(env, t, episode)
            actions = oracle.get_action(env)  
            for s, a in zip(states, actions):
                all_states.append(s)
                all_actions.append(a)

            _, _, done, _, _ = env.step(actions)

            if done:
                break
        

        for i in range(env.simulation_length):
            correct_net_load = sum(env.tr_inflexible_loads[j][i] for j in range(len(env.tr_inflexible_loads)))
            all_states[i][2] = correct_net_load

        states = np.array(all_states)
        actions = np.array(all_actions).reshape(-1,1)
        np.savez("decentralized_approach/data/decentralized_dataset.npz", states=states, actions=actions)

        state_cols = ['time', 'price', 'net_load'] + ['total_soc'] + ['individual_soc']
        action_cols = [f'action']
        all_cols = state_cols + action_cols
        df = pd.DataFrame(np.hstack([states, actions]), columns=all_cols)
        df.to_csv(f"decentralized_approach/data/decentralized_csv/decentralized_dataset_{episode}.csv", index=False)
        
    print(f"Dataset saved: {states.shape[0]} samples, {states.shape[1]} features")
    print("CSV file written as decentralized_dataset.csv")

if __name__ == "__main__":
    generate_dataset("ev2gym-config/V2GProfitPlusLoadsGenerateData.yaml", num_episodes=360)
