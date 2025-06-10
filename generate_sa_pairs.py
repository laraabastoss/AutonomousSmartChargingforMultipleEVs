import numpy as np
import pandas as pd
from ev2gym.models.ev2gym_env import EV2Gym
from milp.profit_max import V2GProfitMaxOracleGB
from tqdm import trange



def extract_state(env, t, episode):
    
    price = env.charge_prices[0, t]

    net_load = 0.0

    socs = []
    satisfaction_vals = []
    connected_flags = []

    for cs in env.charging_stations:
        for ev in cs.evs_connected:
            if ev is not None:
                socs.append(ev.get_soc())
                satisfaction_vals.append(ev.get_user_satisfaction())

                connected = 1
                connected_flags.append(connected)
            else:
                socs.append(0.0)
                connected_flags.append(0)

    satisfaction = np.mean(satisfaction_vals) if satisfaction_vals else 0.0

    future_prices = []
    for i in range(1, 11):
        if t + i < env.simulation_length:
            future_prices.append(env.charge_prices[0, t + i])
        else:
            future_prices.append(0.0)


    return (
        [t % 24, price, net_load, satisfaction] +
        future_prices +
        socs + connected_flags
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

            state = extract_state(env, t, episode)
            actions = oracle.get_action(env)  

            all_states.append(state)
            all_actions.append(actions.tolist())

            _, _, done, _, _ = env.step(actions)

            if done:
                break
        

        for i in range(env.simulation_length):
            correct_net_load = sum(env.tr_inflexible_loads[j][i] for j in range(len(env.tr_inflexible_loads)))
            all_states[idx_offset + i][2] = correct_net_load
        
        idx_offset += env.simulation_length

    states = np.array(all_states)
    actions = np.array(all_actions)
    np.savez("centralized_dataset.npz", states=states, actions=actions)

    num_ports = actions.shape[1]
    
    state_cols = (
        ['time', 'price', 'net_load', 'satisfaction'] +
        [f'future_price_{i}' for i in range(1, 11)] +
        [f'soc_{i}' for i in range(num_ports)] +
        [f'connected_flag_{i}' for i in range(num_ports)]
    )


    action_cols = [f'action_{i}' for i in range(num_ports)]
    all_cols = state_cols + action_cols
    df = pd.DataFrame(np.hstack([states, actions]), columns=all_cols)
    df.to_csv("centralized_dataset.csv", index=False)
    
    print(f"Dataset saved: {states.shape[0]} samples, {states.shape[1]} features")
    print("CSV file written as centralized_dataset.csv")

if __name__ == "__main__":
    generate_dataset("ev2gym-config/V2GProfitPlusLoadsGenerateData.yaml", num_episodes=100)
