"""
This script is used to evaluate the performance of the ev2gym environment.
"""
import sys
import os
import pandas as pd

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)


from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.gurobi_models.tracking_error import PowerTrackingErrorrMin

# from ev2gym.baselines.gurobi_models.profit_max import V2GProfitMaxOracleGB
from milp.profit_max import V2GProfitMaxOracleGB
from milp.profit_max import V2GProfitMaxOracleGB
from ev2gym.baselines.mpc.ocmf_mpc import OCMF_V2G, OCMF_G2V
from ev2gym.baselines.mpc.eMPC import eMPC_V2G, eMPC_G2V
from ev2gym.utilities.utils import get_statistics, print_statistics, calculate_charge_power_potential
from ev2gym.baselines.mpc.eMPC_v2 import eMPC_V2G_v2, eMPC_G2V_v2

from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle
import yaml
from ev2gym.baselines.heuristics import (
    RoundRobin,
    ChargeAsLateAsPossible,
    ChargeAsFastAsPossible,
)
from ev2gym.baselines.heuristics import (
    RoundRobin,
    ChargeAsLateAsPossible,
    ChargeAsFastAsPossible,
)
from ev2gym.baselines.heuristics import ChargeAsFastAsPossibleToDesiredCapacity

from ev2gym.rl_agent.reward import profit_maximization, SqTrError_TrPenalty_UserIncentives, SquaredTrackingErrorReward

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pickle

from decentralized_approach.supervised_decentralized import DecentralizedDNNPolicy

def milp_objective(env, total_costs, user_satisfaction_list, *args):

    reward = total_costs - 100 * sum(user_satisfaction_list)
    return reward

def run_agent(env, agent, episodes=1):
    episode_stats = []
    episode_actions = []
    for ep in range(episodes):
        #print("Episode: ", ep)
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            actions = agent.get_action(env).reshape(-1)
            new_state, reward, done, truncated, stats = env.step(actions)
            total_reward += reward
            episode_actions.append(actions)
        #episode_rewards.append(total_reward)
        print("Final stats: ", stats)
        episode_stats.append(stats)
        #episode_rewards.append(stats['total_profits'])
    return episode_stats, episode_actions


def plot_agent_actions_comparison(milp_actions, sl_actions):
    num_actions = len(milp_actions[0])  # number of action indices per episode
    colors = plt.cm.get_cmap("tab10", num_actions)

    for idx in range(num_actions):
        for ep_idx, (episode_milp, episode_sl) in enumerate(zip(milp_actions, sl_actions)):
            milp_vals = [step[idx] for step in episode_milp]
            sl_vals = [step[idx] for step in episode_sl]

            plt.plot(milp_vals, linestyle='-', color=colors(idx), label=f'Action {idx} MILP' if ep_idx == 0 else "")
            plt.plot(sl_vals, linestyle='--', color=colors(idx), label=f'Action {idx} MPC' if ep_idx == 0 else "")

    plt.xlabel("Time step")
    plt.ylabel("Action value")
    plt.title("MILP vs MPC Actions")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_agent_stats_comparison(milp_stats, sl_stats, title="Average Stats: MILP vs SL Agent"):
    """
    Plot average statistics over all episodes for MILP and SL agents as a bar chart.

    Parameters:
    - milp_stats_list (list of dict): List of episode stats dicts from MILP agent.
    - sl_stats_list (list of dict): List of episode stats dicts from SL agent.
    - title (str): Plot title.
    """

    # Keys to compare
    stat_keys = [
        'total_ev_served', 'total_profits', 'average_user_satisfaction',
        'power_tracker_violation', 'tracking_error', 'energy_tracking_error',
        'energy_user_satisfaction', 'battery_degradation', 'total_reward'
    ]

    # Compute average per key for MILP
    milp_avg = {}
    for key in stat_keys:
        values = [float(np.array(stat[key]).item()) for stat in milp_stats]
        milp_avg[key] = np.mean(values)

    # Compute average per key for SL
    sl_avg = {}
    for key in stat_keys:
        values = [float(np.array(stat[key]).item()) for stat in sl_stats]
        sl_avg[key] = np.mean(values)

    # Plotting
    n_stats = len(stat_keys)
    cols = 3
    rows = (n_stats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axes = axes.flatten()

    for idx, key in enumerate(stat_keys):
        ax = axes[idx]
        ax.bar(["MILP", "SL"], [milp_avg[key], sl_avg[key]], color=["steelblue", "darkorange"])
        ax.set_title(key.replace("_", " ").capitalize(), fontsize=10)
        ax.set_ylabel("Average Value")
        ax.grid(True, axis="y")

    # Hide any extra subplots if present
    for i in range(n_stats, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("decentralized_approach/results/agent_stats_comparison.png")
    plt.show()


def eval(num_episodes=20, save_plots=True):
    config_file = os.path.join(os.getcwd(), "ev2gym-config/V2GProfitPlusLoadsGenerateData.yaml")
    dummy_agent_class = ChargeAsFastAsPossible

    milp_all_stats = []
    milp_all_actions = []
    sl_all_stats = []
    sl_all_actions = []


    for ep in range(num_episodes):
        print(f"\n=== Episode {ep+1}/{num_episodes} ===")
        # 1) Generate replay with dummy agent for this episode
        env_replay_gen = EV2Gym(
            config_file=config_file,
            load_from_replay_path=None,
            verbose=False,
            save_replay=True,
            save_plots=False,
            seed=ep  # use episode index as seed for reproducibility
        )
        state, _ = env_replay_gen.reset()

        dummy_agent = dummy_agent_class()
        done = False
        while not done:
            actions = dummy_agent.get_action(env_replay_gen)
            _, _, done, _, _ = env_replay_gen.step(actions)

        replay_path = f"replay/replay_{env_replay_gen.sim_name}.pkl"
        #print(f"Replay saved to: {replay_path}")

        # 2) Run MILP agent on this replay
        env_milp = EV2Gym(
            config_file=config_file,
            load_from_replay_path=replay_path,
            verbose=False,
            save_replay=False,
            save_plots=save_plots,
        )
        env_milp.set_reward_function(milp_objective)
        milp_agent = V2GProfitMaxOracleGB(replay_path=replay_path, MIPGap=0.0)
        milp_stats, milp_actions = run_agent(env_milp, milp_agent, episodes=1)
        milp_all_stats.extend(milp_stats)
        milp_all_actions.append(milp_actions)
        # 3) Run SL agent on same replay
        env_sl = EV2Gym(
            config_file=config_file,
            load_from_replay_path=replay_path,
            verbose=False,
            save_replay=False,
            save_plots=save_plots,
        )
        env_sl.set_reward_function(milp_objective)
        sl_agent = DecentralizedDNNPolicy(
            model_path="decentralized_approach/models_decentralized/epoch_90_decentralized_ev_policy_without0.pth",
            input_dim=5,
            output_dim=1,
        )
        sl_stats, sl_actions = run_agent(env_sl, sl_agent, episodes=1)
        sl_all_stats.extend(sl_stats)
        sl_all_actions.append(sl_actions)
    
    print_statistics(env_sl)
    print_statistics(env_milp)

    # 4) Plot results aggregated over episodes
    plot_agent_stats_comparison(milp_all_stats, sl_all_stats)
    plot_agent_actions_comparison(milp_all_actions, sl_all_actions)
    
    pd.DataFrame(sl_actions).to_csv('decentralized_approach/results/sl_actions.csv')
    with open("decentralized_approach/results/actions.pkl", "wb") as f:
        pickle.dump({"milp": milp_all_actions, "sl_dec": sl_all_actions}, f)


    # --- Plot comparison ---
    milp_total_profits = [float(np.array(stat['total_profits']).item()) for stat in milp_all_stats]
    sl_total_profits = [float(np.array(stat['total_profits']).item()) for stat in sl_all_stats]


    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, num_episodes+1), milp_total_profits, label="MILP Agent")
    plt.plot(np.arange(1, num_episodes+1), sl_total_profits, label="Supervised Learning Agent")
    plt.xlabel("Episode")
    plt.ylabel("Total Profit")
    plt.title("Reward Comparison: MILP vs SL over Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig("decentralized_approach/results/reward_comparison.png")
    plt.show()



if __name__ == "__main__":
    # while True:
    eval(num_episodes=50)
