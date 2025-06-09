"""
This script is used to evaluate the performance of the ev2gym environment.
"""


from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.gurobi_models.tracking_error import PowerTrackingErrorrMin
import matplotlib.cm as cm


# from ev2gym.baselines.gurobi_models.profit_max import V2GProfitMaxOracleGB
from milp.profit_max import V2GProfitMaxOracleGB
from milp.profit_max import V2GProfitMaxOracleGB
from ev2gym.baselines.mpc.ocmf_mpc import OCMF_V2G, OCMF_G2V
from ev2gym.baselines.mpc.eMPC import eMPC_V2G, eMPC_G2V

from ev2gym.baselines.mpc.eMPC_v2 import eMPC_V2G_v2, eMPC_G2V_v2

from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle

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
import os

from supervised_learning import CentralizedDNNPolicy

def milp_objective(env, total_costs, user_satisfaction_list, *args):

    reward = total_costs - 100 * sum(user_satisfaction_list)
    return reward


def run_agent(env, agent, episodes=1, return_actions=False):
    episode_stats = []
    episode_actions = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        actions_per_timestep = []
        while not done:
            actions = agent.get_action(env)
            actions_per_timestep.append(actions.copy())
            _, reward, done, _, stats = env.step(actions)
            total_reward += reward
        episode_stats.append(stats)
        if return_actions:
            episode_actions.append(np.array(actions_per_timestep))
    return (episode_stats, episode_actions) if return_actions else episode_stats


def plot_action_histogram(actions, episode_idx, agent_name="SL", save_dir="action_histograms"):
    """
    Plot histogram of all action values for one episode (all EVs, all timesteps).
    """
    os.makedirs(save_dir, exist_ok=True)

    flattened_actions = actions.flatten()

    plt.figure(figsize=(8, 6))
    plt.hist(flattened_actions, bins=50, color="steelblue", edgecolor="black")
    plt.title(f"{agent_name} Action Distribution - Episode {episode_idx + 1}")
    plt.xlabel("Action Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{agent_name.lower()}_hist_ep{episode_idx + 1:03}.png"))
    plt.close()



def plot_actions_per_episode(milp_actions, sl_actions, episode_idx, save_dir="action_plots"):
    """
    Plots actions for each EV across time steps for one episode for both MILP and SL policies.
    """
    os.makedirs(save_dir, exist_ok=True)
    #num_ports = milp_actions.shape[1]
    num_ports = 4
    timesteps = milp_actions.shape[0]
    color_map = cm.get_cmap('tab10', num_ports)

    plt.figure(figsize=(14, 8))
    for port in range(num_ports):
        color = color_map(port)
        plt.plot(range(timesteps), milp_actions[:, port], label=f"Action {port} MILP", linewidth=1, color=color)
        plt.plot(range(timesteps), sl_actions[:, port], '--', label=f"Action {port} SL", linewidth=1, color=color)

    plt.xlabel("Time step")
    plt.ylabel("Action value")
    plt.title(f"MILP vs SL Actions for Episode {episode_idx + 1}")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"actions_episode_{episode_idx + 1:03}.png"))
    plt.close()



def plot_agent_stats_per_episode(milp_stat, sl_stat_dict, episode_idx, save_dir):
    """
    Plot 9 bar charts in one image comparing MILP, SL w/GRU, SL w/o GRU for one episode.
    """

    stat_keys = [
        'total_ev_served', 'total_profits', 'average_user_satisfaction',
        'power_tracker_violation', 'tracking_error', 'energy_tracking_error',
        'energy_user_satisfaction', 'battery_degradation', 'total_reward'
    ]

    milp_vals = [float(np.array(milp_stat[key]).item()) for key in stat_keys]
    sl_gru_vals = [float(np.array(sl_stat_dict["with_gru"][key]).item()) for key in stat_keys]
    sl_nogru_vals = [float(np.array(sl_stat_dict["no_gru"][key]).item()) for key in stat_keys]

    os.makedirs(save_dir, exist_ok=True)

    n_stats = len(stat_keys)
    cols = 3
    rows = (n_stats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axes = axes.flatten()

    for idx, key in enumerate(stat_keys):
        ax = axes[idx]
        ax.bar(["MILP", "SL-GRU", "SL-TrueLoad"],
               [milp_vals[idx], sl_gru_vals[idx], sl_nogru_vals[idx]],
               color=["steelblue", "darkorange", "seagreen"])
        ax.set_title(key.replace("_", " ").capitalize(), fontsize=10)
        ax.set_ylabel("Value")
        ax.grid(True, axis="y")

    for idx in range(n_stats, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f"Episode {episode_idx + 1} - MILP vs SL (GRU vs True Load)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_path = os.path.join(save_dir, f"episode_{episode_idx + 1:03}.png")
    plt.savefig(plot_path)
    plt.close()

def eval(num_episodes=5, save_plots=True):
    config_file = "ev2gym-config/V2GProfitPlusLoads.yaml"
    dummy_agent_class = ChargeAsFastAsPossible

    milp_all_stats = []
    sl_all_stats = []


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
        milp_stats, milp_actions = run_agent(env_milp, milp_agent, episodes=1, return_actions=True)
        milp_all_stats.extend(milp_stats)

            # 3a) SL with GRU (predicted netload)
        env_sl_gru = EV2Gym(
            config_file=config_file,
            load_from_replay_path=replay_path,
            verbose=False,
            save_replay=False,
            save_plots=save_plots,
        )
        env_sl_gru.set_reward_function(milp_objective)
        sl_agent_gru = CentralizedDNNPolicy(
            model_path="centralized_ev_policy.pth",
            input_dim=env_sl_gru.number_of_ports + 4,
            output_dim=env_sl_gru.number_of_ports,
            predict_netload=True
        )
        sl_gru_stats = run_agent(env_sl_gru, sl_agent_gru, episodes=1)

        # 3b) SL without GRU (uses true netload)
        env_sl_nogru = EV2Gym(
            config_file=config_file,
            load_from_replay_path=replay_path,
            verbose=False,
            save_replay=False,
            save_plots=save_plots,
        )
        env_sl_nogru.set_reward_function(milp_objective)
        sl_agent_nogru = CentralizedDNNPolicy(
            model_path="centralized_ev_policy.pth",
            input_dim=env_sl_nogru.number_of_ports + 4,
            output_dim=env_sl_nogru.number_of_ports,
            predict_netload=False
        )
        sl_nogru_stats, sl_actions = run_agent(env_sl_nogru, sl_agent_nogru, episodes=1, return_actions=True)

        plot_action_histogram(actions=milp_actions[0], episode_idx=ep, agent_name="MILP")
        plot_action_histogram(actions=sl_actions[0], episode_idx=ep, agent_name="SL")

        # Save stats

        plot_actions_per_episode(
            milp_actions=milp_actions[0], 
            sl_actions=sl_actions[0],
            episode_idx=ep,
            save_dir="action_plots"
        )

        sl_all_stats.append({
            "with_gru": sl_gru_stats[0],
            "no_gru": sl_nogru_stats[0]
        })


    # 4) Plot and save per-episode comparisons
    for i in range(num_episodes):
        plot_agent_stats_per_episode(
            milp_stat=milp_all_stats[i],
            sl_stat_dict=sl_all_stats[i],
            episode_idx=i,
            save_dir="supervised_learning_results"
        )


    # --- Plot comparison ---
    milp_total_profits = [float(np.array(stat['total_profits']).item()) for stat in milp_all_stats]
    sl_gru_total_profits = [float(np.array(stat['with_gru']['total_profits']).item()) for stat in sl_all_stats]
    sl_nogru_total_profits = [float(np.array(stat['no_gru']['total_profits']).item()) for stat in sl_all_stats]



    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, num_episodes+1), milp_total_profits, label="MILP Agent")
    plt.plot(np.arange(1, num_episodes+1), sl_nogru_total_profits, label="Supervised Learning Agent")
    plt.xlabel("Episode")
    plt.ylabel("Total Profit")
    plt.title("Reward Comparison: MILP vs SL over Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    # while True:
    eval(num_episodes=5)
