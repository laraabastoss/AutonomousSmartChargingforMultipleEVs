import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import os
import matplotlib.cm as cm

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
    num_ports = milp_actions.shape[1]
    #num_ports = 1
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
    sl_vals = [float(np.array(sl_stat_dict["no_gru"][key]).item()) for key in stat_keys]

    os.makedirs(save_dir, exist_ok=True)

    n_stats = len(stat_keys)
    cols = 3
    rows = (n_stats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axes = axes.flatten()

    for idx, key in enumerate(stat_keys):
        ax = axes[idx]
        ax.bar(["MILP", "SL-TrueLoad"],
               [milp_vals[idx], sl_vals[idx]],
               color=["steelblue", "darkorange"])
        ax.set_title(key.replace("_", " ").capitalize(), fontsize=10)
        ax.set_ylabel("Value")
        ax.grid(True, axis="y")

    for idx in range(n_stats, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f"Episode {episode_idx + 1} - MILP vs SL", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_path = os.path.join(save_dir, f"episode_{episode_idx + 1:03}.png")
    plt.savefig(plot_path)
    plt.close()