"""
This script is used to evaluate the performance of the ev2gym environment.
"""


from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.gurobi_models.tracking_error import PowerTrackingErrorrMin


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

from supervised_learning import CentralizedDNNPolicy

def milp_objective(env, total_costs, user_satisfaction_list, *args):

    reward = total_costs - 100 * sum(user_satisfaction_list)
    return reward


def run_agent(env, agent, episodes=1):
    episode_stats = []
    for ep in range(episodes):
        #print("Episode: ", ep)
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            actions = agent.get_action(env)
            new_state, reward, done, truncated, stats = env.step(actions)
            total_reward += reward
        #episode_rewards.append(total_reward)
        print("Final stats: ", stats)
        episode_stats.append(stats)
        #episode_rewards.append(stats['total_profits'])
    return episode_stats


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
    plt.show()

'''
def eval2():
    """
    Runs an evaluation of the ev2gym environment.
    """

    save_plots = True

    # replay_path = "./replay/replay_sim_2025_05_12_106720.pkl"
    replay_path = None

    config_file = "ev2gym-config/V2GProfitPlusLoads.yaml"

    env = EV2Gym(
        config_file=config_file,
        load_from_replay_path=replay_path,
        verbose=False,
        save_replay=True,
        save_plots=save_plots,
    )
    env = EV2Gym(
        config_file=config_file,
        load_from_replay_path=replay_path,
        verbose=False,
        save_replay=True,
        save_plots=save_plots,
    )

    new_replay_path = f"replay/replay_{env.sim_name}.pkl"

    state, _ = env.reset()

    ev_profiles = env.EVs_profiles
    max_time_of_stay = max(
        [ev.time_of_departure - ev.time_of_arrival for ev in ev_profiles]
    )
    min_time_of_stay = min(
        [ev.time_of_departure - ev.time_of_arrival for ev in ev_profiles]
    )
    max_time_of_stay = max(
        [ev.time_of_departure - ev.time_of_arrival for ev in ev_profiles]
    )
    min_time_of_stay = min(
        [ev.time_of_departure - ev.time_of_arrival for ev in ev_profiles]
    )

    print(f"Number of EVs: {len(ev_profiles)}")
    print(f"Max time of stay: {max_time_of_stay}")
    print(f"Min time of stay: {min_time_of_stay}")

    # exit()
    # agent = OCMF_V2G(env, control_horizon=30, verbose=True)
    # agent = OCMF_G2V(env, control_horizon=25, verbose=True)
    # agent = eMPC_V2G(env, control_horizon=15, verbose=False)
    # agent = V2GProfitMaxOracle(env,verbose=True)
    # agent = PowerTrackingErrorrMin(new_replay_path)
    # agent = eMPC_G2V(env, control_horizon=15, verbose=False)
    # agent = eMPC_V2G_v2(env, control_horizon=10, verbose=False)
    # agent = RoundRobin(env, verbose=False)
    # agent = ChargeAsLateAsPossible(verbose=False)
    agent = ChargeAsFastAsPossible()
    # agent = ChargeAsFastAsPossibleToDesiredCapacity()
    rewards = []

    for t in range(env.simulation_length):
        actions = agent.get_action(env)

        new_state, reward, done, truncated, stats = env.step(actions)  # takes action
        rewards.append(reward)

        if done:
            #print(stats)
            print(f"End of simulation at step {env.current_step}")
            break

    # return
    # Solve optimally
    # Power tracker optimizer
    # agent = PowerTrackingErrorrMin(replay_path=new_replay_path)
    # # Profit maximization optimizer
    agent = V2GProfitMaxOracleGB(replay_path=new_replay_path)
    # # Simulate in the gym environment and get the rewards

    env = EV2Gym(config_file=config_file,
                       load_from_replay_path=new_replay_path,
                       verbose=False,
                       save_plots=True,
                       )
    state, _ = env.reset()
    rewards_opt = []

    for t in range(env.simulation_length):
        actions = agent.get_action(env)
        # if verbose:
        #     print(f' OptimalActions: {actions}')

        new_state, reward, done, truncated, stats = env.step(
            actions, visualize=False)  # takes action
        rewards_opt.append(reward)

        # if verbose:
        #     print(f'Reward: {reward} \t Done: {done}')

        if done:
            print(stats)
            break
'''

def eval(num_episodes=20, save_plots=True):
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
        milp_stats = run_agent(env_milp, milp_agent, episodes=1)
        milp_all_stats.extend(milp_stats)

        # 3) Run SL agent on same replay
        env_sl = EV2Gym(
            config_file=config_file,
            load_from_replay_path=replay_path,
            verbose=False,
            save_replay=False,
            save_plots=save_plots,
        )
        env_sl.set_reward_function(milp_objective)
        sl_agent = CentralizedDNNPolicy(
            model_path="centralized_ev_policy.pth",
            input_dim=env_sl.number_of_ports + 3,
            output_dim=env_sl.number_of_ports,
        )
        sl_stats = run_agent(env_sl, sl_agent, episodes=1)
        sl_all_stats.extend(sl_stats)

    # 4) Plot results aggregated over episodes
    plot_agent_stats_comparison(milp_all_stats, sl_all_stats)

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
    plt.show()



if __name__ == "__main__":
    # while True:
    eval(num_episodes=50)
