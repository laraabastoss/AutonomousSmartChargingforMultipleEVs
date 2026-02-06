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

from utils import *

from supervised_learning import CentralizedDNNPolicy

from generate_sa_pairs import DATASET_NAME

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
            print(f"Actions: {actions}")
            actions_per_timestep.append(actions.copy())
            _, reward, done, _, stats = env.step(actions)
            total_reward += reward
        episode_stats.append(stats)
        if return_actions:
            episode_actions.append(np.array(actions_per_timestep))
    return (episode_stats, episode_actions) if return_actions else episode_stats


def eval(num_episodes=5, save_plots=True):
    config_file = "ev2gym-config/V2GProfitPlusLoadsGenerateData.yaml"
    dummy_agent_class = ChargeAsFastAsPossible

    milp_all_stats = []
    sl_all_stats = []

    input_dim = np.load(f"./datasets/{DATASET_NAME}.npz")['states'].shape[1]
    output_dim = np.load(f"./datasets/{DATASET_NAME}.npz")['actions'].shape[1]


    for ep in range(num_episodes):
        print(f"\n=== Episode {ep+1}/{num_episodes} ===")


        # 1) Generate replay with dummy agent

        env_replay_gen = EV2Gym(
            config_file=config_file,
            load_from_replay_path=None,
            verbose=False,
            save_replay=True,
            save_plots=False,
            seed=ep  
        )

        _, _ = env_replay_gen.reset()

        dummy_agent = dummy_agent_class()
        done = False
        while not done:
            actions = dummy_agent.get_action(env_replay_gen)
            _, _, done, _, _ = env_replay_gen.step(actions)

        replay_path = f"replay/replay_{env_replay_gen.sim_name}.pkl"

        # 2) Run MILP agent on this replay

        env_milp = EV2Gym(
            config_file=config_file,
            load_from_replay_path=replay_path,
            verbose=False,
            save_replay=False,
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
            print(stats)
            print(f"End of simulation at step {env.current_step}")
            break

    # return
    # Solve optimally
    # Power tracker optimizer
    # agent = PowerTrackingErrorrMin(replay_path=new_replay_path)
    # # Profit maximization optimizer
    agent = V2GProfitMaxOracleGB(replay_path=new_replay_path, MIPGap=0.0)
    # # Simulate in the gym environment and get the rewards

    env = EV2Gym(
        config_file=config_file,
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
            actions, visualize=False
        )  # takes action
        rewards_opt.append(reward)

        # if verbose:
        #     print(f'Reward: {reward} \t Done: {done}')

        if done:
            print(stats)
            break
        env_milp.set_reward_function(milp_objective)
        milp_agent = V2GProfitMaxOracleGB(replay_path=replay_path, MIPGap=0.0)
        milp_stats, milp_actions = run_agent(env_milp, milp_agent, episodes=1, return_actions=True)
        milp_all_stats.extend(milp_stats)

        # 3) SL

        env_sl = EV2Gym(
            config_file=config_file,
            load_from_replay_path=replay_path,
            verbose=False,
            save_replay=False,
            save_plots=save_plots,
        )
        env_sl.set_reward_function(milp_objective)
        sl_agent = CentralizedDNNPolicy(
            model_path=f"./models/{DATASET_NAME}_ev_policy.pth",
            input_dim=input_dim,
            output_dim=output_dim
        )
        sl_stats, sl_actions = run_agent(env_sl, sl_agent, episodes=1, return_actions=True)

        plot_action_histogram(actions=milp_actions[0], episode_idx=ep, agent_name="MILP", save_dir=f"./plots/{DATASET_NAME}/action_plots")
        plot_action_histogram(actions=sl_actions[0], episode_idx=ep, agent_name="SL", save_dir=f"./plots/{DATASET_NAME}/action_plots")


        plot_actions_per_episode(
            milp_actions=milp_actions[0],
            sl_actions=sl_actions[0],
            episode_idx=ep,
            save_dir=f"./plots/{DATASET_NAME}/action_plots"
        )

        sl_all_stats.append({
            "no_gru": sl_stats[0]
        })


    # 4) Plot and save per-episode comparisons
    for i in range(num_episodes):
        plot_agent_stats_per_episode(
            milp_stat=milp_all_stats[i],
            sl_stat_dict=sl_all_stats[i],
            episode_idx=i,
            save_dir=f"./plots/{DATASET_NAME}/all_stats"
        )


    # --- Plot comparison ---
    milp_total_profits = [float(np.array(stat['total_profits']).item()) for stat in milp_all_stats]
    sl_total_profits = [float(np.array(stat['no_gru']['total_profits']).item()) for stat in sl_all_stats]



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
    eval(num_episodes=20)
