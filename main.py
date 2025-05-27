# from ev2gym.models.ev2gym_env import EV2Gym
# from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle
# from ev2gym.baselines.gurobi_models.profit_max import V2GProfitMaxOracleGB
# from ev2gym.baselines.heuristics import ChargeAsFastAsPossible
#
# config_file = "ev2gym-config/V2GProfitPlusLoads.yaml"
#
# # Initialize the environment
# env = EV2Gym(config_file=config_file, save_replay=True, save_plots=True)
# state, _ = env.reset()
# new_replay_path = f"replay/replay_{env.sim_name}.pkl"
# agent = V2GProfitMaxOracleGB(new_replay_path, verbose=True)  # optimal solution
# #        or
# agent = ChargeAsFastAsPossible()  # heuristic
# for t in range(env.simulation_length):
#     actions = agent.get_action(env)  # get action from the agent/ algorithm
#     new_state, reward, done, truncated, stats = env.step(actions)  # takes action

"""
This script is used to evaluate the performance of the ev2gym environment.
"""

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.gurobi_models.tracking_error import PowerTrackingErrorrMin

# from ev2gym.baselines.gurobi_models.profit_max import V2GProfitMaxOracleGB
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
from ev2gym.baselines.heuristics import ChargeAsFastAsPossibleToDesiredCapacity

from ev2gym.rl_agent.reward import profit_maximization, SqTrError_TrPenalty_UserIncentives, SquaredTrackingErrorReward

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from supervised_learning import CentralizedDNNPolicy


def run_agent(env, agent, episodes=50):
    episode_rewards = []
    for ep in range(episodes):
        print("Episode: ", ep)
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            actions = agent.get_action(env)
            new_state, reward, done, truncated, stats = env.step(actions)
            total_reward += reward
        episode_rewards.append(total_reward)
        #episode_rewards.append(stats['total_profits'])
    return episode_rewards

def eval():
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
            #print(stats)
            print(f"End of simulation at step {env.current_step}")
            break

    # return
    # Solve optimally
    # Power tracker optimizer
    # agent = PowerTrackingErrorrMin(replay_path=new_replay_path)

    episodes = 10

    # Prepare environment
    env = EV2Gym(
        config_file=config_file,
        verbose=False,
        save_replay=False,
        save_plots=save_plots,
    )

    env.set_reward_function(profit_maximization)

    # --- Run MILP agent ---
    milp_agent = V2GProfitMaxOracleGB(replay_path=new_replay_path, MIPGap=0.0)
    milp_rewards = run_agent(env, milp_agent, episodes=episodes)

    # --- Run SL agent ---
    # Reload env fresh (important to avoid state carryover)
    env = EV2Gym(
        config_file=config_file,
        verbose=False,
        save_replay=False,
        save_plots=save_plots,
    )

    env.set_reward_function(profit_maximization)  # set custom reward if needed

    sl_agent = CentralizedDNNPolicy(
        model_path="centralized_ev_policy.pth",
        input_dim=env.number_of_ports + 3,
        output_dim=env.number_of_ports,
    )
    sl_rewards = run_agent(env, sl_agent, episodes=episodes)

    # --- Plot comparison ---
    plt.figure(figsize=(10,6))
    plt.plot(np.arange(1, episodes+1), milp_rewards, label="MILP Agent")
    plt.plot(np.arange(1, episodes+1), sl_rewards, label="Supervised Learning Agent")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Comparison: MILP vs SL over 50 Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # while True:
    eval()
