from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.mpc.V2GProfitMax import (
    V2GProfitMaxOracle,
    V2GProfitMaxLoadsOracle,
)
from ev2gym.baselines.mpc.eMPC import eMPC_V2G
from ev2gym.baselines.mpc.ocmf_mpc import OCMF_V2G
from ev2gym.baselines.heuristics import ChargeAsFastAsPossibleToDesiredCapacity
from line_profiler import profile
from ev2gym.rl_agent.state import V2G_profit_max, V2G_profit_max_loads
from ev2gym.rl_agent.reward import *
from collections import defaultdict
import timeit
import numpy as np

config_file = "ev2gym-config/V2GProfitPlusLoads.yaml"


def run(method=V2GProfitMaxLoadsOracle, seed=5, verbose=False):
    env = EV2Gym(
        config_file=config_file,
        state_function=V2G_profit_max_loads,
        reward_function=profit_maximization,
        save_replay=False,
        save_plots=False,
        seed=seed,
    )
    state, _ = env.reset()
    agent = method(env, verbose=False)

    profit_reward = 0
    for t in range(env.simulation_length):
        actions = agent.get_action(env)
        if verbose:
            print(f"Current timestep: {t + 1} | Action: {actions}", end="\r")
        new_state, reward, done, truncated, stats = env.step(actions)

    return stats


if __name__ == "__main__":
    results = defaultdict(list)
    seeds = [i for i in range(50)]
    methods = [eMPC_V2G, OCMF_V2G]
    metrics = [
        "average_user_satisfaction",
        "total_profits",
        "tracking_error",
        "power_tracker_violation",
        "total_energy_charged",
        "total_energy_discharged",
        "battery_degradation",
        "total_transformer_overload",
    ]

    for seed in seeds:
        start_time = timeit.default_timer()
        stats = run(method=OCMF_V2G, seed=seed)
        end_time = timeit.default_timer()
        for metric in metrics:
            results[metric].append(stats[metric])

        results["time"].append(end_time - start_time)

    for metric, values in results.items():
        print(
            f"Metric: {metric} | Mean: {np.round(np.mean(values), 3)} | STD: {np.round(np.std(values), 3)}"
        )
