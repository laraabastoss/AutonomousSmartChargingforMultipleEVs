from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle
from ev2gym.baselines.mpc.eMPC import eMPC_V2G
from ev2gym.baselines.mpc.ocmf_mpc import OCMF_V2G
from ev2gym.baselines.heuristics import ChargeAsFastAsPossibleToDesiredCapacity

config_file = "ev2gym-config/V2GProfitMax.yaml"

if __name__ == "__main__":
    env = EV2Gym(config_file=config_file,
            save_replay=False,
            save_plots=True, 
            seed=5)
    state, _ = env.reset()
    agent = eMPC_V2G(env, verbose=False) 
    agent_her = ChargeAsFastAsPossibleToDesiredCapacity()
    for t in range(env.simulation_length):
        actions = agent.get_action(env) 
        print(f'Current timestep: {t+1} | Action: {actions}', end='\r')
        new_state, reward, done, truncated, stats = env.step(actions)   
