from my_osim import L2M2019Env
from sim_env import SimEnv
import pickle
from walking_strategy import WalkingStrategy
from walking_strategy_population import WalkingStrategyPopulation

# visualize the best
# with open('best-0', 'rb') as file:
#     best_walking_strategy = pickle.load(file)

with open('population', 'rb') as file:
    best_walking_strategy = pickle.load(file).walking_strategies[0]

env = SimEnv(visualize=True)
env.reset()

total_reward = 0

for sim_step in range(500):
    observation = env.step(best_walking_strategy.get_muscle_activations(sim_step))
    # total_reward += reward
    # observation, reward, done, info = env.step(env.action_space.sample())

    # observation, reward, done, info = env.step([
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     1,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     0,
    #     1
    # ])
    # if done:
    #     break

print(f'{total_reward} for {sim_step+1} steps')