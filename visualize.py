from my_osim import L2M2019Env
import pickle
from walking_strategy import WalkingStrategy
from walking_strategy_population import WalkingStrategyPopulation

# visualize the best
# with open('best-0', 'rb') as file:
#     best_walking_strategy = pickle.load(file)

with open('population', 'rb') as file:
    best_walking_strategy = pickle.load(file).walking_strategies[1]

env = L2M2019Env(visualize=True, difficulty=0)
env.reset()

reward = 0

for sim_step in range(10000):
    observation, reward, done, info = env.step(best_walking_strategy.get_muscle_activations(sim_step))
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
    if done:
        break

print(reward)