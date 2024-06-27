from sim_env import SimEnv
import pickle
from walking_strategy import WalkingStrategy
from walking_strategy_population import WalkingStrategyPopulation
from sim import Sim

# visualize the best
# with open('best-0', 'rb') as file:
#     best_walking_strategy = pickle.load(file)

with open('population', 'rb') as file:
    population = pickle.load(file)

best_walking_strategy = population.walking_strategies[0]

sim = Sim(visualize=True)
total_reward, sim_steps = sim.run(best_walking_strategy)

print(f'{total_reward} for {sim_steps} steps')