import pickle
from sim.sim import Sim

with open('results/population_2d', 'rb') as file:
    population = pickle.load(file)

best_walking_strategy = population.walking_strategies[0]

sim = Sim(mode='2D', visualize=True)
fitness, sim_steps, _, _ = sim.run(best_walking_strategy)

print(f'{fitness} for {sim_steps} steps')