import pickle
from sim.sim import Sim

# read evolved population
with open('results/population_3d', 'rb') as file:
    population = pickle.load(file)

# select the best walking strategy
best_walking_strategy = population.walking_strategies[0]

# run visual simulation
sim = Sim(mode='3D', visualize=True)
fitness, sim_steps, _, _ = sim.run(best_walking_strategy)

print(f'{fitness} for {sim_steps} steps')