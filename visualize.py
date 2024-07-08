
import pickle
from walking_strategy import WalkingStrategy
from walking_strategy_population import WalkingStrategyPopulation
from sim import Sim
import numpy as np

# visualize the best
# with open('best-0', 'rb') as file:
#     best_walking_strategy = pickle.load(file)

with open('population', 'rb') as file:
    population = pickle.load(file)
#
# fitness_values = np.array([ws.evaluated_fitness for ws in population.walking_strategies])
# min_fitness = np.min(fitness_values)
# if min_fitness < 0:
#     fitness_values = fitness_values + np.abs(min_fitness)
#
# fit_map = fitness_values / np.sum(fitness_values)
#
# parent1, parent2 = np.random.choice(population.walking_strategies, size=2, p=fit_map)
#
# best_walking_strategy = parent1.crossover(parent2)

# with open(f'test', 'wb') as file:
#     pickle.dump(best_walking_strategy, file)

# with open('test', 'rb') as file:
#     best_walking_strategy = pickle.load(file)
#
# #
# best_walking_strategy = best_walking_strategy.mutate(0.3, 0.3).mutate(0.3, 0.3).mutate(0.3, 0.3).mutate(0.3, 0.3).mutate(0.3, 0.3)

best_walking_strategy = population.walking_strategies[0]
# best_walking_strategy = best_walking_strategy.mutate(0.3, 0.3).mutate(0.3, 0.3).mutate(0.3, 0.3).mutate(0.3, 0.3).mutate(0.3, 0.3)



sim = Sim(mode='2D', visualize=True)
total_reward, sim_steps = sim.run(best_walking_strategy)

print(f'{total_reward} for {sim_steps} steps')
print(f'expected {best_walking_strategy.evaluated_fitness}')