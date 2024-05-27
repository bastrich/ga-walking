from osim.env import L2M2019Env
import numpy as np
import copy
import pickle
import random
from walking_strategy import WalkingStrategy

def crossover(walking_strategy_1, walking_strategy_2):
    switch_index = random.randint(0, 220)

    fourier_1 = [
        v
        for muscle in walking_strategy_1.muscle_activations_fourier_coefficients
        for v in muscle
    ]

    fourier_2 = [
        v
        for muscle in walking_strategy_2.muscle_activations_fourier_coefficients
        for v in muscle
    ]

    new_muscle_activations_fourier_coefficients = np.array_split(np.concatenate((fourier_1[:switch_index], fourier_2[switch_index:])), 22)

    return WalkingStrategy(100, new_muscle_activations_fourier_coefficients)


def mutate(walking_strategy):
    #TODO: add dynamic mutation factor when there are no improvement for a long time

    new_muscle_activations_fourier_coefficients = copy.deepcopy(walking_strategy.muscle_activations_fourier_coefficients)

    for i in range(10):
        for j in range(new_muscle_activations_fourier_coefficients[i].shape[0]):
            if random.random() < 0.05:
                new_muscle_activations_fourier_coefficients[i][j] += np.random.randn()

    """
    for i in range(9):
        delta=[]
        for j in range(4):
            delta.append(np.random.randn()*0.5/((j+1)**2))
        delta=np.asarray(delta)
        w[i]+=delta
    """

    return WalkingStrategy(100, new_muscle_activations_fourier_coefficients)

class WalkingStrategyPopulation:
    def __init__(self, **kwargs):
        walking_strategies = kwargs.get('walking_strategies')
        if walking_strategies is not None:
            self.walking_strategies = walking_strategies
            return

        size = kwargs.get('size')
        if size is not None:
            self.walking_strategies = [WalkingStrategy(100) for _ in range(size)]
            return

        raise Exception('Wrong arguments')

    @staticmethod
    def get_fitness_map(fitness_values):
        fitmap = []
        total = 0
        for f in fitness_values:
            total = total + f
            fitmap.append(total)
        return fitmap

    def select_parent(self, fitmap):
        r = np.random.rand()  # 0-1
        r = r * fitmap[-1]
        for i in range(len(fitmap)):
            if r <= fitmap[i]:
                return self.walking_strategies[i]

        return self.walking_strategies[len(fitmap) - 1]


# testing
# walking_strategy = WalkingStrategy(100)
# env = L2M2019Env(visualize=True)
# env.reset()
# for i in range(1000):
#     env.step(walking_strategy.calculate_muscle_activations(i))

iterations = 15000
sim_steps_per_iteration = 1000

env = L2M2019Env(visualize=False, difficulty=0)

population = WalkingStrategyPopulation(size=10)

for iteration in range(iterations):
    print(f'Iteration: {iteration + 1}/{iterations}')

    env.reset()

    fitness_values = np.array([0 for _ in range(len(population.walking_strategies))])

    # eval population
    for i, walking_strategy in enumerate(population.walking_strategies):
        for sim_step in range(sim_steps_per_iteration):
            observation, reward, done, info = env.step(walking_strategy.calculate_muscle_activations(sim_step))
            fitness_values[i] += reward
            if done:
                break

    # give a birth to a new population
    fit_map = population.get_fitness_map(fitness_values)
    new_walking_strategies = []
    for _ in range(len(population.walking_strategies)):
        parent1 = population.select_parent(fit_map)
        parent2 = population.select_parent(fit_map)
        new_walking_strategy = crossover(parent1, parent2)

        new_walking_strategy = mutate(new_walking_strategy)

        new_walking_strategy.normalize()

        new_walking_strategies.append(new_walking_strategy)

    # preserve elites
    max_fits = -np.partition(-fitness_values, 2)[:2]
    elites_saved = 0
    for i, walking_strategy in enumerate(population.walking_strategies):
        if fitness_values[i] in max_fits:
            new_walking_strategies[elites_saved] = walking_strategy
            elites_saved += 1

        if elites_saved == 2:
            break

    population = WalkingStrategyPopulation(walking_strategies=new_walking_strategies)

# save the best
with open('best', 'wb') as file:
    pickle.dump(population.walking_strategies[0], file)

# visualize the best
with open('best', 'rb') as file:
    best_walking_strategy = pickle.load(file)

env = L2M2019Env(visualize=True, difficulty=0)
env.reset()
for sim_step in range(sim_steps_per_iteration):
    observation, reward, done, info = env.step(best_walking_strategy.calculate_muscle_activations(sim_step))
    if done:
        break