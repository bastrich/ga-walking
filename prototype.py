from osim.env import L2M2019Env
import numpy as np
import copy
import pickle

# TODO: we should exploit the Fourier property for which higher harmonics weights tend to decays as 1/x^n for smooth and continous functions

class WalkingStrategy:
    def __init__(self, period):
        self.muscle_activations_fourier_coefficients = [WalkingStrategy.generate_single_muscle_activation() for _ in range(22)]
        self.period = period

    @staticmethod
    def generate_single_muscle_activation():
        result = []

        a0 = np.random.uniform(-10, 10)
        result.append(a0)

        current_sum = a0
        current_negative_sum = a0

        for _ in range(9):
            coefficient = np.random.uniform(-10, min(1 - current_sum, current_negative_sum))

            result.append(coefficient)

            current_sum += coefficient
            current_negative_sum -= coefficient

        return result

    def calculate_muscle_activations(self, time):
        return [self.calculate_fourier_series_sum(coefficients, time) for coefficients in self.muscle_activations_fourier_coefficients]

    def calculate_fourier_series_sum(self, coefficients, time):
        a0 = coefficients[0]
        result = a0

        for i in range(1, len(coefficients), 3):
            an = coefficients[i]
            bn = coefficients[i + 1]
            phase = coefficients[i + 2]
            result += an * np.cos((i // 3 + 1) * 2 * np.pi * time / self.period + phase) + bn * np.cos(
                (i // 3 + 1) * 2 * np.pi * time / self.period + phase)

        return result

def crossover(walking_strategy_1, walking_strategy_2):
    return walking_strategy_1


def mutate(walking_strategy):
    for i in range(9):
        w[i] += np.random.randn(8) * alpha
    """
    for i in range(9):
        delta=[]
        for j in range(4):
            delta.append(np.random.randn()*0.5/((j+1)**2))
        delta=np.asarray(delta)
        w[i]+=delta
    """

    return walking_strategy

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

iterations = 200
sim_steps_per_iteration = 1000

env = L2M2019Env(visualize=False, difficulty=0)

population = WalkingStrategyPopulation(size=10)

for iteration in range(iterations):
    print(f'Iteration: {iteration + 1}/{iterations}')

    # # if it doens't get better for more than 10 iterations, increase alpha to allow bigger changes
    # # Increase alpha then set unev_runs back to 0
    # if unev_runs > 30:
    #     print("Augmenting alpha")
    #     alpha += alpha_0
    #     unev_runs = 0

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
# working with pickle
# fileObject = open(file_Name, 'r')
# w_best = pickle.load(fileObject)
# pickle.dump(w_best,fileObject)

# visualize the best
# fileObject = open(file_Name, 'r')
# w_best = pickle.load(fileObject)
# pickle.dump(w_best,fileObject)