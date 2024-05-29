from osim.env import L2M2019Env
import numpy as np
import copy
import pickle
import random
from walking_strategy import WalkingStrategy

from concurrent.futures import ProcessPoolExecutor

import time

period = 400

def crossover(walking_strategy_1, walking_strategy_2):
    switch_indexes = np.random.randint(1, 175, np.random.randint(1, 5))
    switch = True
    new_muscle_activations_fourier_coefficients = []

    for i in range(176):
        if i in switch_indexes:
            switch = not switch
        if switch is True:
            new_muscle_activations_fourier_coefficients.append(walking_strategy_1.muscle_activations_fourier_coefficients[i // 16][i % 16])
        else:
            new_muscle_activations_fourier_coefficients.append(walking_strategy_2.muscle_activations_fourier_coefficients[i // 16][i % 16])

    new_muscle_activations_fourier_coefficients = np.array_split(new_muscle_activations_fourier_coefficients, 11)

    return WalkingStrategy(period, new_muscle_activations_fourier_coefficients)


class WalkingStrategyPopulation:
    def __init__(self, **kwargs):
        walking_strategies = kwargs.get('walking_strategies')
        if walking_strategies is not None:
            self.walking_strategies = walking_strategies
            return

        size = kwargs.get('size')
        if size is not None:
            self.walking_strategies = [WalkingStrategy(period) for _ in range(size)]
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


iterations = 10000
sim_steps_per_iteration = 1000

population = WalkingStrategyPopulation(size=15)

envs = [L2M2019Env(visualize=False, difficulty=0) for _ in range(len(population.walking_strategies))]

def evaluate(i, walking_strategy):
    global envs
    envs[i].reset()
    total_reward = 0
    for sim_step in range(sim_steps_per_iteration):
        observation, reward, done, info = envs[i].step(walking_strategy.get_muscle_activations(sim_step))
        total_reward += reward
        if done:
            break
    return total_reward


if __name__ == "__main__":

    def mutate(walking_strategy, shrink_growth_rate, mutation_rate, mutation_coefficient):
        new_muscle_activations_fourier_coefficients = copy.deepcopy(walking_strategy.muscle_activations_fourier_coefficients)

        for i in range(11):
            for j in range(16):
                if random.random() < shrink_growth_rate:
                    new_muscle_activations_fourier_coefficients[i][j] = random.choice([0, 1])
                elif random.random() < mutation_rate:
                    new_muscle_activations_fourier_coefficients[i][j] += mutation_coefficient * np.random.normal()

        return WalkingStrategy(period, new_muscle_activations_fourier_coefficients)

    executor = ProcessPoolExecutor(max_workers=len(population.walking_strategies))

    shrink_growth_rate = 0.01
    mutation_rate = 0.01
    mutation_coefficient = 0.01

    total_best_fitness_value = 0
    current_best_fitness_value = 0

    iterations_with_fitness_improvement = 0
    iterations_without_fitness_improvement = 0

    start_time = time.time()

    for iteration in range(iterations):
        print(f'Last fitness: {current_best_fitness_value}, Best fitness: {total_best_fitness_value}')
        print(f'Iteration: {iteration + 1}/{iterations}')

        # eval population

        futures = [executor.submit(evaluate, i, walking_strategy) for i, walking_strategy in enumerate(population.walking_strategies)]
        fitness_values = np.array([future.result() for future in futures])

        current_best_fitness_value = fitness_values.max()
        if current_best_fitness_value > total_best_fitness_value:
            total_best_fitness_value = current_best_fitness_value
            iterations_without_fitness_improvement = 0
            iterations_with_fitness_improvement += 1
        else:
            iterations_with_fitness_improvement = 0
            iterations_without_fitness_improvement += 1

        if iterations_without_fitness_improvement > 5:
            print('5 generations without improvement, increasing mutation rate')
            shrink_growth_rate += 0.01
            mutation_rate += 0.01
            mutation_coefficient += 0.01
            iterations_without_fitness_improvement = 0
        elif iterations_with_fitness_improvement > 0:
            print('1 generation with improvement, decreasing mutation rate')
            shrink_growth_rate -= 0.01
            mutation_rate -= 0.01
            mutation_coefficient -= 0.01

        shrink_growth_rate = np.clip(shrink_growth_rate, 0.01, 0.2)
        mutation_rate = np.clip(mutation_rate, 0.01, 0.5)
        mutation_coefficient = np.clip(mutation_coefficient, 0.01, 4)

        # give a birth to a new population
        fit_map = population.get_fitness_map(fitness_values)
        new_walking_strategies = []
        for _ in range(len(population.walking_strategies)):
            parent1 = population.select_parent(fit_map)
            parent2 = population.select_parent(fit_map)
            new_walking_strategy = crossover(parent1, parent2)

            new_walking_strategy = mutate(new_walking_strategy, shrink_growth_rate, mutation_rate, mutation_coefficient)

            new_walking_strategies.append(new_walking_strategy)

        # preserve elites
        max_fits = -np.partition(-fitness_values, 1)[:1]
        elites_saved = 0
        for i, walking_strategy in enumerate(population.walking_strategies):
            if fitness_values[i] in max_fits:
                new_walking_strategies[elites_saved] = walking_strategy
                # save the best
                with open(f'best-{elites_saved}', 'wb') as file:
                    pickle.dump(walking_strategy, file)
                elites_saved += 1

            if elites_saved == 1:
                break

        population = WalkingStrategyPopulation(walking_strategies=new_walking_strategies)

    end_time = time.time()

    print(f'Execution time: {(end_time - start_time) / 60} minutes')

    executor.shutdown()

