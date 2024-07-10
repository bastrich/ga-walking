from sim import Sim
import numpy as np
import copy
import pickle
import random
from walking_strategy import WalkingStrategy
from walking_strategy_population import WalkingStrategyPopulation

from concurrent.futures import ProcessPoolExecutor

import heapq

import time

period = 200


iterations = 10000
sim_steps_per_iteration = 500

population = WalkingStrategyPopulation(period, size=150)
# with open('population', 'rb') as file:
#     population = pickle.load(file)

# for walking_strategy in population.walking_strategies:
#     walking_strategy.change_precision(5)

sims = [Sim('2D', False) for _ in range(len(population.walking_strategies))]

def evaluate(i, walking_strategy):
    global sims
    fitness, steps = sims[i].run(walking_strategy, sim_steps_per_iteration)
    walking_strategy.evaluated_fitness = np.round(fitness, 2)
    return walking_strategy
    # if doesn't bacame worse, allow save new mutation'


def give_birth_to_new_walking_strategy(walking_strategies, fit_map, mutation_rate, mutation_amount):
    parent1, parent2 = np.random.choice(walking_strategies, size=2, p=fit_map)
    new_walking_strategy = parent1.crossover(parent2)
    new_walking_strategy = new_walking_strategy.mutate(mutation_rate, mutation_amount)
    return new_walking_strategy

if __name__ == "__main__":

    simulations_executor = ProcessPoolExecutor(max_workers=30)
    populations_executor = ProcessPoolExecutor(max_workers=30)

    # shrink_growth_rate = 0.01
    mutation_rate = 0.3
    mutation_amount = 0.3
    # mutation_coefficient = 0.01

    total_best_fitness_value = -1000
    current_best_fitness_value = -1000

    iterations_with_fitness_improvement = 0
    iterations_without_fitness_improvement = 0

    start_time = time.time()

    for iteration in range(iterations):
        print(f'Last fitness: {current_best_fitness_value}, Best fitness: {total_best_fitness_value}')
        print(f'Iteration: {iteration + 1}/{iterations}')

        # eval population

        print('Running simulations...')

        futures = [simulations_executor.submit(evaluate, i, walking_strategy) for i, walking_strategy in enumerate(population.walking_strategies)]
        walking_strategies = [future.result() for future in futures]
        fitness_values = np.array([walking_strategy.evaluated_fitness for walking_strategy in walking_strategies])

        # save current population
        with open(f'population', 'wb') as file:
            pickle.dump(population, file)

        current_best_fitness_value = fitness_values.max()
        if current_best_fitness_value > total_best_fitness_value + 0.01:
            iterations_without_fitness_improvement = 0
            iterations_with_fitness_improvement += 1
        elif current_best_fitness_value <= total_best_fitness_value:
            iterations_with_fitness_improvement = 0
            iterations_without_fitness_improvement += 1
        if current_best_fitness_value > total_best_fitness_value:
            total_best_fitness_value = current_best_fitness_value

        if iterations_without_fitness_improvement > 5:
            print('30 generations without improvement, increasing mutation rate')
            # shrink_growth_rate += 0.01
            mutation_rate += 0.05
            mutation_amount += 0.05
            # mutation_coefficient += 0.01
            iterations_without_fitness_improvement = 0
        elif iterations_with_fitness_improvement > 2:
            print('1 generation with improvement, decreasing mutation rate')
            # shrink_growth_rate -= 0.01
            mutation_rate -= 0.05
            mutation_amount -= 0.05
            # mutation_coefficient -= 0.01


        # give a birth to a new population
        new_walking_strategies = []

        min_fitness = np.min(fitness_values)
        if min_fitness < 0:
            fitness_values = fitness_values + np.abs(min_fitness)

        print('Selecting elites...')

        # preserve elites
        number_to_preserve = 20
        preserved_walking_strategies = [walking_strategy for walking_strategy in heapq.nlargest(number_to_preserve, walking_strategies, key=lambda walking_strategy: walking_strategy.evaluated_fitness)]
        # np.random.shuffle(preserved_walking_strategies)
        new_walking_strategies += preserved_walking_strategies

        # shrink_growth_rate = np.clip(shrink_growth_rate, 0.01, 0.1)
        mutation_rate = np.clip(mutation_rate, 0.05, 1.05)
        # mutation_amount = np.clip(mutation_amount, 0.1, 3)
        mutation_amount = np.clip(mutation_amount, 0.05, 1.05)
        # mutation_coefficient = np.clip(mutation_coefficient, 0.1, 5)

        # preserve elites with mutation
        # preserved_walking_strategies_with_mutation = [walking_strategy.mutate(mutation_rate, mutation_amount) for walking_strategy in preserved_walking_strategies]
        # new_walking_strategies += preserved_walking_strategies_with_mutation

        fit_map = fitness_values / np.sum(fitness_values)

        print('Creating new population...')

        new_walking_strategies_futures = [populations_executor.submit(give_birth_to_new_walking_strategy, population.walking_strategies, fit_map, mutation_rate, mutation_amount) for _ in range(len(population.walking_strategies) - number_to_preserve)]
        new_walking_strategies += [future.result() for future in new_walking_strategies_futures]

        population = WalkingStrategyPopulation(period, walking_strategies=new_walking_strategies)



    end_time = time.time()

    print(f'Execution time: {(end_time - start_time) / 60} minutes')

    simulations_executor.shutdown()
    populations_executor.shutdown()

