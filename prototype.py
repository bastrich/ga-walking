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
sim_steps_per_iteration = 1000

population = WalkingStrategyPopulation(period, size=150)
# with open('population', 'rb') as file:
#     population = pickle.load(file)

sims = [Sim(visualize=False) for _ in range(len(population.walking_strategies))]

def evaluate(i, walking_strategy):
    global sims
    fitness, steps = sims[i].run(walking_strategy, sim_steps_per_iteration)
    return walking_strategy, np.round(fitness, 2)

if __name__ == "__main__":

    executor = ProcessPoolExecutor(max_workers=30)

    # shrink_growth_rate = 0.01
    mutation_rate = 0.5
    mutation_amount = 1.5
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

        futures = [executor.submit(evaluate, i, walking_strategy) for i, walking_strategy in enumerate(population.walking_strategies)]
        simulation_results = [future.result() for future in futures]
        fitness_values = np.array([simulation_result[1] for simulation_result in simulation_results])

        current_best_fitness_value = fitness_values.max()
        if current_best_fitness_value > total_best_fitness_value + 0.01:
            iterations_without_fitness_improvement = 0
            iterations_with_fitness_improvement += 1
        elif current_best_fitness_value <= total_best_fitness_value:
            iterations_with_fitness_improvement = 0
            iterations_without_fitness_improvement += 1
        if current_best_fitness_value > total_best_fitness_value:
            total_best_fitness_value = current_best_fitness_value

        if iterations_without_fitness_improvement > 10:
            print('30 generations without improvement, increasing mutation rate')
            # shrink_growth_rate += 0.01
            mutation_rate += 0.01
            mutation_amount += 0.1
            # mutation_coefficient += 0.01
            iterations_without_fitness_improvement = 0
        elif iterations_with_fitness_improvement > 2:
            print('1 generation with improvement, decreasing mutation rate')
            # shrink_growth_rate -= 0.01
            mutation_rate -= 0.01
            mutation_amount -= 0.1
            # mutation_coefficient -= 0.01


        # give a birth to a new population
        new_walking_strategies = []

        min_fitness = np.min(fitness_values)
        if min_fitness < 0:
            fitness_values = fitness_values + np.abs(min_fitness)



        # preserve elites
        number_to_preserve = 15
        preserved_walking_strategies = [simulation_result[0] for simulation_result in heapq.nlargest(number_to_preserve, simulation_results, key=lambda simulation_result: simulation_result[1])]
        new_walking_strategies += preserved_walking_strategies

        # shrink_growth_rate = np.clip(shrink_growth_rate, 0.01, 0.1)
        mutation_rate = np.clip(mutation_rate, 0.01, 0.5)
        mutation_amount = np.clip(mutation_amount, 0.1, 3)
        # mutation_coefficient = np.clip(mutation_coefficient, 0.1, 5)

        fit_map = population.get_fitness_map(fitness_values)

        for _ in range(len(population.walking_strategies) - number_to_preserve):
            parent1, parent2 = population.select_parents(fit_map)
            new_walking_strategy = parent1.crossover(parent2)

            new_walking_strategy = new_walking_strategy.mutate(mutation_rate, mutation_amount)

            new_walking_strategies.append(new_walking_strategy)

        population = WalkingStrategyPopulation(period, walking_strategies=new_walking_strategies)
        # save current population
        with open(f'population', 'wb') as file:
            pickle.dump(population, file)

    end_time = time.time()

    print(f'Execution time: {(end_time - start_time) / 60} minutes')

    executor.shutdown()

