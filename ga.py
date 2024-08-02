from sim.parallel_sim import ParallelSim
import numpy as np
import pickle
from walking_strategy.walking_strategy_population import WalkingStrategyPopulation
from concurrent.futures import ProcessPoolExecutor
import heapq
import time

# configuration options
POPULATION_SIZE = 150
POPULATION_FILE_PATH = None  # 'results/population'
MODE = '2D'  # 3D
PARALLELIZATION = 30
NUMBER_OF_GENERATIONS = 100
SIM_STEPS_PER_GENERATION = 1000




if POPULATION_FILE_PATH is None:
    population = WalkingStrategyPopulation(size=POPULATION_SIZE)
else:
    with open(POPULATION_FILE_PATH, 'rb') as file:
        population = pickle.load(file)


def give_birth_to_new_walking_strategy(walking_strategies, fit_map, mutation_rate, mutation_amount):
    parent1, parent2 = np.random.choice(walking_strategies, size=2, p=fit_map)
    new_walking_strategy = parent1.crossover(parent2)
    new_walking_strategy = new_walking_strategy.mutate(mutation_rate, mutation_amount)
    return new_walking_strategy

# required for multiprocessing
if __name__ == "__main__":

    parallel_sim = ParallelSim(mode=MODE, parallelization=PARALLELIZATION)

    mutation_rate = 0.8
    period_mutation_rate = 0.2
    type_mutation_rate = 0.3
    sampling_interval_mutation_rate = 0.3
    precision_mutation_rate = 0.3
    components_mutation_rate = 0.3
    components_mutation_amount = 0.3


    populations_executor = ProcessPoolExecutor(max_workers=30)


    total_best_fitness_value = -1000
    current_best_fitness_value = -1000

    iterations_with_fitness_improvement = 0
    iterations_without_fitness_improvement = 0


    for generation in range(NUMBER_OF_GENERATIONS):
        print(f'Last fitness: {current_best_fitness_value}, Best fitness: {total_best_fitness_value}')
        print(f'Iteration: {iteration + 1}/{iterations}')

        # eval population

        print('Running simulations...')

        simulation_results = parallel_sim.run(population.walking_strategies, SIM_STEPS_PER_GENERATION)

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
        mutation_rate = np.clip(mutation_rate, 0.05, 1)
        # mutation_amount = np.clip(mutation_amount, 0.1, 3)
        mutation_amount = np.clip(mutation_amount, 0.05, 1)
        # mutation_coefficient = np.clip(mutation_coefficient, 0.1, 5)

        # preserve elites with mutation
        # preserved_walking_strategies_with_mutation = [walking_strategy.mutate(mutation_rate, mutation_amount) for walking_strategy in preserved_walking_strategies]
        # new_walking_strategies += preserved_walking_strategies_with_mutation

        fit_map = fitness_values / np.sum(fitness_values)

        print('Creating new population...')

        new_walking_strategies_futures = [populations_executor.submit(give_birth_to_new_walking_strategy, population.walking_strategies, fit_map, mutation_rate, mutation_amount) for _ in range(len(population.walking_strategies) - number_to_preserve)]
        new_walking_strategies += [future.result() for future in new_walking_strategies_futures]

        population = WalkingStrategyPopulation(walking_strategies=new_walking_strategies)

        # save current population
        with open(f'results/population', 'wb') as file:
            pickle.dump(population, file)



    end_time = time.time()

    print(f'Execution time: {(end_time - start_time) / 60} minutes')

    populations_executor.shutdown()

