import copy

from sim.parallel_sim import ParallelSim
import numpy as np
import pickle

from walking_strategy.population_evaluator import PopulationEvaluator
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
MUTABILITY_DECREASE_THRESHOLD = 5
MUTABILITY_INCREASE_THRESHOLD = 10

def update_mutation_parameters(generations_with_improvement, generations_without_improvement, current_mutation_parameters):
    new_mutation_parameters = copy.deepcopy(current_mutation_parameters)

    if generations_with_improvement > MUTABILITY_DECREASE_THRESHOLD:
        print(f'{MUTABILITY_DECREASE_THRESHOLD} generations with improvement, decreasing mutability')
        # shrink_growth_rate += 0.01
        mutation_rate += 0.05
        mutation_amount += 0.05
        # mutation_coefficient += 0.01
        iterations_without_fitness_improvement = 0
    elif generations_without_improvement > MUTABILITY_INCREASE_THRESHOLD:
        print(f'{MUTABILITY_INCREASE_THRESHOLD} generation without improvement, increasing mutability')
        # shrink_growth_rate -= 0.01
        mutation_rate -= 0.05
        mutation_amount -= 0.05
        # mutation_coefficient -= 0.01

    for key in new_mutation_parameters:
        new_mutation_parameters[key] = np.clip(new_mutation_parameters[key], 0, 1)

    return new_mutation_parameters



if POPULATION_FILE_PATH is None:
    population = WalkingStrategyPopulation(size=POPULATION_SIZE)
else:
    with open(POPULATION_FILE_PATH, 'rb') as file:
        population = pickle.load(file)

# required for multiprocessing
if __name__ == "__main__":

    parallel_sim = ParallelSim(mode=MODE, parallelization=PARALLELIZATION)
    population_evaluator = PopulationEvaluator(mode=MODE, parallelization=PARALLELIZATION)

    mutation_parameters = {
        'mutation_rate': 0.8,
        'period_mutation_rate': 0.2,
        'type_mutation_rate': 0.3,
        'sampling_interval_mutation_rate': 0.3,
        'precision_mutation_rate': 0.3,
        'components_mutation_rate': 0.3,
        'components_mutation_amount': 0.3
    }

    best_fitness = -1000
    current_fitness = -1000
    generations_with_improvement = 0
    generations_without_improvement = 0


    for generation in range(NUMBER_OF_GENERATIONS):
        print(f'Starting generation: {generation + 1}/{NUMBER_OF_GENERATIONS}')

        print('Running simulations...')
        simulation_results = parallel_sim.run(population.walking_strategies, SIM_STEPS_PER_GENERATION)
        print('Finished simulations')

        current_fitness = max([simulation_result['fitness'] for simulation_result in simulation_results])
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            generations_with_improvement += 1
            generations_without_improvement = 0
        else:
            generations_with_improvement = 0
            generations_without_improvement += 1

        print('Saving current population to disk...')
        with open(f'results/population', 'wb') as file:
            pickle.dump(population, file)

        print(f'Current fitness: {current_fitness}, Best fitness: {best_fitness}')

        print('Adjusting mutation parameters...')
        mutation_parameters = update_mutation_parameters(generations_with_improvement, generations_without_improvement, mutation_parameters)

        print('Calculating new population...')
        population = population_evaluator.breed_new_population(simulation_results, mutation_parameters)

    print('Finished execution of genetic algorithm')