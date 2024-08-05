import copy

from sim.parallel_sim import ParallelSim
import numpy as np
import pickle

from walking_strategy.population_evaluator import PopulationEvaluator
from walking_strategy.walking_strategy_population import WalkingStrategyPopulation

import atexit

import time

from itertools import groupby

# configuration options
POPULATION_SIZE = 150
POPULATION_FILE_PATH = 'results/population'
READ_POPULATION_FROM_FILE = False
ANALYTICS_FILE_PATH = 'results/analytics_fitness_function_from_design'
MODE = '2D'  # 3D
PARALLELIZATION = 30
NUMBER_OF_GENERATIONS = 50
SIM_STEPS_PER_GENERATION = 1000
MUTABILITY_DECREASE_THRESHOLD = 5
MUTABILITY_INCREASE_THRESHOLD = 10
ELITES_RATIO = 0.2

def update_mutation_parameters(generations_with_improvement, generations_without_improvement, current_mutation_parameters):
    new_mutation_parameters = copy.deepcopy(current_mutation_parameters)

    if generations_with_improvement >= 1 and generations_with_improvement % MUTABILITY_DECREASE_THRESHOLD == 0:
        print(f'{MUTABILITY_DECREASE_THRESHOLD} generations with improvement, decreasing mutability')

        mutation_parameter_key = np.random.choice(list(filter(lambda key: key != 'mutation_rate', list(current_mutation_parameters.keys()))))
        if mutation_parameter_key == 'period_mutation_rate':
            new_mutation_parameters[mutation_parameter_key] += 0.01
        else:
            new_mutation_parameters[mutation_parameter_key] += 0.02
    elif generations_without_improvement >= 1 and generations_without_improvement % MUTABILITY_INCREASE_THRESHOLD == 0:
        print(f'{MUTABILITY_INCREASE_THRESHOLD} generation without improvement, increasing mutability')

        mutation_parameter_key = np.random.choice(list(filter(lambda key: key != 'mutation_rate', list(current_mutation_parameters.keys()))))
        if mutation_parameter_key == 'period_mutation_rate':
            new_mutation_parameters[mutation_parameter_key] -= 0.01
        else:
            new_mutation_parameters[mutation_parameter_key] -= 0.02

    for key in new_mutation_parameters:
        new_mutation_parameters[key] = np.clip(new_mutation_parameters[key], 0, 1)

    return new_mutation_parameters


if READ_POPULATION_FROM_FILE:
    with open(POPULATION_FILE_PATH, 'rb') as file:
        population = pickle.load(file)
else:
    population = WalkingStrategyPopulation(size=POPULATION_SIZE)

analytics = []

# required for multiprocessing
if __name__ == "__main__":

    parallel_sim = ParallelSim(mode=MODE, parallelization=PARALLELIZATION)
    population_evaluator = PopulationEvaluator(parallelization=PARALLELIZATION)

    def shutdown_hook():
        parallel_sim.shutdown()
        population_evaluator.shutdown()
    atexit.register(shutdown_hook)

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
        simulation_start_time = time.time()
        simulation_results = parallel_sim.run(population.walking_strategies, SIM_STEPS_PER_GENERATION)
        simulation_duration = time.time() - simulation_start_time
        print('Finished simulations')

        print('Saving current population to disk...')
        with open(POPULATION_FILE_PATH, 'wb') as file:
            pickle.dump(population, file)

        best_simulation_result = max(simulation_results, key=lambda simulation_result: simulation_result['fitness'])
        current_fitness = best_simulation_result['fitness']
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            generations_with_improvement += 1
            generations_without_improvement = 0
        else:
            generations_with_improvement = 0
            generations_without_improvement += 1

        periods_distribution = {period: len(list(periods)) for period, periods in groupby(sorted([simulation_result['walking_strategy'].period for simulation_result in simulation_results]))}
        types_distribution = {type: len(list(types)) for type, types in groupby(sorted(np.concatenate([[muscle.type for muscle in simulation_result['walking_strategy'].muscles] for simulation_result in simulation_results])))}
        sampling_intervals_distribution = {sampling_interval: len(list(sampling_intervals)) for sampling_interval, sampling_intervals in groupby(sorted(np.concatenate([[muscle.sampling_interval for muscle in simulation_result['walking_strategy'].muscles] for simulation_result in simulation_results])))}
        precisions_distribution = {precision: len(list(precisions)) for precision, precisions in groupby(sorted(np.concatenate([[muscle.precision for muscle in simulation_result['walking_strategy'].muscles] for simulation_result in simulation_results])))}

        print(f'Current fitness: {current_fitness}, Best fitness: {best_fitness}')

        print('Adjusting mutation parameters...')
        mutation_parameters = update_mutation_parameters(generations_with_improvement, generations_without_improvement, mutation_parameters)

        print('Calculating new population...')
        evaluation_start_time = time.time()
        population = population_evaluator.breed_new_population(simulation_results, mutation_parameters, ELITES_RATIO)
        evaluation_duration = time.time() - evaluation_start_time

        print('Saving analytics to disk...')
        analytics.append({
            'fitness': current_fitness,
            'simulation_duration': np.round(simulation_duration),
            'evaluation_duration': np.round(evaluation_duration),
            'walking_stability': best_simulation_result['steps'],
            'distance': best_simulation_result['distance'],
            'energy': best_simulation_result['energy'],
            'periods_distribution': periods_distribution,
            'types_distribution': types_distribution,
            'sampling_intervals_distribution': sampling_intervals_distribution,
            'precisions_distribution': precisions_distribution
        })
        with open(ANALYTICS_FILE_PATH, 'wb') as file:
            pickle.dump(analytics, file)


    print('Finished execution of genetic algorithm')