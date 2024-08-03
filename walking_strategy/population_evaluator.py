from concurrent.futures import ProcessPoolExecutor
import heapq
import numpy as np
from walking_strategy.walking_strategy_population import WalkingStrategyPopulation


class PopulationEvaluator:
    def __init__(self, parallelization):
        self.populations_executor = ProcessPoolExecutor(max_workers=parallelization)

    def breed_new_population(self, simulation_results, mutation_parameters, elites_ratio):
        new_walking_strategies = []

        print('Preserving elites...')
        number_to_preserve = int(np.round(elites_ratio * len(simulation_results)))
        new_walking_strategies += [simulation_result['walking_strategy'] for simulation_result in heapq.nlargest(number_to_preserve, simulation_results, key=lambda simulation_result: simulation_result['fitness'])]

        print('Adjusting fitness values...')
        fits = [simulation_result['fitness'] for simulation_result in simulation_results]
        min_fit = np.min(fits)
        if min_fit < 0:
            fits = fits + np.abs(min_fit)

        print('Executing crossover and mutation...')
        fit_map = fits / np.sum(fits)
        walking_strategies = [simulation_result['walking_strategy'] for simulation_result in simulation_results]
        new_walking_strategies_futures = [
            self.populations_executor.submit(
                self.breed_new_walking_strategy,
                walking_strategies,
                fit_map,
                mutation_parameters['mutation_rate'],
                mutation_parameters['period_mutation_rate'],
                mutation_parameters['type_mutation_rate'],
                mutation_parameters['sampling_interval_mutation_rate'],
                mutation_parameters['precision_mutation_rate'],
                mutation_parameters['components_mutation_rate'],
                mutation_parameters['components_mutation_amount']
            ) for _ in range(len(walking_strategies) - number_to_preserve)
        ]
        new_walking_strategies += [future.result() for future in new_walking_strategies_futures]

        return WalkingStrategyPopulation(walking_strategies=new_walking_strategies)

    @staticmethod
    def breed_new_walking_strategy(walking_strategies, fit_map, mutation_rate, period_mutation_rate, type_mutation_rate, sampling_interval_mutation_rate, precision_mutation_rate, components_mutation_rate, components_mutation_amount):
        parent1, parent2 = np.random.choice(walking_strategies, size=2, p=fit_map)
        new_walking_strategy = parent1.crossover(parent2)
        new_walking_strategy = new_walking_strategy.mutate(mutation_rate, period_mutation_rate, type_mutation_rate, sampling_interval_mutation_rate, precision_mutation_rate, components_mutation_rate, components_mutation_amount)
        return new_walking_strategy

    def shutdown(self):
        if self.populations_executor:
            self.populations_executor.shutdown()
