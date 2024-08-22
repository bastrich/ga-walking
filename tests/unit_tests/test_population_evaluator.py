import unittest
from walking_strategy.population_evaluator import PopulationEvaluator
import numpy as np

# unittest MagicMock doesn't work in multiprocessing, so I need my own implementation of mock
class MockedWalkingStrategy:

    def __init__(self, mutate_should_be_called, crossover_should_be_called):
        self.id = np.random.randint(0, 100000)
        self.mutate_should_be_called = mutate_should_be_called
        self.mutate_was_called = False
        self.crossover_should_be_called = crossover_should_be_called
        self.crossover_was_called = False
        self.crossover_result = None


    def mutate(self, mutation_rate, period_mutation_rate, type_mutation_rate, sampling_interval_mutation_rate, precision_mutation_rate, components_mutation_rate, components_mutation_amount):
        assert self.mutate_should_be_called
        self.mutate_was_called = True

        assert mutation_rate == 0.1
        assert period_mutation_rate == 0.2
        assert type_mutation_rate == 0.3
        assert sampling_interval_mutation_rate == 0.4
        assert precision_mutation_rate == 0.5
        assert components_mutation_rate == 0.6
        assert components_mutation_amount == 0.7

        return self

    def crossover(self, other):
        assert self.crossover_should_be_called
        self.crossover_was_called = True
        assert other is self
        return self.crossover_result

class TestPopulationEvaluator(unittest.TestCase):

    def test_breeding(self):
        population_evaluator = PopulationEvaluator(parallelization=1)
        mutation_parameters = {
            'mutation_rate': 0.1,
            'period_mutation_rate': 0.2,
            'type_mutation_rate': 0.3,
            'sampling_interval_mutation_rate': 0.4,
            'precision_mutation_rate': 0.5,
            'components_mutation_rate': 0.6,
            'components_mutation_amount': 0.7
        }

        crossover_result = MockedWalkingStrategy(mutate_should_be_called=True, crossover_should_be_called=False)

        simulation_results = [{'walking_strategy': MockedWalkingStrategy(mutate_should_be_called=False, crossover_should_be_called=False), 'fitness': -1 } for i in range(10)]
        simulation_results[7]['fitness'] = 1
        simulation_results[8]['fitness'] = 10000000
        simulation_results[9]['fitness'] = 2
        simulation_results[8]['walking_strategy'].crossover_should_be_called = True
        simulation_results[8]['walking_strategy'].crossover_result = crossover_result

        new_population = population_evaluator.breed_new_population(simulation_results, mutation_parameters, 0.3)

        self.assertIs(simulation_results[8]['walking_strategy'], new_population.walking_strategies[0])
        self.assertIs(simulation_results[9]['walking_strategy'], new_population.walking_strategies[1])
        self.assertIs(simulation_results[7]['walking_strategy'], new_population.walking_strategies[2])
        for i in range(3, 10):
            self.assertEqual(crossover_result.id, new_population.walking_strategies[i].id)