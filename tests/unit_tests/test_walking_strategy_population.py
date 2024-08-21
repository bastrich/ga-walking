import unittest
from walking_strategy.walking_strategy import WalkingStrategy
from walking_strategy.walking_strategy_population import WalkingStrategyPopulation


class TestMuscle(unittest.TestCase):

    def test_create_new(self):
        self.assertRaises(ValueError, WalkingStrategyPopulation, '2D', None, None)
        self.assertRaises(ValueError, WalkingStrategyPopulation, None, None, 10)
        self.assertRaises(ValueError, WalkingStrategyPopulation, None, None, None)

        population = WalkingStrategyPopulation('2D', None, 10)

        self.assertEqual(10, len(population.walking_strategies))
        for walking_strategy in population.walking_strategies:
            self.assertIsInstance(walking_strategy, WalkingStrategy)

    def test_create_from_existing_strategies(self):
        walking_strategies = [WalkingStrategy('2D') for _ in range(5)]
        population = WalkingStrategyPopulation(None, walking_strategies, None)

        self.assertEqual(5,len(population.walking_strategies))
        for walking_strategy in population.walking_strategies:
            self.assertIsInstance(walking_strategy, WalkingStrategy)
