import unittest
from unittest.mock import MagicMock, patch
import  numpy as np
from walking_strategy.muscle import Muscle
from walking_strategy.walking_strategy import WalkingStrategy


class TestWalkingStrategy(unittest.TestCase):

    def test_validation(self):
        self.assertRaises(ValueError, WalkingStrategy, muscles = [Muscle(period=200, mode='2D') for _ in range(7)])
        self.assertRaises(ValueError, WalkingStrategy, muscles = [Muscle(period=np.random.choice([120, 160, 200]), mode='2D') for _ in range(11)])
        self.assertRaises(ValueError, WalkingStrategy, fixed_period = 160, muscles = [Muscle(period=200, mode='2D') for _ in range(11)])
        self.assertRaises(ValueError, WalkingStrategy, fixed_type = 'direct', muscles = [Muscle(period=200, mode='2D', type = 'fourier') for _ in range(11)])
        self.assertRaises(ValueError, WalkingStrategy, fixed_sampling_interval = 5, muscles = [Muscle(period=200, mode='2D', sampling_interval = 10) for _ in range(11)])
        self.assertRaises(ValueError, WalkingStrategy, fixed_precision = 10, muscles = [Muscle(period=200, mode='2D', precision = 5) for _ in range(11)])

    def test_no_mutate(self):
        walking_strategy = WalkingStrategy(mode='2D')
        walking_strategy.mutate(mutation_rate=0, period_mutation_rate=None, type_mutation_rate=None, sampling_interval_mutation_rate=None, precision_mutation_rate=None, components_mutation_rate=None, components_mutation_amount=None)
        for _ in range(100):
            self.assertIs(
                walking_strategy.mutate(mutation_rate=0, period_mutation_rate=None, type_mutation_rate=None, sampling_interval_mutation_rate=None, precision_mutation_rate=None, components_mutation_rate=None, components_mutation_amount=None),
                walking_strategy
            )

    def test_full_mutate(self):
        muscles = [Muscle(period=200, mode='2D') for _ in range(11)]
        period_muscles = [Muscle(period=200, mode='2D') for _ in range(11)]
        type_muscles = [Muscle(period=200, mode='2D') for _ in range(11)]
        sampling_interval_muscles = [Muscle(period=200, mode='2D') for _ in range(11)]
        precision_muscles = [Muscle(period=200, mode='2D') for _ in range(11)]
        components_muscles = [Muscle(period=200, mode='2D') for _ in range(11)]

        for i, _ in enumerate(muscles):
            muscles[i].with_period = MagicMock(return_value=period_muscles[i])
            period_muscles[i].mutate_type = MagicMock(return_value=type_muscles[i])
            type_muscles[i].mutate_sampling_interval = MagicMock(return_value=sampling_interval_muscles[i])
            sampling_interval_muscles[i].mutate_precision = MagicMock(return_value=precision_muscles[i])
            precision_muscles[i].mutate_components = MagicMock(return_value=components_muscles[i])

        walking_strategy = WalkingStrategy(muscles=muscles)
        walking_strategy = walking_strategy.mutate(mutation_rate=1, period_mutation_rate=1, type_mutation_rate=1, sampling_interval_mutation_rate=1, precision_mutation_rate=1, components_mutation_rate=1, components_mutation_amount=1)

        for i, _ in enumerate(muscles):
            muscles[i].with_period.assert_called_once()
            period_muscles[i].mutate_type.assert_called_once()
            type_muscles[i].mutate_sampling_interval.assert_called_once()
            sampling_interval_muscles[i].mutate_precision.assert_called_once()
            precision_muscles[i].mutate_components.assert_called_once()

        self.assertEqual(components_muscles, walking_strategy.muscles)

    @patch('numpy.random.choice')
    @patch('numpy.random.randint')
    def test_crossover(self, mock_randint, mock_choice):
        muscles_1 = [Muscle(period=200, mode='2D') for _ in range(11)]
        walking_strategy_1 = WalkingStrategy(mode='2D', muscles=muscles_1)

        muscles_2 = [Muscle(period=240, mode='2D') for _ in range(11)]
        muscles_2_with_period = [Muscle(period=200, mode='2D') for _ in range(8)]
        for i in range(3, 11):
            muscles_2[i].with_period = MagicMock(return_value=muscles_2_with_period[i-3])
        walking_strategy_2 = WalkingStrategy(mode='2D', muscles=muscles_2)

        mock_choice.return_value = walking_strategy_1.period
        mock_randint.return_value = 3

        new_walking_strategy = walking_strategy_1.crossover(walking_strategy_2)
        for i in range(3):
            self.assertEqual(muscles_1[i], new_walking_strategy.muscles[i])
        for i in range(3, 11):
            self.assertEqual(muscles_2_with_period[i-3], new_walking_strategy.muscles[i])

        mock_choice.assert_called_once_with([walking_strategy_1.period, walking_strategy_2.period])
        mock_randint.assert_called_once_with(0, 12)
