import unittest
from unittest.mock import patch, call
from walking_strategy.muscle import Muscle


class TestMuscle(unittest.TestCase):

    def test_genotype_validation(self):
        self.assertRaises(ValueError, Muscle, 200, '4D')

    # @patch('numpy.random.choice')
    # @patch.object(Muscle, 'generate_random_activations_dna')
    # @patch.object(Muscle, 'calculate_fourier_activations')
    def test_random_init(self):
    # def test_random_init(self, mock_calculate_fourier_activations, mock_generate_random_activations_dna, mock_choice):
        period = 120
        discretization_frequency = 40
        activations_format = 'fourier'
        activations_dna = [1, 2, 3]
        muscle_activations = [4, 5, 6]

        # mock_choice.side_effect = [discretization_frequency, activations_format]
        # mock_generate_random_activations_dna.return_value = activations_dna
        # mock_calculate_fourier_activations.return_value = muscle_activations

        muscle = Muscle(period, '2D')

        # self.assertEqual(muscle.discretization_frequency, discretization_frequency)
        # self.assertEqual(muscle.activations_format, activations_format)
        # self.assertEqual(muscle.activations_dna, activations_dna)
        # self.assertEqual(muscle.muscle_activations, muscle_activations)
        # self.assertEqual(
        #     mock_choice.call_args_list,
        #     [call([5, 10, 20, 40]), call(['direct', 'fourier'])]
        # )
        # mock_generate_random_activations_dna.assert_called_once_with(period, activations_format, discretization_frequency)
        # mock_calculate_fourier_activations.assert_called_once_with(activations_dna, period)


# if __name__ == '__main__':
#     unittest.main()
