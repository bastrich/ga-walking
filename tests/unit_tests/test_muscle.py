import unittest
import numpy as np
from walking_strategy.muscle import Muscle


class TestMuscle(unittest.TestCase):

    def test_validation(self):
        self.assertRaises(ValueError, Muscle, period=200, mode='4D')
        self.assertRaises(ValueError, Muscle, period=123, mode='2D')
        self.assertRaises(ValueError, Muscle, period=200, mode='2D', type='asd')
        self.assertRaises(ValueError, Muscle, period=200, mode='2D', sampling_interval=456)
        self.assertRaises(ValueError, Muscle, period=200, mode='2D', precision=1)
        self.assertRaises(ValueError, Muscle, period=200, mode='2D', components=[123])
        self.assertRaises(ValueError, Muscle, period=200, mode='2D', sampling_interval=40, precision=10)

    def test_components_and_activations(self):
        muscle = Muscle(period=400, mode='2D', sampling_interval=20, precision=10)
        self.assertEqual(10, len(muscle.components))
        self.assertEqual(400, len(muscle.activations))

        muscle = Muscle(period=200, mode='2D', sampling_interval=10, precision=5)
        self.assertEqual(5, len(muscle.components))
        self.assertEqual(200, len(muscle.activations))

    def test_change_period(self):
        muscle = Muscle(period=400, mode='2D')
        muscle = muscle.with_period(240)
        self.assertEqual(240, len(muscle.activations))

        muscle = Muscle(period=160, mode='2D')
        muscle = muscle.with_period(320)
        self.assertEqual(320, len(muscle.activations))

    def test_mutate_components(self):
        muscle = Muscle(period=200, mode='2D', type='direct')
        new_muscle = muscle.mutate_components(1, 1)
        for i in range(muscle.precision):
            self.assertTrue(muscle.components[i] != new_muscle.components[i] or muscle.components[i] == 0 or muscle.components[i] == 1)
            self.assertIsInstance(new_muscle.components[i], float)
            self.assertTrue(abs(new_muscle.components[i] - muscle.components[i]) < 0.5)

        muscle = Muscle(period=200, mode='2D', type='fourier')
        new_muscle = muscle.mutate_components(1, 1)
        for i in range(muscle.precision):
            self.assertNotEqual(muscle.components[i], new_muscle.components[i])
            self.assertIsInstance(new_muscle.components[i], complex)
            self.assertTrue(abs(abs(new_muscle.components[i]) - abs(muscle.components[i])) < muscle.period / muscle.sampling_interval)
            print(abs(np.angle(new_muscle.components[i]) - np.angle(muscle.components[i])))
            self.assertTrue(abs(np.angle(new_muscle.components[i]) - np.angle(muscle.components[i])) < 6)
