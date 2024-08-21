import unittest
from sim.parallel_sim import ParallelSim
from walking_strategy.walking_strategy import WalkingStrategy


class TestSim(unittest.TestCase):

    def test_2d_sim(self):
        parallel_sim = ParallelSim(mode='2D', parallelization=1, integrator_accuracy=0.001)

        simulation_results =  parallel_sim.run(walking_strategies = [WalkingStrategy(mode='2D') for _ in range(3)], number_of_steps=5)

        self.assertEqual(len(simulation_results), 3)
        for simulation_result in simulation_results:
            self.assertIsNotNone(simulation_result['walking_strategy'])
            self.assertIsInstance(simulation_result['walking_strategy'], WalkingStrategy)

            self.assertIsNotNone(simulation_result['fitness'])
            self.assertIsInstance(simulation_result['fitness'], float)

            self.assertIsNotNone(simulation_result['distance'])
            self.assertIsInstance(simulation_result['fitness'], float)

            self.assertIsNotNone(simulation_result['energy'])
            self.assertIsInstance(simulation_result['fitness'], float)

            self.assertEqual(simulation_result['steps'], 5)

        parallel_sim.shutdown()

    def test_3d_sim(self):
        parallel_sim = ParallelSim(mode='3D', parallelization=2, integrator_accuracy=0.001)

        simulation_results =  parallel_sim.run(walking_strategies = [WalkingStrategy(mode='3D') for _ in range(1)], number_of_steps=10)

        self.assertEqual(len(simulation_results), 1)
        for simulation_result in simulation_results:
            self.assertIsNotNone(simulation_result['walking_strategy'])
            self.assertIsInstance(simulation_result['walking_strategy'], WalkingStrategy)

            self.assertIsNotNone(simulation_result['fitness'])
            self.assertIsInstance(simulation_result['fitness'], float)

            self.assertIsNotNone(simulation_result['distance'])
            self.assertIsInstance(simulation_result['fitness'], float)

            self.assertIsNotNone(simulation_result['energy'])
            self.assertIsInstance(simulation_result['fitness'], float)

            self.assertEqual(simulation_result['steps'], 10)

        parallel_sim.shutdown()