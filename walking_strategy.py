import numpy as np
from simple_muscle import Muscle

class WalkingStrategy:
    def __init__(self, period, muscles=None):
        self.period = period
        if muscles is not None:
            self.muscles = muscles
        else:
            self.muscles = [self.generate_muscle(period, i) for i in range(11)]
        self.evaluated_fitness = 0

    @staticmethod
    def generate_muscle(period, i):
        if i in [2, 5, 10]:
            activations = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0]) / 100
            return Muscle(period, np.fft.fft(activations)[:Muscle.PRECISION])

        if i == 9:
            activations = np.array( [10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0]) / 100
            return Muscle(period, np.fft.fft(activations)[:Muscle.PRECISION])

        return Muscle(period)


    def mutate(self, mutation_rate, mutation_amount):
        return WalkingStrategy(
            period=self.period,
            muscles=[muscle.mutate(mutation_rate, mutation_amount) for muscle in self.muscles]
        )

    def crossover(self, other):
        switch_index = np.random.randint(0, 11)

        return WalkingStrategy(
            period=self.period,
            muscles=self.muscles[:switch_index] + other.muscles[switch_index:]
        )

    def get_muscle_activations(self, time):
        return ([muscle.get_muscle_activation(time) for muscle in self.muscles] +
                [muscle.get_muscle_activation(time + self.period // 2) for muscle in self.muscles])
