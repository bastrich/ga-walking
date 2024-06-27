import numpy as np
from muscle import Muscle

class WalkingStrategy:
    def __init__(self, period, muscles=None):
        self.period = period
        if muscles is not None:
            self.muscles = muscles
        else:
            self.muscles = [Muscle(period) for _ in range(11)]

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
