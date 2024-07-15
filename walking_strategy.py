import numpy as np
from simple_muscle import Muscle

class WalkingStrategy:

    PERIODS = [120, 160, 200, 240, 320, 400]
    PRECISION = 16

    def __init__(self, period, muscles=None):
        if period is not None:
            self.period = period
        else:
            self.period = np.random.choice(self.PERIODS)

        if muscles is not None:
            self.muscles = muscles
        else:
            self.muscles = [self.generate_muscle(self.period, i) for i in range(11)]

        self.evaluated_fitness = 0
        self.precision = WalkingStrategy.PRECISION

    @staticmethod
    def generate_muscle(period, i):
        # if i in [2, 5, 10]:
        #     activations = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0]) / 100
        #     return Muscle(period=period, fourier_coefficients=np.fft.fft(activations)[:5])
        #
        # if i == 9:
        #     activations = np.array( [10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0]) / 100
        #     return Muscle(period=period, fourier_coefficients=np.fft.fft(activations)[:5])

        return Muscle(period=period, precision=WalkingStrategy.PRECISION)


    def mutate(self, mutation_rate, mutation_amount):

        if np.random.uniform() < 0.3:
            return self

        new_muscles = [muscle.mutate(mutation_rate, mutation_amount) if np.random.uniform() < mutation_rate else muscle for muscle in self.muscles]

        new_period = self.period
        if np.random.uniform() < 0.2 * mutation_rate:
            new_period = np.random.choice(self.PERIODS)
            new_muscles = [muscle.with_period(new_period) for muscle in new_muscles]

        return WalkingStrategy(
            period=new_period,
            muscles=new_muscles
        )

    def crossover(self, other):
        new_period = np.random.choice([self.period, other.period])

        switch_index = np.random.randint(0, len(self.muscles) * self.precision + 1)
        muscles = []
        for i in range(len(self.muscles)):
            if i < switch_index // self.precision:
                muscles.append(self.muscles[i])
            elif i > switch_index // self.precision:
                muscles.append(other.muscles[i])
            else:
                muscles.append(
                    Muscle(
                        period=new_period,
                        precision=self.precision,
                        fourier_coefficients=np.concatenate((self.muscles[i].fourier_coefficients[:(switch_index - i * self.precision)], other.muscles[i].fourier_coefficients[(switch_index - i * self.precision):]))
                    )
                )

        return WalkingStrategy(
            period=new_period,
            muscles=muscles
        )

    def with_precision(self, precision):
        # TODO ADD self.precision
        return WalkingStrategy(
            period=self.period,
            muscles=[muscle.with_precision(precision) for muscle in self.muscles]
        )

    def with_period(self, period):
        return WalkingStrategy(
            period=period,
            muscles=[muscle.with_period(period) for muscle in self.muscles]
        )

    def get_muscle_activations(self, time):
        return ([muscle.get_muscle_activation(time) for muscle in self.muscles] +
                [muscle.get_muscle_activation(time + self.period // 2) for muscle in self.muscles])
