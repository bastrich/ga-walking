import numpy as np
from perlin_noise import PerlinNoise
from scipy.interpolate import interp1d
import copy


class Muscle:
    PRECISION = 8
    LEVELS = np.linspace(0, 0.8, 5)

    def __init__(self, period, levels=None):
        self.period = period

        if levels is not None:
            self.levels = levels
        else:
            self.levels = self.generate_random_levels()

        self.activations = self.calculate_activations(self.levels, self.period)

    def generate_random_levels(self):
        return [np.random.choice(self.LEVELS) for _ in range(self.PRECISION)]

    def calculate_activations(self, levels, period):
        return np.repeat(levels, period // self.PRECISION)

    def mutate(self, mutation_rate, mutation_amount):
        new_levels = copy.deepcopy(self.levels)

        for i in range(len(new_levels)):
            if np.random.uniform() < mutation_rate:
                new_levels[i] = self.LEVELS[int(np.round(np.random.normal(np.argmax(self.LEVELS == new_levels[i]), len(self.LEVELS)))) % len(self.LEVELS)]

        return Muscle(period=self.period, levels=new_levels)

    def get_muscle_activation(self, time):
        return self.activations[time % self.period]

