import numpy as np
from perlin_noise import PerlinNoise
from scipy.interpolate import interp1d

class WalkingStrategy:
    def __init__(self, period, symetric=True, dna=None):
        self.period = period
        if dna is not None:
            self.dna = dna
        else:
            self.dna = [self.get_random_gene(period) for _ in range(11 if symetric else 22)]

        self.muscle_activations = [self.calculate_muscle_activation(gene, period) for gene in self.dna]

        # if muscle_activations is None:
        #     self.muscle_activations = [WalkingStrategy.generate_single_muscle_activations() for _ in range(11)]
        # else:
        #     self.muscle_activations = muscle_activations
        #
        # self.muscle_activations_cache = np.vstack([self.normalize(self.calculate_fourier_series_sums_1(coefficients)) for coefficients in self.muscle_activations] +
        #  [self.normalize(self.calculate_fourier_series_sums_2(coefficients)) for coefficients in self.muscle_activations]).transpose()

    @staticmethod
    def get_random_gene(period):
        format_type = np.random.uniform(low=0.0, high=0.25)

        noise = PerlinNoise(octaves=np.random.uniform(low=0.4, high=5))
        number_of_values = period // 5
        muscle_activation = np.array([noise(i / number_of_values) for i in range(number_of_values)])

        min_value = np.min(muscle_activation)
        max_value = np.max(muscle_activation)
        if min_value != max_value:
            muscle_activation = (muscle_activation - min_value) / (max_value - min_value)

        return np.insert(muscle_activation, 0, format_type)

    @staticmethod
    def calculate_muscle_activation(gene, period):
        if gene[0] < 0.5:
            return WalkingStrategy.calculate_direct_muscle_activation(gene[1:], period)
        else:
            return WalkingStrategy.calculate_fourier_muscle_activation(gene[1:], period)
    @staticmethod
    def calculate_direct_muscle_activation(values, period):
        current_indexes = np.arange(len(values))
        new_indexes = np.linspace(0, len(values) - 1, period)
        interpolator = interp1d(current_indexes, values, kind='cubic', fill_value='extrapolate')
        return interpolator(new_indexes)

    @staticmethod
    def calculate_fourier_muscle_activation(values, period):
        signal = np.real(np.fft.ifft(values))
        current_indexes = np.arange(len(signal))
        new_indexes = np.linspace(0, len(signal) - 1, period)
        interpolator = interp1d(current_indexes, signal, kind='cubic', fill_value='extrapolate')
        return interpolator(new_indexes)

    def crossover(self, other):
        return self

    def mutate(self):
        return self

    # def normalize(self, muscle_activations):
    #     min_value = np.min(muscle_activations)
    #     max_value = np.max(muscle_activations)
    #
    #     if min_value == max_value:
    #         return muscle_activations
    #
    #     return (muscle_activations - min_value) / (max_value - min_value)
    #
    # def calculate_fourier_series_sums_1(self, single_muscle_coefficients):
    #     return np.array([self.calculate_fourier_series_sum(single_muscle_coefficients, i) for i in range(self.period)])
    #
    # def calculate_fourier_series_sums_2(self, single_muscle_coefficients):
    #     return np.array([self.calculate_fourier_series_sum(single_muscle_coefficients, i + self.period // 2) for i in range(self.period)])
    #
    # @staticmethod
    # def generate_single_muscle_activations():
    #     return [np.random.uniform(0, 1) for _ in range(40)]
    #
    # def get_muscle_activations(self, time):
    #     return self.muscle_activations_cache[time % self.period]
    #
    # def calculate_fourier_series_sum(self, coefficients, time):
    #     return coefficients[(time % self.period) // 5]
