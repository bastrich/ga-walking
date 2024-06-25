import numpy as np
from perlin_noise import PerlinNoise
from scipy.interpolate import interp1d
import copy

class WalkingStrategy:
    def __init__(self, period, symmetric=True, dna=None):
        self.period = period
        self.symmetric = symmetric
        if dna is not None:
            self.dna = dna
        else:
            self.dna = [self.get_random_gene(period) for _ in range(11 if self.symmetric else 22)]

        self.muscle_activations = [self.calculate_muscle_activation(gene, period) for gene in self.dna]
        if self.symmetric:
            self.other_leg_muscle_activations = np.roll(self.muscle_activations, self.period // 2)

        # if any(isinstance(a, complex) for a in self.muscle_activations) or any(isinstance(a, complex) for a in self.other_leg_muscle_activations):
        #     a = 5

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
        result = interpolator(new_indexes)
        return result

    def crossover(self, other):
        if self.symmetric != other.symmetric:
            raise ValueError("Cannot crossover strategies of different symmetric types")

        switch_index = np.random.randint(0, 11 if self.symmetric else 22)

        return WalkingStrategy(
            period=self.period,
            symmetric=self.symmetric,
            dna=self.dna[:switch_index] + other.dna[switch_index:]
        )

    def mutate(self):
        new_dna = copy.deepcopy(self.dna)
        for i in range(len(new_dna)):
            for j in range(len(new_dna[i])):
                if np.random.uniform() < 0.05:
                    new_dna[i][j] += 0.5 * np.random.normal()

            new_dna[i][0] = np.clip(np.abs(new_dna[i][0]), 0, 0.999999)

            if self.dna[i][0] < 0.5 <= new_dna[i][0]:
                new_gene = np.fft.fft(np.clip(np.real(np.fft.ifft(new_dna[i][1:])), 0, 1))
                new_gene = np.insert(new_gene, 0, new_dna[i][0] + 0j)
                new_dna[i] = new_gene
            elif self.dna[i][0] >= 0.5 > new_dna[i][0]:
                new_gene = np.fft.fft(np.clip(np.real(np.fft.ifft(new_dna[i][1:])), 0, 1))
                new_gene = np.insert(new_gene, 0, new_dna[i][0] + 0j)
                new_dna[i] = new_gene

            if new_dna[i][0] < 0.5:
                new_dna[i][1:] = np.clip(new_dna[i][1:], 0, 1)
            else:
                new_gene = np.fft.fft(np.clip(np.real(np.fft.ifft(new_dna[i][1:])), 0, 1))
                new_gene = np.insert(new_gene, 0, new_dna[i][0] + 0j)
                new_dna[i] = new_gene

        return WalkingStrategy(period=self.period, symmetric=self.symmetric, dna=new_dna)

    def get_muscle_activations(self, time):
        if self.symmetric:
            result = [muscle_activation[time % self.period] for muscle_activation in self.muscle_activations] + [muscle_activation[time % self.period] for muscle_activation in self.other_leg_muscle_activations]
            # if any(isinstance(a, complex) for a in result):
            #     a = 5
            return result
        else:
            return [muscle_activation[time % self.period] for muscle_activation in self.muscle_activations]
