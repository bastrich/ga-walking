import random

import numpy as np
from perlin_noise import PerlinNoise
from scipy.interpolate import interp1d
import copy

class Muscle:

    ACTIVATIONS_FORMATS = ['direct', 'fourier']
    DISCRETIZATION_FREQUENCIES = [5, 10, 20, 40]

    def __init__(self, period, discretization_frequency=None, activations_format=None, activations_dna=None):
        self.period = period

        if discretization_frequency is not None and discretization_frequency not in self.DISCRETIZATION_FREQUENCIES:
            raise ValueError('discretization_frequency must be one of 5, 10, 20, 40')
        if activations_format is not None and activations_format not in self.ACTIVATIONS_FORMATS:
            raise ValueError('activations_format must be either "direct" or "fourier"')

        if discretization_frequency is not None:
            self.discretization_frequency = discretization_frequency
        else:
            self.discretization_frequency = np.random.choice(self.DISCRETIZATION_FREQUENCIES)

        if activations_format is not None:
            self.activations_format = activations_format
        else:
            self.activations_format = np.random.choice(['direct', 'fourier'])

        if activations_dna is not None:
            self.activations_dna = activations_dna
        else:
            self.activations_dna = self.generate_random_activations_dna(
                self.period,
                self.activations_format,
                self.discretization_frequency
            )

        if self.activations_format == 'direct':
            self.muscle_activations = self.calculate_direct_activations(self.activations_dna, self.period)
        else:
            self.muscle_activations = self.calculate_fourier_activations(self.activations_dna, self.period)

    @staticmethod
    def generate_random_activations_dna(period, activations_format, discretization_frequency):
        noise = PerlinNoise(octaves=np.random.uniform(low=0.4, high=5))
        number_of_values = period // discretization_frequency
        activations_dna = np.array([noise(i / number_of_values) for i in range(number_of_values)])

        min_value = np.min(activations_dna)
        max_value = np.max(activations_dna)
        if min_value != max_value:
            activations_dna = (activations_dna - min_value) / (max_value - min_value)

        if activations_format == 'direct':
            return activations_dna
        elif activations_format == 'fourier':
            return np.fft.fft(activations_dna)
        else:
            raise ValueError('activations_format must be either "direct" or "fourier"')

    @staticmethod
    def calculate_direct_activations(activations_dna, period):
        current_indexes = np.arange(len(activations_dna))
        new_indexes = np.linspace(0, len(activations_dna) - 1, period)
        interpolator = interp1d(current_indexes, activations_dna, kind='cubic', fill_value='extrapolate')
        return interpolator(new_indexes)

    @staticmethod
    def calculate_fourier_activations(activations_dna, period):
        signal = np.real(np.fft.ifft(activations_dna))
        current_indexes = np.arange(len(signal))
        new_indexes = np.linspace(0, len(signal) - 1, period)
        interpolator = interp1d(current_indexes, signal, kind='cubic', fill_value='extrapolate')
        result = interpolator(new_indexes)
        return result

    def mutate(self, mutation_rate, mutation_amount):
        new_discretization_frequency = self.discretization_frequency
        new_activations_format = self.activations_format
        new_activations_dna = copy.deepcopy(self.activations_dna)

        if np.random.uniform() < mutation_rate:
            new_discretization_frequency = np.random.choice(self.DISCRETIZATION_FREQUENCIES)

        if np.random.uniform() < mutation_rate:
            new_activations_format = np.random.choice(self.ACTIVATIONS_FORMATS)

        for i in range(len(new_activations_dna)):
            if np.random.uniform() < mutation_rate:
                if isinstance(new_activations_dna[i], complex):
                    real_mutation = np.random.normal(0, max(0.1, mutation_amount * np.abs(np.real(new_activations_dna[i]))))
                    imag_mutation = np.random.normal(0, max(0.1, mutation_amount * np.abs(np.imag(new_activations_dna[i]))))
                    new_activations_dna[i] += real_mutation + 1j * imag_mutation
                else:
                    new_activations_dna[i] += np.random.normal(0, max(0.1, mutation_amount * np.abs(new_activations_dna[i])))

        #????
        if np.random.uniform() < 0.1 * mutation_rate:
            new_activations_dna = np.roll(new_activations_dna, int(np.random.normal(0, 0.2 * self.period)))

        activations = new_activations_dna if self.activations_format == 'direct' else np.real(np.fft.ifft(new_activations_dna))
        activations = np.clip(activations, 0, 1)

        if self.discretization_frequency != new_discretization_frequency:
            current_indexes = np.arange(len(activations))
            new_indexes = np.linspace(0, len(activations) - 1, self.period // new_discretization_frequency)
            interpolator = interp1d(current_indexes, activations, kind='cubic', fill_value='extrapolate')
            activations = interpolator(new_indexes)

        return Muscle(
            period=self.period,
            discretization_frequency=new_discretization_frequency,
            activations_format=new_activations_format,
            activations_dna=activations if new_activations_format == 'direct' else np.fft.fft(activations)
        )

    def get_muscle_activation(self, time):
        return self.muscle_activations[time % self.period]

