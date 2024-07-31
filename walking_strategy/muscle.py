import random

import numpy as np
from perlin_noise import PerlinNoise
from scipy.interpolate import interp1d
import copy

class Muscle:

    TYPES = ['direct', 'fourier']
    PERIODS = [120, 160, 200, 240, 320, 400]
    SAMPLING_INTERVALS = [5, 10, 20, 40]
    PRECISIONS = [5, 10, 20, 40]

    def __init__(self, period, type=None, sampling_interval=None, precision=None, components=None):
        if period in self.PERIODS:
            self.period = period
        else:
            raise ValueError(f'period must be one of {self.PERIODS}')

        if type is None:
            self.type = random.choice(self.TYPES)
        elif type in self.TYPES:
            self.type = type
        else:
            raise ValueError(f'type must be one of {self.TYPES}')

        if sampling_interval is None:
            self.sampling_interval = random.choice(list(filter(lambda si: self.period // si >= min(self.PRECISIONS), self.SAMPLING_INTERVALS)))
        elif sampling_interval in self.SAMPLING_INTERVALS:
            if self.period // sampling_interval < min(self.PRECISIONS):
                raise ValueError(f'period // sampling_interval must be equal or greater than {min(self.PRECISIONS)}')
            self.sampling_interval = sampling_interval
        else:
            raise ValueError(f'sampling_interval must be one of {self.SAMPLING_INTERVALS}')

        if precision is None:
            self.precision = random.choice(list(filter(lambda p: self.period // self.sampling_interval >= p, self.PRECISIONS)))
        elif precision in self.PRECISIONS:
            if self.period // self.sampling_interval < precision:
                raise ValueError('period // sampling_interval must be equal or greater than precision')
            self.precision = precision
        else:
            raise ValueError(f'precision must be one of {self.PRECISIONS}')

        if components is None:
            self.components = self.generate_random_components(self.period, self.type, self.sampling_interval, self.precision)
        elif self.precision != len(components):
            raise ValueError('length of components must be equal to precision')
        elif self.type == 'direct' and isinstance(components[0], complex) or self.type == 'fourier' and isinstance(components[0], float):
            raise ValueError('type of components elements must correspond to genotype type')
        else:
            self.components = components

        self.activations = self.calculate_activations(self.type, self.components, self.period, self.sampling_interval)

    @staticmethod
    def generate_random_components(period, type, sampling_interval, precision):
        noise = PerlinNoise(octaves=np.random.uniform(low=0.4, high=3))
        activations = np.array([noise(i / (period // sampling_interval)) for i in range(period // sampling_interval)])

        min_value = np.min(activations)
        max_value = np.max(activations)
        if min_value != max_value:
            activations = (activations - min_value) / (max_value - min_value)
        else:
            raise ValueError('Issues in random genotype generation')

        activations = np.random.uniform(1, 2) * np.clip(activations - 0.8 * np.average(activations), 0, 1)

        if type == 'fourier':
            return np.fft.fft(activations)[:precision]

        current_indexes = np.arange(len(activations))
        new_indexes = np.linspace(0, len(activations) - 1, precision)
        interpolator = interp1d(current_indexes, activations, kind='quadratic', fill_value='extrapolate')
        return interpolator(new_indexes)

    @staticmethod
    def calculate_activations(type, components, period, sampling_interval):
        if type == 'fourier':
            activations = np.real(np.fft.ifft(np.pad(components, (0, period // sampling_interval - len(components)), 'constant')))
        elif type == 'direct':
            current_indexes = np.arange(len(components))
            new_indexes = np.linspace(0, len(components) - 1, period // sampling_interval)
            interpolator = interp1d(current_indexes, components, kind='quadratic', fill_value='extrapolate')
            activations = interpolator(new_indexes)
        else:
            raise ValueError('Logic error: unexpected behavior')

        return activations.repeat(sampling_interval)


    def get_activation(self, time):
        return self.activations[time % self.period]

    def __str__(self):
        current_print_options = np.get_printoptions()
        np.set_printoptions(linewidth=np.inf)

        text = f'''
Muscle:
    Type = {self.type};
    Period = {self.period};
    Sampling Interval = {self.sampling_interval};
    Precision = {self.precision};
    Components = {self.components};
        '''

        np.set_printoptions(**current_print_options)
        return text

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



