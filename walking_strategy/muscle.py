import random

import numpy as np
from perlin_noise import PerlinNoise
from scipy.interpolate import interp1d
import copy

class Muscle:
    """
    Represents genotype and phenotype of a single muscle

    Attributes:
        TYPES (list): List of available values for genotype type.
        PERIODS (list): List of available periods.
        SAMPLING_INTERVALS_2D (list): List of available sampling intervals for 2D env.
        PRECISIONS_2D (list): List of available precisions for 2D env.
        SAMPLING_INTERVALS_3D (list): List of available sampling intervals for 3D env.
        PRECISIONS_3D (list): List of available precisions for 3D env.
        mode (str): 2D or 3D.
        type (str): Type of representation of activation values, 'direct' or 'fourier'.
        period (int): numbers of simulation steps in a single walking period.
        precision (int): how many values to use to represent activation function.
        sampling_interval (int): how often to change activation values, simulation steps.
        components (list): list of real or complex values representing activation function.
        activations (list): phenotype, values of calculated activation function.
        frame_skipping (str): use action_repeat or interpolation for frame skipping.
    """

    TYPES = ['direct', 'fourier']
    PERIODS = [120, 160, 200, 240, 320, 400]

    SAMPLING_INTERVALS_2D = [5, 10, 20, 40]
    PRECISIONS_2D = [5, 10, 20, 40]

    SAMPLING_INTERVALS_3D = [5, 10]
    PRECISIONS_3D = [8, 16, 32]

    def __init__(self, period, mode, type=None, sampling_interval=None, precision=None, components=None, initial_generation='perlin', frame_skipping='action_repeat'):
        """
        Initiates a muscle.

        Args:
            initial_generation (str): How to generate initial activation function: perlin or random.
            Other arguments are described in the class attributes.
        """
        if mode not in ['2D', '3D']:
            raise ValueError(f'mode must be one of 2D or 3D')
        self.mode = mode
        if self.mode == '2D':
            self.PRECISIONS = self.PRECISIONS_2D
            self.SAMPLING_INTERVALS = self.SAMPLING_INTERVALS_2D
        else:
            self.PRECISIONS = self.PRECISIONS_3D
            self.SAMPLING_INTERVALS = self.SAMPLING_INTERVALS_3D

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
            self.components = self.generate_random_components(self.period, self.type, self.sampling_interval, self.precision, initial_generation)
        elif self.precision != len(components):
            raise ValueError('length of components must be equal to precision')
        elif self.type == 'direct' and isinstance(components[0], complex) or self.type == 'fourier' and isinstance(components[0], float):
            raise ValueError('type of components elements must correspond to genotype type')
        else:
            self.components = components

        self.frame_skipping = frame_skipping

        self.activations = self.calculate_activations(self.type, self.components, self.period, self.sampling_interval, self.frame_skipping)

    @staticmethod
    def generate_random_components(period, type, sampling_interval, precision, generation):
        """
        Generates random activation function and calculates its components.
        """
        if generation == 'perlin':
            noise = PerlinNoise(octaves=np.random.uniform(low=0.4, high=3))
            activations = np.array([noise(i / (period // sampling_interval)) for i in range(period // sampling_interval)])

            min_value = np.min(activations)
            max_value = np.max(activations)
            if min_value != max_value:
                activations = (activations - min_value) / (max_value - min_value)
            else:
                raise ValueError('Issues in random genotype generation')

            activations = np.random.uniform(1, 2) * np.clip(activations - 0.8 * np.average(activations), 0, 1)
        elif generation == 'random':
            activations = np.random.uniform(size=period // sampling_interval)
        else:
            raise ValueError(f'generation must be perlin or random')

        if type == 'fourier':
            return np.fft.fft(activations)[:precision]

        current_indexes = np.arange(len(activations))
        new_indexes = np.linspace(0, len(activations) - 1, precision)
        interpolator = interp1d(current_indexes, activations, kind='quadratic', fill_value='extrapolate')
        return interpolator(new_indexes)

    @staticmethod
    def calculate_activations(type, components, period, sampling_interval, frame_skipping):
        """
        Calculates activation function (phenotype) from genotype.
        """
        if type == 'fourier':
            activations = np.real(np.fft.ifft(np.pad(components, (0, period // sampling_interval - len(components)), 'constant')))
        elif type == 'direct':
            current_indexes = np.arange(len(components))
            new_indexes = np.linspace(0, len(components) - 1, period // sampling_interval)
            interpolator = interp1d(current_indexes, components, kind='quadratic', fill_value='extrapolate')
            activations = interpolator(new_indexes)
        else:
            raise ValueError('Logic error: unexpected behavior')

        if frame_skipping == 'action_repeat':
            return activations.repeat(sampling_interval)
        elif frame_skipping == 'interpolation':
            current_indexes = np.arange(len(activations))
            new_indexes = np.linspace(0, len(activations) - 1, period)
            interpolator = interp1d(current_indexes, activations, kind='quadratic', fill_value='extrapolate')
            return interpolator(new_indexes)
        else:
            raise ValueError(f'frame_skipping must be action_repeat or interpolation')


    def get_activation(self, time):
        """
        Provides muscle activation value at specified time.
        """
        return self.activations[time % self.period]

    def with_period(self, period):
        """
        Creates a new instance of Muscle with modified period.
        """
        if period == self.period:
            return self

        if period not in self.PERIODS:
            raise ValueError(f'period must be one of {self.PERIODS}')

        new_sampling_interval = self.sampling_interval
        new_precision = self.precision

        if self.type == 'fourier':
            new_components = self.components

            if period // self.sampling_interval < self.precision:
                available_sampling_intervals = list(filter(lambda si: period // si >= self.precision, self.SAMPLING_INTERVALS))
                if len(available_sampling_intervals) != 0:
                    new_sampling_interval = next(reversed(available_sampling_intervals))
                else:
                    new_precision = next(reversed(list(filter(lambda p: period // self.sampling_interval >= p, self.PRECISIONS))))

                new_components = np.real(np.fft.ifft(np.pad(self.components, (0, self.period // self.sampling_interval - len(self.components)),'constant')))

                current_indexes = np.arange(len(new_components))
                new_indexes = np.linspace(0, len(new_components) - 1, period // new_sampling_interval)
                interpolator = interp1d(current_indexes, new_components, kind='quadratic', fill_value='extrapolate')
                new_components = interpolator(new_indexes)

                new_components = np.fft.fft(new_components)[:new_precision]
            else:
                new_components = new_components * (period // self.sampling_interval) / (self.period // self.sampling_interval)

        else:
            new_components = self.components

            if period // self.sampling_interval < self.precision:
                available_sampling_intervals = list(filter(lambda si: period // si >= self.precision, self.SAMPLING_INTERVALS))
                if len(available_sampling_intervals) != 0:
                    new_sampling_interval = next(reversed(available_sampling_intervals))
                else:
                    new_precision = next(reversed(list(filter(lambda p: period // new_sampling_interval >= p, self.PRECISIONS))))

                current_indexes = np.arange(len(self.components))
                new_indexes = np.linspace(0, len(self.components) - 1, self.period // new_sampling_interval)
                interpolator = interp1d(current_indexes, self.components, kind='quadratic', fill_value='extrapolate')
                new_components = interpolator(new_indexes)

                current_indexes = np.arange(len(new_components))
                new_indexes = np.linspace(0, len(new_components) - 1, new_precision)
                interpolator = interp1d(current_indexes, new_components, kind='quadratic', fill_value='extrapolate')
                new_components = interpolator(new_indexes)

        return Muscle(
            period=period,
            mode=self.mode,
            type=self.type,
            sampling_interval=new_sampling_interval,
            precision=new_precision,
            components=new_components,
            frame_skipping=self.frame_skipping
        )

    def mutate_type(self):
        """
        Creates a new instance of Muscle with mutated genotype type.
        """
        if self.type == 'fourier':
            new_type = 'direct'

            new_components = np.real(np.fft.ifft(np.pad(self.components, (0, self.period // self.sampling_interval - len(self.components)), 'constant')))

            current_indexes = np.arange(len(new_components))
            new_indexes = np.linspace(0, len(new_components) - 1, self.precision)
            interpolator = interp1d(current_indexes, new_components, kind='quadratic', fill_value='extrapolate')
            new_components = interpolator(new_indexes)
        else:
            new_type = 'fourier'

            current_indexes = np.arange(len(self.components))
            new_indexes = np.linspace(0, len(self.components) - 1, self.period // self.sampling_interval)
            interpolator = interp1d(current_indexes, self.components, kind='quadratic', fill_value='extrapolate')
            new_components = interpolator(new_indexes)

            new_components = np.fft.fft(new_components)[:self.precision]

        return Muscle(
            period=self.period,
            mode=self.mode,
            type=new_type,
            sampling_interval=self.sampling_interval,
            precision=self.precision,
            components=new_components,
            frame_skipping=self.frame_skipping
        )

    def mutate_sampling_interval(self):
        """
        Creates a new instance of Muscle with mutated sampling interval.
        """
        new_sampling_interval = random.choice(list(filter(lambda si: self.period // si >= self.precision, self.SAMPLING_INTERVALS)))
        if new_sampling_interval == self.sampling_interval:
            return self

        if self.type == 'fourier':
            new_components = np.real(np.fft.ifft(np.pad(self.components, (0, self.period // self.sampling_interval - len(self.components)), 'constant')))

            current_indexes = np.arange(len(new_components))
            new_indexes = np.linspace(0, len(new_components) - 1, self.period // new_sampling_interval)
            interpolator = interp1d(current_indexes, new_components, kind='quadratic', fill_value='extrapolate')
            new_components = interpolator(new_indexes)

            new_components = np.fft.fft(new_components)[:self.precision]
        else:
            current_indexes = np.arange(len(self.components))
            new_indexes = np.linspace(0, len(self.components) - 1, self.period // new_sampling_interval)
            interpolator = interp1d(current_indexes, self.components, kind='quadratic', fill_value='extrapolate')
            new_components = interpolator(new_indexes)

            current_indexes = np.arange(len(new_components))
            new_indexes = np.linspace(0, len(new_components) - 1, self.precision)
            interpolator = interp1d(current_indexes, new_components, kind='quadratic', fill_value='extrapolate')
            new_components = interpolator(new_indexes)

        return Muscle(
            period=self.period,
            mode=self.mode,
            type=self.type,
            sampling_interval=new_sampling_interval,
            precision=self.precision,
            components=new_components,
            frame_skipping=self.frame_skipping
        )

    def mutate_precision(self):
        """
        Creates a new instance of Muscle with mutated precision.
        """
        new_precision = random.choice(list(filter(lambda p: self.period // self.sampling_interval >= p, self.PRECISIONS)))
        if new_precision == self.precision:
            return self

        if self.type == 'fourier':
            new_components = np.real(np.fft.ifft(np.pad(self.components, (0, self.period // self.sampling_interval - len(self.components)), 'constant')))

            current_indexes = np.arange(len(new_components))
            new_indexes = np.linspace(0, len(new_components) - 1, self.period // self.sampling_interval)
            interpolator = interp1d(current_indexes, new_components, kind='quadratic', fill_value='extrapolate')
            new_components = interpolator(new_indexes)

            new_components = np.fft.fft(new_components)[:new_precision]
        else:
            current_indexes = np.arange(len(self.components))
            new_indexes = np.linspace(0, len(self.components) - 1, self.period // self.sampling_interval)
            interpolator = interp1d(current_indexes, self.components, kind='quadratic', fill_value='extrapolate')
            new_components = interpolator(new_indexes)

            current_indexes = np.arange(len(new_components))
            new_indexes = np.linspace(0, len(new_components) - 1, new_precision)
            interpolator = interp1d(current_indexes, new_components, kind='quadratic', fill_value='extrapolate')
            new_components = interpolator(new_indexes)

        return Muscle(
            period=self.period,
            mode=self.mode,
            type=self.type,
            sampling_interval=self.sampling_interval,
            precision=new_precision,
            components=new_components,
            frame_skipping=self.frame_skipping
        )

    def mutate_components(self, mutation_rate, mutation_amount):
        """
        Creates a new instance of Muscle with mutated components.
        """
        if self.type == 'fourier':
            new_components = self.mutate_fourier_components(mutation_rate, mutation_amount)
        else:
            new_components = self.mutate_direct_components(mutation_rate, mutation_amount)

        return Muscle(
            period=self.period,
            mode=self.mode,
            type=self.type,
            sampling_interval=self.sampling_interval,
            precision=self.precision,
            components=new_components,
            frame_skipping=self.frame_skipping
        )

    def mutate_fourier_components(self, mutation_rate, mutation_amount):
        """
        Creates new mutated version of fourier components.
        """
        new_components = copy.deepcopy(self.components)

        for i in range(len(new_components)):
            if np.random.uniform() < mutation_rate:
                if np.random.uniform() < 0.8:
                    # mutate amplitude
                    mutation = mutation_amount * np.clip(np.random.normal(0, self.period // self.sampling_interval), -self.period // self.sampling_interval * 0.5, self.period // self.sampling_interval * 0.5)
                    if np.abs(mutation) < self.period // self.sampling_interval * 0.05:
                        mutation = np.sign(mutation) * self.period // self.sampling_interval * 0.05

                    if new_components[i] == 0j:
                        phase = np.random.uniform(-np.pi, np.pi)
                        mutation *= np.cos(phase) + 1j * np.sin(phase)
                    else:
                        mutation *= new_components[i] / np.abs(new_components[i])

                    new_components[i] += mutation

                    # try to keep activation inside [1, 0]
                    signal = np.real(np.fft.ifft(np.pad(new_components, (0, self.period // self.sampling_interval - len(new_components)), 'constant')))
                    min_value = np.min(signal)
                    max_value = np.max(signal)
                    if min_value < 0 and max_value > 1:
                        if np.abs(min_value) > max_value - 1:
                            new_components[i] *= 1 / (1 + np.abs(min_value))
                        else:
                            new_components[i] *= 1 / np.abs(max_value)
                    elif min_value < 0:
                        new_components[i] *= 1 / (1 + np.abs(min_value))
                    elif max_value > 1:
                        new_components[i] *= 1 / np.abs(max_value)
                elif np.random.choice([True, False]):
                    # imitation of shrink mutation
                    new_components[i] = 0j
                else:
                    # mutate phase
                    phase = np.random.uniform(-np.pi, np.pi)
                    new_components[i] = 0.5 * self.period // self.sampling_interval * (np.cos(phase) + 1j * np.sin(phase))

            if i != 0 and np.random.uniform() < mutation_rate:
                mutation = mutation_amount * np.clip(np.random.normal(0, np.pi), -np.pi / 2, np.pi / 2)
                if np.abs(mutation) < 2 * np.pi * 0.05:
                    mutation = np.sign(mutation) * 2 * np.pi * 0.05

                new_components[i] *= np.exp(1j * mutation)

        return new_components

    def mutate_direct_components(self, mutation_rate, mutation_amount):
        """
        Creates new mutated version of direct components.
        """
        new_components = copy.deepcopy(self.components)
        for i in range(len(new_components)):
            if np.random.uniform() <= mutation_rate:
                new_components[i] += mutation_amount * np.random.normal(0, 0.1)
                new_components[i] = np.clip(new_components[i], 0, 1)
        return new_components

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