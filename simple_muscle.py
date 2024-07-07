import numpy as np
from perlin_noise import PerlinNoise
from scipy.interpolate import interp1d
import copy


class Muscle:
    PRECISION = 20
    DISCRETIZATION = 5

    def __init__(self, period, fourier_coefficients=None):
        self.period = period

        if fourier_coefficients is not None:
            self.fourier_coefficients = fourier_coefficients
        else:
            self.fourier_coefficients = self.generate_random_fourier_coefficients(self.period)

        self.activations = self.calculate_activations(self.fourier_coefficients, self.period)

    def generate_random_fourier_coefficients(self, period):
        noise = PerlinNoise(octaves=np.random.uniform(low=0.4, high=3))
        activations = np.array([noise(i / (period // self.DISCRETIZATION)) for i in range(period // self.DISCRETIZATION)])

        min_value = np.min(activations)
        max_value = np.max(activations)
        if min_value != max_value:
            activations = (activations - min_value) / (max_value - min_value)

        activations = np.clip(activations - 0.5 * np.average(activations), 0, 1)

        return np.fft.fft(activations)[:self.PRECISION]

    def calculate_activations(self, fourier_coefficients, period):
        signal = np.real(np.fft.ifft(np.pad(fourier_coefficients, (0, period // self.DISCRETIZATION - len(fourier_coefficients)), 'constant')))
        # current_indexes = np.arange(len(signal))
        # new_indexes = np.linspace(0, len(signal) - 1, period)
        # interpolator = interp1d(current_indexes, signal, kind='linear', fill_value='extrapolate')
        # result = interpolator(new_indexes)
        return signal.repeat(self.DISCRETIZATION)

    def mutate(self, mutation_rate, mutation_amount):
        new_fourier_coefficients = copy.deepcopy(self.fourier_coefficients)

        for i in range(len(new_fourier_coefficients)):
            real_mutation = 0
            if np.random.uniform() < mutation_rate:
                real_mutation = np.random.normal(0, max(0.1, mutation_amount * np.abs(np.real(new_fourier_coefficients[i]))))

            imag_mutation = 0
            if i != 0 and np.random.uniform() < mutation_rate:
                imag_mutation = np.random.normal(0, max(0.1, mutation_amount * np.abs(np.imag(new_fourier_coefficients[i]))))

            new_fourier_coefficients[i] += real_mutation + 1j * imag_mutation

            # real_mutation = mutation_amount * np.random.normal(scale=40)
            #
            # imag_mutation = 0
            # if i != 0:
            #     imag_mutation = mutation_amount * np.random.normal(scale=40)
            #
            # new_fourier_coefficients[i] += real_mutation + 1j * imag_mutation

        new_fourier_coefficients[0] = np.clip(np.real(new_fourier_coefficients[0]), 0, 0.75 * self.period // self.DISCRETIZATION) + 0j

        signal = np.real(np.fft.ifft(np.pad(new_fourier_coefficients, (0, self.period // self.DISCRETIZATION - len(new_fourier_coefficients)), 'constant')))

        if np.min(signal) < 0 - 0.1 or np.max(signal) > 1 + 0.1:
            signal = np.mod(signal, 1.00000000001)
            new_fourier_coefficients = np.fft.fft(signal)[:self.PRECISION]

        return Muscle(period=self.period, fourier_coefficients=new_fourier_coefficients)

    def get_muscle_activation(self, time):
        return self.activations[time % self.period]

