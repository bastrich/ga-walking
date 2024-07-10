import numpy as np
from perlin_noise import PerlinNoise
from scipy.interpolate import interp1d
import copy


class Muscle:
    DISCRETIZATION = 5

    def __init__(self, period, precision=20, fourier_coefficients=None):
        self.period = period
        self.precision = precision

        if fourier_coefficients is not None:
            self.fourier_coefficients = fourier_coefficients
        else:
            self.fourier_coefficients = self.generate_random_fourier_coefficients(self.period)

        self.activations = self.calculate_activations(self.fourier_coefficients, self.period)

    def change_precision(self, precision):
        if not hasattr(self, 'precision'):
            self.precision = 20

        if precision > self.precision:
            return

        self.precision = precision

        if self.precision > len(self.fourier_coefficients):
            self.fourier_coefficients = np.pad(self.fourier_coefficients, (0 + 0j, self.period // self.DISCRETIZATION - self.precision + len(self.fourier_coefficients)), 'constant')
        else:
            self.fourier_coefficients = self.fourier_coefficients[:self.precision]

        self.activations = self.calculate_activations(self.fourier_coefficients, self.period)

    def generate_random_fourier_coefficients(self, period):
        noise = PerlinNoise(octaves=np.random.uniform(low=0.4, high=3))
        activations = np.array([noise(i / (period // self.DISCRETIZATION)) for i in range(period // self.DISCRETIZATION)])

        min_value = np.min(activations)
        max_value = np.max(activations)
        if min_value != max_value:
            activations = (activations - min_value) / (max_value - min_value)

        activations = np.random.uniform(1, 2) * (np.clip(activations - 0.5 * np.average(activations), 0, 1))

        return np.fft.fft(activations)[:self.precision]

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
            if np.random.uniform() < mutation_rate:
                # print(f'b - {np.real(new_fourier_coefficients[i]) / np.imag(new_fourier_coefficients[i])}')
                mutation = mutation_amount * np.clip(np.random.normal(0, self.period // self.DISCRETIZATION * 0.5), -self.period // self.DISCRETIZATION * 0.5, self.period // self.DISCRETIZATION * 0.5)
                if np.abs(mutation) < self.period // self.DISCRETIZATION * 0.01:
                    mutation = np.sign(mutation) * self.period // self.DISCRETIZATION * 0.01
                mutation = new_fourier_coefficients[i] / np.abs(new_fourier_coefficients[i]) * mutation

                new_fourier_coefficients[i] += mutation
                # print(f'a - {np.real(new_fourier_coefficients[i]) / np.imag(new_fourier_coefficients[i])}')
                signal = np.real(np.fft.ifft(np.pad(new_fourier_coefficients, (0, self.period // self.DISCRETIZATION - len(new_fourier_coefficients)), 'constant')))
                min_value = np.min(signal)
                max_value = np.max(signal)
                if min_value < 0 and max_value > 1:
                    if np.abs(min_value) > max_value - 1:
                        new_fourier_coefficients[i] *= 1 / (1 + np.abs(min_value))
                    else:
                        new_fourier_coefficients[i] *= 1 / np.abs(max_value)
                elif min_value < 0:
                    new_fourier_coefficients[i] *= 1 / (1 + np.abs(min_value))
                elif max_value > 1:
                    new_fourier_coefficients[i] *= 1 / np.abs(max_value)




            if i != 0 and np.random.uniform() < mutation_rate:
                # print(f'b - {np.abs(new_fourier_coefficients[i])}')

                mutation = mutation_amount * np.clip(np.random.normal(0, np.pi / 2), -np.pi / 2, np.pi / 2)
                if np.abs(mutation) < 2 * np.pi * 0.01:
                    mutation = np.sign(mutation) * 2 * np.pi * 0.01

                new_fourier_coefficients[i] *= np.exp(1j * mutation)

                # print(f'a - {np.abs(new_fourier_coefficients[i])}')

        return Muscle(period=self.period, fourier_coefficients=new_fourier_coefficients)

    def get_muscle_activation(self, time):
        return self.activations[time % self.period]

