import numpy as np

class WalkingStrategy:
    def __init__(self, period, muscle_activations=None):
        self.period = period

        if muscle_activations is None:
            self.muscle_activations = [WalkingStrategy.generate_single_muscle_activations() for _ in range(11)]
        else:
            self.muscle_activations = muscle_activations

        self.muscle_activations_cache = np.vstack([self.normalize(self.calculate_fourier_series_sums_1(coefficients)) for coefficients in self.muscle_activations] +
         [self.normalize(self.calculate_fourier_series_sums_2(coefficients)) for coefficients in self.muscle_activations]).transpose()

    def normalize(self, muscle_activations):
        min_value = np.min(muscle_activations)
        max_value = np.max(muscle_activations)

        if min_value == max_value:
            return muscle_activations

        return (muscle_activations - min_value) / (max_value - min_value)

    def calculate_fourier_series_sums_1(self, single_muscle_coefficients):
        return np.array([self.calculate_fourier_series_sum(single_muscle_coefficients, i) for i in range(self.period)])

    def calculate_fourier_series_sums_2(self, single_muscle_coefficients):
        return np.array([self.calculate_fourier_series_sum(single_muscle_coefficients, i + self.period // 2) for i in range(self.period)])

    @staticmethod
    def generate_single_muscle_activations():
        return [np.random.uniform(0, 1) for _ in range(40)]

    def get_muscle_activations(self, time):
        return self.muscle_activations_cache[time % self.period]

    def calculate_fourier_series_sum(self, coefficients, time):
        return coefficients[(time % self.period) // 5]
