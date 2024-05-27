import numpy as np

class WalkingStrategy:
    def __init__(self, period, muscle_activations_fourier_coefficients=None):

        self.period = period
        if muscle_activations_fourier_coefficients is None:
            self.muscle_activations_fourier_coefficients = [WalkingStrategy.generate_single_muscle_activation() for _ in range(22)]
        else:
            self.muscle_activations_fourier_coefficients = muscle_activations_fourier_coefficients

    @staticmethod
    def generate_single_muscle_activation():
        result = []

        a0 = np.random.uniform(0, 1)
        result.append(a0)

        current_sum = 0

        coefficients = []
        for _ in range(6):
            coefficient = np.random.uniform(0, 0.5 - current_sum)

            coefficients.append(coefficient)

            current_sum += np.abs(coefficient)

        return [
            a0,
            coefficients[0],
            coefficients[1],
            np.random.uniform(-np.pi / 2, np.pi / 2),
            coefficients[2],
            coefficients[3],
            np.random.uniform(-np.pi / 2, np.pi / 2),
            coefficients[4],
            coefficients[5],
            np.random.uniform(-np.pi / 2, np.pi / 2),
        ]

    def calculate_muscle_activations(self, time):
        return [self.calculate_fourier_series_sum(coefficients, time) for coefficients in self.muscle_activations_fourier_coefficients]

    def calculate_fourier_series_sum(self, coefficients, time):
        a0 = coefficients[0]
        result = a0

        for i in range(1, len(coefficients), 3):
            an = coefficients[i]
            bn = coefficients[i + 1]
            phase = coefficients[i + 2]
            result += an * np.cos((i // 3 + 1) * 2 * np.pi * time / self.period + phase) + bn * np.sin(
                (i // 3 + 1) * 2 * np.pi * time / self.period + phase)

        return result

    def normalize(self):
        for muscle in self.muscle_activations_fourier_coefficients:
            if muscle[0] < 0:
                muscle[0] = 0
            elif muscle[0] > 1:
                muscle[0] = 1

            sum = np.abs(muscle[1]) + np.abs(muscle[2]) + np.abs(muscle[4]) + np.abs(muscle[5]) + np.abs(muscle[7]) + np.abs(muscle[8])

            if sum > 1/2:
                current_sum = 0
                for j in range(1, 10, 3):
                    limit = 0.5 - current_sum
                    if muscle[j] > limit:
                        muscle[j] -= (muscle[j] - limit)
                        break
                    current_sum += np.abs(muscle[j])

                    limit = 0.5 - current_sum
                    if muscle[j+1] > limit:
                        muscle[j+1] -= (muscle[j+1] - limit)
                        break
                    current_sum += np.abs(muscle[j+1])
