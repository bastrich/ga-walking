import pickle

import numpy as np

with open('../results/population', 'rb') as file:
    population = pickle.load(file)

for walking_strategy in population.walking_strategies:
    period = walking_strategy.period
    precision = walking_strategy.precision
    muscles = walking_strategy.muscles
    print(period)
    print(precision)
    print(np.unique([muscle.period for muscle in muscles]))
    print(np.unique([len(muscle.fourier_coefficients) for muscle in muscles]))
