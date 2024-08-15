import pickle
import matplotlib.pyplot as plt
import numpy as np

def evaluate(file_path, learning_rate_index):
    with open(file_path, 'rb') as file:
        analytics = pickle.load(file)

    simulation_durations = [a['simulation_duration'] for a in analytics]
    print(f'''Simulation duration:
    Avg: {np.round(np.average(simulation_durations), 2)}
    Min: {np.round(min(simulation_durations), 2)}
    Max: {np.round(max(simulation_durations), 2)}
    Mdn: {np.round(np.median(simulation_durations), 2)}''')

    evaluation_durations = [a['evaluation_duration'] for a in analytics]
    print(f'''Evaluation duration:
    Avg: {np.round(np.average(evaluation_durations), 2)}
    Min: {np.round(min(evaluation_durations), 2)}
    Max: {np.round(max(evaluation_durations), 2)}
    Mdn: {np.round(np.median(evaluation_durations), 2)}''')

    print(f'Walking stability: {analytics[-1]["walking_stability"]}')

    print(f'Distance: {np.round(analytics[-1]["distance"], 2)}')

    print(f'Energy: {np.round(analytics[-1]["energy"] / analytics[-1]["distance"], 2)}')

    fits = np.diff([a['fitness'] for a in analytics], prepend=0)[:learning_rate_index]
    print(f'''Learning rate:
    Increase: 
        Avg: {np.round(np.average(fits), 2)}
        Min: {np.round(min(fits), 2)}
        Max: {np.round(max(fits), 2)}
        Mdn: {np.round(np.median(fits), 2)}
    Best fitness: {np.round(analytics[-1]["fitness"], 2)}
    ''')

print('2D')
evaluate('../results/analytics_population_2d', 225)

print('3D')
evaluate('../results/analytics_population_3d', 1100)