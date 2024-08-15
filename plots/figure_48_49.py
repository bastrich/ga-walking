import pickle
import matplotlib.pyplot as plt


def build_duration_plot(file_path, xticks):
    with open(file_path, 'rb') as file:
        analytics = pickle.load(file)

    x = [i + 1 for i in range(len(analytics))]
    y = {
        'simulation_duration': [],
        'evaluation_duration': []
    }
    for generation in analytics:
        y['simulation_duration'].append(generation['simulation_duration'])
        y['evaluation_duration'].append(generation['evaluation_duration'])

    plt.stackplot(x, y.values(), labels=y.keys(), alpha=0.8)
    plt.xlabel('Generation')
    plt.ylabel('Duration, seconds')
    plt.xticks(xticks)
    plt.legend()
    plt.tight_layout()
    plt.show()

build_duration_plot('../results/analytics_population_2d', [1] + list(range(100, 301, 100)))
build_duration_plot('../results/analytics_population_3d', [1] + list(range(200, 2001, 200)))