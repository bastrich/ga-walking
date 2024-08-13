import pickle
import matplotlib.pyplot as plt


def build_plot(file_path):
    with open(file_path, 'rb') as file:
        population = pickle.load(file)

    best_walking_strategy = population.walking_strategies[0]

    _, axs = plt.subplots(11, 1, figsize=(5, 20), sharex=True)

    muscle_names = [
        'HAB: hip abductor',
        'HAD: hip adductor',
        'HFL: hip flexor',
        'GLU: glutei',
        'HAM: hamstrings',
        'RF: rectus femoris',
        'VAS: vastii',
        'BFSH: biceps femoris, short head',
        'GAS: gastrocnemius',
        'SOL: soleus',
        'TA: tibialis anterior'
    ]
    for i, muscle in enumerate(best_walking_strategy.muscles):
        x = [i for i in range(muscle.period)]
        y = [muscle.get_activation(i) for i in range(muscle.period)]

        axs.flat[i].plot(x, y)
        axs.flat[i].set(
            title=muscle_names[i],
            ylabel='Activation',
            xticks=[1] + list(range(20, muscle.period + 1, 20))
        )
    axs[-1].set_xlabel('Simulation step')

    plt.tight_layout()
    plt.show()


build_plot('../results/population_3d_initial')
build_plot('../results/population_3d')
build_plot('../results/population_2d_initial')
build_plot('../results/population_2d')