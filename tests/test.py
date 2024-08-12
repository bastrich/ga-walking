import pickle

from walking_strategy.muscle import Muscle
import matplotlib.pyplot as plt

with open('../results/population_3d', 'rb') as file:
    population = pickle.load(file)

muscle = population.walking_strategies[0].muscles[1]

x1 = [i for i in range(muscle.period)]
y1 = [muscle.get_activation(i) for i in range(muscle.period)]

for i in range(10):
    new_muscle = muscle.mutate_components(0.8, 0.8)

x2 = [i for i in range(new_muscle.period)]
y2 = [new_muscle.get_activation(i) for i in range(new_muscle.period)]


plt.plot(x1, y1, label=f'muscle activation original')
plt.plot(x2, y2, label=f'muscle activation mutated')
plt.xlabel("Time step")
plt.ylabel("Activation")
plt.legend()
plt.show()
