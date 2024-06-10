import pickle

import matplotlib.pyplot as plt

from walking_strategy import WalkingStrategy

# create data
x = [i for i in range(200)]

with open('population', 'rb') as file:
    walking_strategy = pickle.load(file).walking_strategies[0]
# walking_strategy = WalkingStrategy(200)

# with open('best', 'rb') as file:
#     walking_strategy = pickle.load(file)

y1 = [walking_strategy.get_muscle_activations(i)[2] for i in x]
y2 = [walking_strategy.get_muscle_activations(i+200)[2] for i in x]

# plot lines
plt.plot(x, y1, label="line 1")
plt.plot(x, y2, label="line 2")
plt.legend()
plt.show()