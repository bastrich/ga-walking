from walking_strategy.muscle import Muscle
import matplotlib.pyplot as plt
import numpy as np

muscle = Muscle(period=200, mode='2D', generation='perlin')
print(muscle)

x = [i for i in range(muscle.period)]
y = np.array([muscle.get_activation(i) for i in range(muscle.period)])

plt.plot(x, y, label="Perlin noise")
plt.xlabel("Time step")
plt.ylabel("Activation")
plt.legend()
plt.show()