from walking_strategy.muscle import Muscle
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

muscle = Muscle(period=200)
print(muscle)

x = [i for i in range(muscle.period)]
y1 = np.array([muscle.get_activation(i) for i in range(muscle.period)])

y2 = y1.reshape((len(y1) // muscle.sampling_interval, muscle.sampling_interval))[:, 0]
current_indexes = np.arange(len(y2))
new_indexes = np.linspace(0, len(y2) - 1, muscle.period)
interpolator = interp1d(current_indexes, y2, kind='quadratic', fill_value='extrapolate')
y2 = interpolator(new_indexes)

plt.plot(x, y1, label="Action repeat")
plt.plot(x, y2, label="Quadratic interpolation")
plt.xlabel("Time step")
plt.ylabel("Activation")
plt.legend()
plt.show()