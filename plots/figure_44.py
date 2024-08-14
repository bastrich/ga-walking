import pickle
import matplotlib.pyplot as plt
import numpy as np

x = [i+1 for i in range(50)]

with open('../results/analytics_integrator_accuracy_0_00005', 'rb') as file:
    y1 = [analytics['simulation_duration'] for analytics in pickle.load(file)]
with open('../results/analytics_integrator_accuracy_0_001', 'rb') as file:
    y2 = [analytics['simulation_duration'] for analytics in pickle.load(file)]
with open('../results/analytics_integrator_accuracy_0_005', 'rb') as file:
    y3 = [analytics['simulation_duration'] for analytics in pickle.load(file)]


plt.plot(x, y1, label=f'integrator_accuracy=0.00005, avg = {np.average(y1)}')
plt.plot(x, y2, label=f'integrator_accuracy=0.001, avg = {np.average(y2)}')
plt.plot(x, y3, label=f'integrator_accuracy=0.005, avg = {np.average(y3)}')
plt.xlabel("Generation")
plt.ylabel("Simulation duration")
plt.xticks([1] + list(range(5, 51, 5)))
plt.legend()
plt.show()