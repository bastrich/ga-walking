from walking_strategy.muscle import Muscle
import matplotlib.pyplot as plt

muscle = Muscle(period=200, type='fourier')
print(muscle)

x1 = [i for i in range(muscle.period)]
y1 = [muscle.get_activation(i) for i in range(muscle.period)]

new_muscle = muscle.mutate_components(1, 1)

x2 = [i for i in range(new_muscle.period)]
y2 = [new_muscle.get_activation(i) for i in range(new_muscle.period)]


plt.plot(x1, y1, label=f'muscle activation original')
plt.plot(x2, y2, label=f'muscle activation mutated')
plt.xlabel("Time step")
plt.ylabel("Activation")
plt.legend()
plt.show()
