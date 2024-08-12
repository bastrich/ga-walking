from walking_strategy.muscle import Muscle
import matplotlib.pyplot as plt

muscle = Muscle(period=200, mode='2D')
print(muscle)

x = [i for i in range(muscle.period)]
y = [muscle.get_activation(i) for i in range(muscle.period)]

plt.plot(x, y, label="muscle activation")
plt.xlabel("Time step")
plt.ylabel("Activation")
plt.legend()
plt.show()