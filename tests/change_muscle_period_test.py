from walking_strategy.muscle import Muscle
import matplotlib.pyplot as plt

muscle = Muscle(period=200)
print(muscle)

x1 = [i for i in range(muscle.period)]
y1 = [muscle.get_activation(i) for i in range(muscle.period)]

# new_muscle = muscle.with_period(120)
new_muscle = muscle.with_period(320)

x2 = [i for i in range(new_muscle.period)]
y2 = [new_muscle.get_activation(i) for i in range(new_muscle.period)]


plt.plot(x1, y1, label=f'muscle activation {muscle.period}')
plt.plot(x2, y2, label=f'muscle activation {new_muscle.period}')
plt.xlabel("Time step")
plt.ylabel("Activation")
plt.legend()
plt.show()
