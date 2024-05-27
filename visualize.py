from osim.env import L2M2019Env
import pickle
from walking_strategy import WalkingStrategy

# visualize the best
with open('best', 'rb') as file:
    best_walking_strategy = pickle.load(file)

env = L2M2019Env(visualize=True, difficulty=0)
env.reset()
for sim_step in range(10000):
    observation, reward, done, info = env.step(best_walking_strategy.calculate_muscle_activations(sim_step))
    # observation, reward, done, info = env.step(env.action_space.sample())
    # if done:
    #     break