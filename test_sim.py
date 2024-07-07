from my_osim import L2M2019Env
import numpy as np
import time

class Sim:

    def __init__(self, visualize=False):
        self.env = L2M2019Env(visualize=visualize, difficulty=0, integrator_accuracy=0.001)
        # self.refresh_counter = 10
        self.visualize = visualize
        # self.env.change_model(model='2D', difficulty=0)

    def run(self, walking_strategy, number_of_steps=1000):
        # if self.refresh_counter == 0:
        #     self.env = L2M2019Env(visualize=self.visualize, difficulty=0, integrator_accuracy=0.001)
        #     self.refresh_counter = 10

        self.env.reset()
        fitness = 0

        start_time = time.time()

        for sim_step in range(number_of_steps):

            _, reward, done, _ = self.env.step(walking_strategy.get_muscle_activations(sim_step))

            if done or time.time() - start_time > 10:
                break

            # time.sleep(0.1)
            fitness += reward

        # self.refresh_counter -= 1
        return fitness, sim_step