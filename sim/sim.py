from sim.sim_env import SimEnv
import numpy as np
import time

class Sim:
    """
    Runs and manages the simulation of a walking strategy in the OpenSim environment.

    Attributes:
        MASS (float): The mass of a human model.
        G (float): The gravitational constant.
        mode (str): 2D or 3D.
        visualize (bool): Visualize or not.
        env (SimEnv): Simulation environment.
        fitness_helpers (dict): Intermediate fitness calculation data.
        footstep (dict): Footstep-related information during the simulation.
    """

    MASS = 75.16460000000001
    G = 9.80665

    def __init__(self, mode, visualize, integrator_accuracy=0.001):
        """
        Initializes the simulation environment and sets up helper structures for fitness calculation.
        """
        self.visualize = visualize
        self.mode = mode
        self.env = SimEnv(mode, visualize, integrator_accuracy)
        self.fitness_helpers = {}
        self.footstep = {}

    def run(self, walking_strategy, number_of_steps=1000):
        """
        Runs the simulation for the walking strategy for specified number of steps.

        Returns:
        tuple: A tuple containing the fitness score (float), the number of completed simulation steps (int), the total distance traveled (float), and the total energy spent (float).
        """
        prev_state = self.env.reset()

        self.init_fitness_helpers()
        self.init_footstep_info(prev_state)
        fitness = 0
        distance = 0
        energy = 0

        start_time = time.time()

        for sim_step in range(number_of_steps):

            current_state = self.env.step(walking_strategy.get_muscles_activations(sim_step))

            if self.is_failed(current_state) or (time.time() - start_time > 20 and not self.visualize):
                break

            self.update_footstep(current_state)

            if self.mode == '2D':
                fitness += self.calculate_2d_fitness(prev_state, current_state, walking_strategy.period)
            else:
                fitness += self.calculate_3d_fitness(prev_state, current_state, walking_strategy.period, sim_step)

            distance += self.calculate_current_distance_delta(prev_state, current_state)
            energy += self.calculate_current_energy(current_state)

            prev_state = current_state

        return np.round(fitness, 2), sim_step + 1, distance, energy

    def is_failed(self, state):
        """
        Checks if the model has fallen.
        """
        return state['body_pos']['pelvis'][1] < 0.6

    def init_fitness_helpers(self):
        """
        Initializes the fitness data.
        """
        self.fitness_helpers['footstep_effort'] = 0
        self.fitness_helpers['footstep_duration'] = 0

    def init_footstep_info(self, state):
        """
        Initializes the footstep tracking data.
        """
        self.footstep['n'] = 0
        self.footstep['new'] = False
        self.footstep['r_contact'] = True
        self.footstep['l_contact'] = False
        self.footstep['r_x'] = state['body_pos']['calcn_r'][0]
        self.footstep['l_x'] = state['body_pos']['calcn_l'][0]
        self.footstep['last_step'] = 'l'

    def update_footstep(self, current_state):
        """
        Counts footsteps based on the current state of the simulation.
        """
        r_contact = True if current_state['forces']['foot_r'][1] < -0.05*(self.MASS*self.G) else False
        l_contact = True if current_state['forces']['foot_l'][1] < -0.05*(self.MASS*self.G) else False
        r_x = current_state['body_pos']['calcn_r'][0]
        l_x = current_state['body_pos']['calcn_l'][0]

        self.footstep['new'] = False
        pelvis_is_between_feet = self.pelvis_is_between_feet(current_state)

        if pelvis_is_between_feet and not self.footstep['r_contact'] and r_contact and r_x > self.footstep['r_x'] and r_x > l_x and r_x > self.footstep['l_x'] and self.footstep['last_step'] == 'l':
            footstep_made = True
            self.footstep['last_step'] = 'r'
        elif pelvis_is_between_feet and not self.footstep['l_contact'] and l_contact and l_x > self.footstep['l_x'] and l_x > r_x and l_x > self.footstep['r_x'] and self.footstep['last_step'] == 'r':
            footstep_made = True
            self.footstep['last_step'] = 'l'
        else:
            footstep_made = False

        if footstep_made:
            self.footstep['new'] = True
            self.footstep['n'] += 1
            self.footstep['r_x'] = r_x
            self.footstep['l_x'] = l_x

        self.footstep['r_contact'] = r_contact
        self.footstep['l_contact'] = l_contact

    def pelvis_is_between_feet(self, current_state):
        """
        Checks if the pelvis currently is between the feet.
        """
        pelvis_x = current_state['body_pos']['pelvis'][0]
        r_x = current_state['body_pos']['calcn_r'][0]
        l_x = current_state['body_pos']['calcn_l'][0]
        return min(r_x, l_x) <= pelvis_x <= max(r_x, l_x)

    def calculate_2d_fitness(self, prev_state, current_state, period):
        """
        Calculates the fitness value for 2D simulation.
        """
        result = 0.1
        dt = self.env.osim_model.stepsize

        self.fitness_helpers['footstep_effort'] += self.calculate_current_energy(current_state)

        self.fitness_helpers['footstep_duration'] += dt

        distance_traveled = current_state['body_pos']['pelvis'][0] - prev_state['body_pos']['pelvis'][0]
        self.fitness_helpers['footstep_delta_x'] += distance_traveled

        result += 10 * distance_traveled * np.abs(distance_traveled) / np.linalg.norm(np.array([current_state['body_pos']['pelvis'][0], current_state['body_pos']['pelvis'][2]]) - np.array([prev_state['body_pos']['pelvis'][0], prev_state['body_pos']['pelvis'][2]]))

        right_hip = np.array([current_state['body_pos']['femur_r'][0], current_state['body_pos']['femur_r'][2]])
        left_hip = np.array([current_state['body_pos']['femur_l'][0], current_state['body_pos']['femur_l'][2]])
        right_heel = np.array([current_state['body_pos']['calcn_r'][0], current_state['body_pos']['calcn_r'][2]])
        left_heel = np.array([current_state['body_pos']['calcn_l'][0], current_state['body_pos']['calcn_l'][2]])
        hips = left_hip - right_hip
        hips_unit = hips / np.linalg.norm(hips)
        mid_hip = (left_hip + right_hip) / 2
        mid_hip_right_heel = right_heel - mid_hip
        mid_hip_left_heel = left_heel - mid_hip
        projection_right_heel = np.dot(mid_hip_right_heel, hips_unit)
        projection_left_heel = np.dot(mid_hip_left_heel, hips_unit)
        if projection_right_heel > projection_left_heel:
            result -= 1

        if self.footstep['new']:
            result += 20

            if self.footstep['n'] == 1:
                result += 10 * min(self.fitness_helpers['footstep_duration'], 0.8 * period // 4 * dt)
            elif self.footstep['n'] > 1:
                result += 10 * min(self.fitness_helpers['footstep_duration'], 0.8 * period // 2 * dt)

            # on of the options for calculation of fitness function, used for experiments
            # result += 10 * self.fitness_helpers['footstep_duration']

            result -= self.fitness_helpers['footstep_effort']

            self.fitness_helpers['footstep_duration'] = 0
            self.fitness_helpers['footstep_effort'] = 0
        return result


    def calculate_3d_fitness(self, prev_state, current_state, period, sim_step):
        """
        Calculates the fitness value for 3D simulation.
        """
        result = 0

        if sim_step < period:
            result += 0.1

        if current_state['body_pos']['pelvis'][1] < 0.7:
            result -= 1

        if abs(current_state['body_pos']['pelvis'][2]) > 0.5:
            result -= 1

        if current_state['body_pos']['head'][1] - current_state['body_pos']['pelvis'][1] < 0.4:
            result -= 1

        if abs(current_state['body_pos']['femur_r'][0] - current_state['body_pos']['femur_l'][0]) > 0.12:
            result -= 1

        if current_state['body_pos']['pelvis'][0] < -0.2:
            result -= 1

        dt = self.env.osim_model.stepsize

        self.fitness_helpers['footstep_effort'] += self.calculate_current_energy(current_state)

        self.fitness_helpers['footstep_duration'] += dt

        distance_traveled = current_state['body_pos']['pelvis'][0] - prev_state['body_pos']['pelvis'][0]
        self.fitness_helpers['footstep_delta_x'] += distance_traveled

        result += 30 * distance_traveled * np.abs(distance_traveled) / np.linalg.norm(np.array([current_state['body_pos']['pelvis'][0], current_state['body_pos']['pelvis'][2]]) - np.array([prev_state['body_pos']['pelvis'][0], prev_state['body_pos']['pelvis'][2]]))

        right_hip = np.array([current_state['body_pos']['femur_r'][0], current_state['body_pos']['femur_r'][2]])
        left_hip = np.array([current_state['body_pos']['femur_l'][0], current_state['body_pos']['femur_l'][2]])
        right_heel = np.array([current_state['body_pos']['calcn_r'][0], current_state['body_pos']['calcn_r'][2]])
        left_heel = np.array([current_state['body_pos']['calcn_l'][0], current_state['body_pos']['calcn_l'][2]])
        hips = left_hip - right_hip
        hips_unit = hips / np.linalg.norm(hips)
        mid_hip = (left_hip + right_hip) / 2
        mid_hip_right_heel = right_heel - mid_hip
        mid_hip_left_heel = left_heel - mid_hip
        projection_right_heel = np.dot(mid_hip_right_heel, hips_unit)
        projection_left_heel = np.dot(mid_hip_left_heel, hips_unit)
        if projection_right_heel > projection_left_heel:
            result -= 1

        if self.footstep['new']:
            result += 20
        return result

    def calculate_current_distance_delta(self, prev_state, current_state):
        """
        Calculates the x-axis delta between the previous and the current simulation steps.
        """
        return current_state['body_pos']['pelvis'][0] - prev_state['body_pos']['pelvis'][0]

    def calculate_current_energy(self, current_state):
        """
        Calculates the energy spent by muscle in the current simulation step.
        """
        dt = self.env.osim_model.stepsize
        ACT2 = 0
        for muscle in current_state['muscles'].keys():
            ACT2 += np.square(current_state['muscles'][muscle]['activation'])
        return ACT2 * dt
