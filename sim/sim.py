from sim.sim_env import SimEnv
import numpy as np
import time

class Sim:

    def __init__(self, mode, visualize):
        self.visualize = visualize
        self.mode = mode
        self.env = SimEnv(mode, visualize)
        self.fitness_helpers = {}
        self.footstep = {}

    def run(self, walking_strategy, number_of_steps=1000):
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
            fitness += self.calculate_current_fitness(prev_state, current_state, walking_strategy.period)
            distance += self.calculate_current_distance_delta(prev_state, current_state)
            energy += self.calculate_current_energy(current_state)

            prev_state = current_state

        return np.round(fitness, 2), sim_step + 1, distance, energy

    def is_failed(selfself, state):
        return state['body_pos']['pelvis'][1] < 0.6

    def init_fitness_helpers(self):
        self.fitness_helpers['alive'] = 0.1
        self.fitness_helpers['footstep_effort'] = 0
        self.fitness_helpers['footstep_duration'] = 0
        self.fitness_helpers['footstep_delta_x'] = 0
        self.fitness_helpers['footstep_delta_v'] = np.array([0.0, 0.0])
        self.fitness_helpers['footstep_side_error'] = 0
        self.fitness_helpers['footstep_x_error'] = 0

    def init_footstep_info(self, state):
        self.footstep['n'] = 0
        self.footstep['new'] = False
        self.footstep['r_contact'] = True
        self.footstep['l_contact'] = False
        self.footstep['r_x'] = state['body_pos']['calcn_r'][0]
        self.footstep['l_x'] = state['body_pos']['calcn_l'][0]
        self.footstep['last_step'] = 'l'

    def update_footstep(self, current_state):
        r_contact = True if current_state['forces']['foot_r'][1] < -0.05*(self.env.MASS*self.env.G) else False
        l_contact = True if current_state['forces']['foot_l'][1] < -0.05*(self.env.MASS*self.env.G) else False
        r_x = current_state['body_pos']['calcn_r'][0]
        l_x = current_state['body_pos']['calcn_l'][0]

        self.footstep['new'] = False
        pelvis_is_between_foots = self.pelvis_is_between_foots(current_state)

        if pelvis_is_between_foots and not self.footstep['r_contact'] and r_contact and r_x > self.footstep['r_x'] and r_x > l_x and r_x > self.footstep['l_x'] and self.footstep['last_step'] == 'l':
            footstep_made = True
            self.footstep['last_step'] = 'r'
        elif pelvis_is_between_foots and not self.footstep['l_contact'] and l_contact and l_x > self.footstep['l_x'] and l_x > r_x and l_x > self.footstep['r_x'] and self.footstep['last_step'] == 'r':
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

    def pelvis_is_between_foots(self, current_state):
        pelvis_x = current_state['body_pos']['pelvis'][0]
        r_x = current_state['body_pos']['calcn_r'][0]
        l_x = current_state['body_pos']['calcn_l'][0]
        return min(r_x, l_x) <= pelvis_x <= max(r_x, l_x)

    def calculate_current_fitness(self, prev_state, current_state, period):
        result = 0.1
        dt = self.env.osim_model.stepsize

        self.fitness_helpers['footstep_effort'] += self.calculate_current_energy(current_state)

        self.fitness_helpers['footstep_duration'] += dt

        distance_traveled = current_state['body_pos']['pelvis'][0] - prev_state['body_pos']['pelvis'][0]
        self.fitness_helpers['footstep_delta_x'] += distance_traveled

        result += 10 * distance_traveled * np.abs(distance_traveled) / np.linalg.norm(np.array([current_state['body_pos']['pelvis'][0], current_state['body_pos']['pelvis'][2]]) - np.array([prev_state['body_pos']['pelvis'][0], prev_state['body_pos']['pelvis'][2]]))

        if self.mode == '3D':
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

    def calculate_current_distance_delta(self, prev_state, current_state):
        return current_state['body_pos']['pelvis'][0] - prev_state['body_pos']['pelvis'][0]

    def calculate_current_energy(self, current_state):
        dt = self.env.osim_model.stepsize
        ACT2 = 0
        for muscle in current_state['muscles'].keys():
            ACT2 += np.square(current_state['muscles'][muscle]['activation'])
        return ACT2 * dt

