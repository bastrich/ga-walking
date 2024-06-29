from sim_env import SimEnv
import numpy as np
import time

class Sim:

    def __init__(self, visualize=False):
        self.env = SimEnv(visualize=visualize)
        self.fitness_helpers = {}
        self.footstep = {}
        self.footstep['n'] = 0
        self.footstep['new'] = False
        self.footstep['r_contact'] = 1
        self.footstep['l_contact'] = 1

    def run(self, walking_strategy, number_of_steps=1000):
        self.init_fitness_helpers()
        fitness = 0
        prev_state = self.env.reset()

        start_time = time.time()

        for sim_step in range(number_of_steps):

            current_state = self.env.step(walking_strategy.get_muscle_activations(sim_step))

            if self.is_failed(current_state) or time.time() - start_time > 10:
                break

            self.update_footstep(current_state)
            fitness += self.calculate_current_fitness(prev_state, current_state)

            prev_state = current_state

        if not self.is_failed(prev_state) and sim_step >= number_of_steps - 1:
            fitness += 50

        return fitness, sim_step

    def is_failed(selfself, state):
        return state['body_pos']['pelvis'][1] < 0.6

    def init_fitness_helpers(self):
        self.fitness_helpers['alive'] = 0.1
        self.fitness_helpers['footstep_effort'] = 0
        self.fitness_helpers['footstep_duration'] = 0
        self.fitness_helpers['footstep_delta_x'] = 0

    def update_footstep(self, current_state):
        r_contact = True if current_state['forces']['foot_r'][1] < -0.05*(self.env.MASS*self.env.G) else False
        l_contact = True if current_state['forces']['foot_l'][1] < -0.05*(self.env.MASS*self.env.G) else False

        self.footstep['new'] = False
        if (not self.footstep['r_contact'] and r_contact) or (not self.footstep['l_contact'] and l_contact):
            self.footstep['new'] = True
            self.footstep['n'] += 1

        self.footstep['r_contact'] = r_contact
        self.footstep['l_contact'] = l_contact

    def calculate_current_fitness(self, prev_state, current_state):
        result = 0

        dt = self.env.osim_model.stepsize

        # alive fitness to search for success
        result += self.fitness_helpers['alive']

        # effort ~ muscle fatigue ~ (muscle activation)^2
        ACT2 = 0
        for muscle in current_state['muscles'].keys():
            ACT2 += np.square(current_state['muscles'][muscle]['activation'])
        self.fitness_helpers['footstep_effort'] += ACT2*dt

        self.fitness_helpers['footstep_duration'] += dt

        distance_traveled = current_state['body_pos']['pelvis'][0] - prev_state['body_pos']['pelvis'][0]
        self.fitness_helpers['footstep_delta_x'] += distance_traveled

        # reward += 40 * distance_traveled !!!!!!!

        # reward from velocity (penalize from deviating from v_tgt)

        # p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
        # v_body = [state_desc['body_vel']['pelvis'][0], -state_desc['body_vel']['pelvis'][2]]
        # v_tgt = self.vtgt.get_vtgt(p_body).T

        # self.d_reward['footstep']['del_v'] += (v_body - v_tgt)*dt

        # prev_pelvis_head = np.array(self.get_prev_state_desc()['body_pos']['head']) - np.array(self.get_prev_state_desc()['body_pos']['pelvis'])
        # prev_projection_pelvis_head = np.linalg.norm([prev_pelvis_head[0], prev_pelvis_head[2]])
        # prev_pelvis_head_angle = np.arctan2(prev_projection_pelvis_head, prev_pelvis_head[1])
        #
        # pelvis_head = np.array(state_desc['body_pos']['head']) - np.array(state_desc['body_pos']['pelvis'])
        # projection_pelvis_head = np.linalg.norm([pelvis_head[0], pelvis_head[2]])
        # current_pelvis_head_angle = np.arctan2(projection_pelvis_head, pelvis_head[1])

        # reward += -10 * (current_pelvis_head_angle - prev_pelvis_head_angle) !!!!!!!!!!

        right_hip = np.array([current_state['body_pos']['femur_r'][0], current_state['body_pos']['femur_r'][2]])
        left_hip = np.array([current_state['body_pos']['femur_l'][0], current_state['body_pos']['femur_l'][2]])
        right_heel = np.array([current_state['body_pos']['calcn_r'][0], current_state['body_pos']['calcn_r'][2]])
        left_heel = np.array([current_state['body_pos']['calcn_l'][0], current_state['body_pos']['calcn_l'][2]])

        hips = left_hip - right_hip
        hips_unit = hips / np.linalg.norm(hips)
        mid_hip = (left_hip + right_hip) / 2
        mid_hip_right_heel = right_heel - mid_hip
        mid_hip_left_heel = left_heel - mid_hip

        # Находим проекции векторов mid_C и mid_D на единичный вектор направления AB
        projection_right_heel = np.dot(mid_hip_right_heel, hips_unit)
        projection_left_heel = np.dot(mid_hip_left_heel, hips_unit)

        if projection_right_heel < projection_left_heel:
            # print('NOT crossing')
            result += 0.05
        elif projection_right_heel > projection_left_heel:
            # print('crossing')
            result -= 0.1

        result += 10 * distance_traveled


        #
        # if self.visualize:
        #     # def add_custom_decorations(system, state):
        #     #     # Создаем декорацию линии
        #     #     line = opensim.simbody.DecorativeLine(state_desc['joint_pos']['hip_l'], state_desc['body_pos']['calcn_l'])
        #     #     line.setColor(opensim.Vec3(0, 1, 0))  # Устанавливаем цвет линии (красный)
        #     #     line.setLineThickness(0.1)
        #     #
        #     #     # Добавляем декорацию к системе
        #     #     system.addDecoration(state, line)
        #
        #     self.visualize
        #
        #     line = opensim.simbody.DecorativeLine(
        #         # opensim.Vec3(-state_desc['joint_pos']['hip_l'][0], state_desc['joint_pos']['hip_l'][1], -state_desc['joint_pos']['hip_l'][2]),
        #         # opensim.Vec3(-state_desc['body_pos']['calcn_l'][0], state_desc['body_pos']['calcn_l'][1], -state_desc['body_pos']['calcn_l'][2])
        #         opensim.Vec3(0, 0, 0),
        #         opensim.Vec3(state_desc['body_pos']['femur_l'])
        #     )
        #     line.setColor(opensim.Green)  # Устанавливаем цвет линии (красный)
        #     line.setLineThickness(5)
        #
        #     # if self.last_line_idx is not None:
        #     #     self.visualizer.updDecoration(self.last_line_idx).setOpacity(255)
        #     self.last_line_idx = self.visualizer.addDecoration(0, opensim.Transform(), line)

        # b = (right_heel[0] - right_hip[0])*(left_heel[1] - left_hip[1]) - (right_heel[1] - right_hip[1])*(left_heel[0] - left_hip[0])
        # a = (right_heel[0] - right_hip[0])*(right_hip[1] - left_hip[1]) - (right_heel[1] - right_hip[1])*(right_hip[0] - left_hip[0])
        # c = (left_heel[0] - left_hip[0])*(right_hip[1] - left_hip[1]) - (left_heel[1] - left_hip[1])*(right_hip[0] - left_hip[0])
        #
        # legs_crossing = False
        # if b == 0:
        #     legs_crossing = False
        # elif 0 <= a/b <= 1 or 0 <= c/b <= 1:
        #     legs_crossing = True



        # limit reward for big steps
        # add np.roll mutation
        ## add direction to fitness value

        # footstep reward (when made a new step)
        if self.footstep['new']:
            if self.fitness_helpers['footstep_delta_x'] > 0:
                result += 10 * self.fitness_helpers['footstep_duration']
                # step_size = np.abs(current_state['body_pos']['calcn_r'][0] - current_state['body_pos']['calcn_l'][0])
                # result += 10 * np.exp(step_size)  #???
                result += 10 * np.exp(self.fitness_helpers['footstep_delta_x'])
            # result += 10 * self.fitness_helpers['footstep_delta_x']
            # reward += 10 * np.exp(self.delta_of_last_step)

            # panalize effort
            result -= self.fitness_helpers['footstep_effort']


            self.fitness_helpers['footstep_duration'] = 0
            self.fitness_helpers['footstep_effort'] = 0
            self.fitness_helpers['footstep_delta_x'] = 0

        return result

