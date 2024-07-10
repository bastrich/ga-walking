from sim_env import SimEnv
import numpy as np
import time

class Sim:

    def __init__(self, mode, visualize):
        self.visualize = visualize
        self.env = SimEnv(mode, visualize)
        self.fitness_helpers = {}
        self.footstep = {}

    def run(self, walking_strategy, number_of_steps=1000):
        prev_state = self.env.reset()

        self.init_fitness_helpers()
        self.init_footstep_info(prev_state)
        fitness = 0


        start_time = time.time()

        for sim_step in range(number_of_steps):

            current_state = self.env.step(walking_strategy.get_muscle_activations(sim_step))

            # if self.is_failed(current_state) or (time.time() - start_time > 10 and not self.visualize):
            if self.is_failed(current_state) or time.time() - start_time > 10:
                break

            self.update_footstep(current_state)
            fitness += self.calculate_current_fitness(prev_state, current_state)

            prev_state = current_state

            # time.sleep(0.1)

        # if not self.is_failed(prev_state) and sim_step >= number_of_steps - 1:
        #     fitness += 50

        return fitness, sim_step

    def is_failed(selfself, state):
        return state['body_pos']['pelvis'][1] < 0.6

    def init_fitness_helpers(self):
        self.fitness_helpers['alive'] = 0.1
        self.fitness_helpers['footstep_effort'] = 0
        self.fitness_helpers['footstep_duration'] = 0
        self.fitness_helpers['footstep_delta_x'] = 0
        self.fitness_helpers['footstep_delta_v'] = 0
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
            print(f'new footstep - {self.footstep["n"]}')


        self.footstep['r_contact'] = r_contact
        self.footstep['l_contact'] = l_contact

    def pelvis_is_between_foots(self, current_state):
        pelvis_x = current_state['body_pos']['pelvis'][0]
        r_x = current_state['body_pos']['calcn_r'][0]
        l_x = current_state['body_pos']['calcn_l'][0]
        return min(r_x, l_x) <= pelvis_x <= max(r_x, l_x)

    def calculate_original_fitness_from_NIPS_challenge(self, prev_state, current_state):
        result = 0

        dt = self.env.osim_model.stepsize

        result += 0.1

        result += 10 * (current_state['body_pos']['pelvis'][0] - prev_state['body_pos']['pelvis'][0]) #not original, added by me

        velocity = np.array([current_state['body_vel']['pelvis'][0], -current_state['body_vel']['pelvis'][2]])
        target_velocity = np.array([1.4, 0])

        self.fitness_helpers['footstep_delta_v'] += (velocity - target_velocity)*dt

        ACT2 = 0
        for muscle in current_state['muscles'].keys():
            ACT2 += np.square(current_state['muscles'][muscle]['activation'])
        self.fitness_helpers['footstep_effort'] += ACT2*dt

        if self.footstep['new']:
            reward_footstep_0 = 10 * self.fitness_helpers['footstep_duration']
            reward_footstep_v = -np.linalg.norm(self.fitness_helpers['footstep_delta_v'])
            reward_footstep_e = -self.fitness_helpers['footstep_effort']

            self.fitness_helpers['footstep_duration'] = 0
            self.fitness_helpers['footstep_delta_v'] = 0
            self.fitness_helpers['footstep_effort'] = 0

            result += reward_footstep_0 + reward_footstep_v + reward_footstep_e

        return result

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

        self.fitness_helpers['footstep_side_error'] += np.abs(current_state['body_vel']['pelvis'][2])
        self.fitness_helpers['footstep_x_error'] += np.abs(distance_traveled) if distance_traveled < 0 else 0

        # result -= 10 * np.abs(distance_traveled) if distance_traveled < 0 else 0
        # result -= 10 * np.abs(current_state['body_vel']['pelvis'][2])

        result += 10 * distance_traveled #!!!!!!!

        # reward from velocity (penalize from deviating from v_tgt)

        # p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
        velocity = np.array([current_state['body_vel']['pelvis'][0], -current_state['body_vel']['pelvis'][2]])
        # velocity_magnitude = np.linalg.norm(velocity)
        # if velocity_magnitude != 0:
        #     unit_velocity = velocity / velocity_magnitude
        # else:
        #     unit_velocity = 0

        target_velocity = np.array([0.5, 0])

        self.fitness_helpers['footstep_delta_v'] += (velocity - target_velocity)*dt


        # print(unit_velocity)
        # print(np.linalg.norm(unit_velocity))
        # v_tgt = self.vtgt.get_vtgt(p_body).T

        # self.d_reward['footstep']['del_v'] += (v_body - v_tgt)*dt

        # prev_pelvis_head = np.array(prev_state['body_pos']['head']) - np.array(prev_state['body_pos']['pelvis'])
        # prev_projection_pelvis_head = np.linalg.norm([prev_pelvis_head[0], prev_pelvis_head[2]])
        # prev_pelvis_head_angle = np.arctan2(prev_projection_pelvis_head, prev_pelvis_head[1])
        #
        # pelvis_head = np.array(current_state['body_pos']['head']) - np.array(current_state['body_pos']['pelvis'])
        # projection_pelvis_head = np.linalg.norm([pelvis_head[0], pelvis_head[2]])
        # current_pelvis_head_angle = np.arctan2(projection_pelvis_head, pelvis_head[1])

        # result += -10 * (current_pelvis_head_angle - prev_pelvis_head_angle) #!!!!!!!!!!

        # right_hip = np.array([current_state['body_pos']['femur_r'][0], current_state['body_pos']['femur_r'][2]])
        # left_hip = np.array([current_state['body_pos']['femur_l'][0], current_state['body_pos']['femur_l'][2]])
        # right_heel = np.array([current_state['body_pos']['calcn_r'][0], current_state['body_pos']['calcn_r'][2]])
        # left_heel = np.array([current_state['body_pos']['calcn_l'][0], current_state['body_pos']['calcn_l'][2]])
        #
        # hips = left_hip - right_hip
        # hips_unit = hips / np.linalg.norm(hips)
        # mid_hip = (left_hip + right_hip) / 2
        # mid_hip_right_heel = right_heel - mid_hip
        # mid_hip_left_heel = left_heel - mid_hip
        #
        # # Находим проекции векторов mid_C и mid_D на единичный вектор направления AB
        # projection_right_heel = np.dot(mid_hip_right_heel, hips_unit)
        # projection_left_heel = np.dot(mid_hip_left_heel, hips_unit)
        #
        # if projection_right_heel < projection_left_heel:
        #     # print('NOT crossing')
        #     result += 0.1
        # elif projection_right_heel > projection_left_heel:
        #     # print('crossing')
        #     result -= 0.1

        # result += 10 * distance_traveled


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
            result += 20

            # if self.fitness_helpers['footstep_delta_x'] > 0:
            result += 10 * self.fitness_helpers['footstep_duration']
            # result += 50
                # step_size = np.abs(current_state['body_pos']['calcn_r'][0] - current_state['body_pos']['calcn_l'][0])
                # result += 10 * np.exp(step_size)  #???
            # result += 100 * np.sign(self.fitness_helpers['footstep_delta_x']) * np.square(self.fitness_helpers['footstep_delta_x'])
            # result += 10 * np.exp(current_state['body_vel']['pelvis'][0])
            # result += 100 * self.fitness_helpers['footstep_delta_x']
            # reward += 10 * np.exp(self.delta_of_last_step)

            # result -= 10 * self.fitness_helpers['footstep_side_error'] * np.exp(self.fitness_helpers['footstep_side_error'])
            # result -= 10 * self.fitness_helpers['footstep_x_error'] * np.exp(self.fitness_helpers['footstep_x_error'])

            # panalize effort
            result -= self.fitness_helpers['footstep_effort']


            # result -= np.linalg.norm(self.fitness_helpers['footstep_delta_v'])


            self.fitness_helpers['footstep_duration'] = 0
            self.fitness_helpers['footstep_effort'] = 0
            self.fitness_helpers['footstep_delta_x'] = 0
            self.fitness_helpers['footstep_delta_v'] = 0
            self.fitness_helpers['footstep_side_error'] = 0
            self.fitness_helpers['footstep_x_error'] = 0

        return result

