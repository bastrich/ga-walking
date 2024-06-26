from sim_env import SimEnv
import numpy as np

class Sim:

    def __init__(self, visualize=False):
        self.env = SimEnv(visualize=visualize)
        self.fitness_value_helper_state = {}

    def run(self, walking_strategy, number_of_steps=1000):
        fitness = 0
        prev_state = self.env.reset()
        for sim_step in range(number_of_steps):

            current_state = self.env.step(walking_strategy.get_muscle_activations(sim_step))

            if current_state['body_pos']['pelvis'][1] < 0.6:
                break

            fitness += self.calculate_current_fitness(prev_state, current_state)

    def init_reward(self):
        self.d_reward = {}

        self.d_reward['weight'] = {}
        self.d_reward['weight']['footstep'] = 10
        self.d_reward['weight']['effort'] = 100

        self.d_reward['alive'] = 0.1
        self.d_reward['effort'] = 0

        self.d_reward['footstep'] = {}
        self.d_reward['footstep']['effort'] = 0
        self.d_reward['footstep']['del_t'] = 0
        self.d_reward['footstep']['del_v'] = 0

    def calculate_current_fitness(self, prev_state, current_state):
        result = 0

        dt = self.osim_model.stepsize

        # alive reward
        # should be large enough to search for 'success' solutions (alive to the end) first
        reward += self.d_reward['alive']

        # effort ~ muscle fatigue ~ (muscle activation)^2
        ACT2 = 0
        for muscle in sorted(state_desc['muscles'].keys()):
            ACT2 += np.square(state_desc['muscles'][muscle]['activation'])
        self.d_reward['effort'] += ACT2*dt
        self.d_reward['footstep']['effort'] += ACT2*dt

        self.d_reward['footstep']['del_t'] += dt

        distance_traveled = state_desc['body_pos']['pelvis'][0] - self.last_x
        self.last_x = state_desc['body_pos']['pelvis'][0]

        # reward += 40 * distance_traveled !!!!!!!

        self.delta_of_last_step += distance_traveled

        # reward from velocity (penalize from deviating from v_tgt)

        p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
        v_body = [state_desc['body_vel']['pelvis'][0], -state_desc['body_vel']['pelvis'][2]]
        # v_tgt = self.vtgt.get_vtgt(p_body).T

        # self.d_reward['footstep']['del_v'] += (v_body - v_tgt)*dt

        prev_pelvis_head = np.array(self.get_prev_state_desc()['body_pos']['head']) - np.array(self.get_prev_state_desc()['body_pos']['pelvis'])
        prev_projection_pelvis_head = np.linalg.norm([prev_pelvis_head[0], prev_pelvis_head[2]])
        prev_pelvis_head_angle = np.arctan2(prev_projection_pelvis_head, prev_pelvis_head[1])

        pelvis_head = np.array(state_desc['body_pos']['head']) - np.array(state_desc['body_pos']['pelvis'])
        projection_pelvis_head = np.linalg.norm([pelvis_head[0], pelvis_head[2]])
        current_pelvis_head_angle = np.arctan2(projection_pelvis_head, pelvis_head[1])

        # reward += -10 * (current_pelvis_head_angle - prev_pelvis_head_angle) !!!!!!!!!!

        right_hip = np.array([state_desc['body_pos']['femur_r'][0], state_desc['body_pos']['femur_r'][2]])
        left_hip = np.array([state_desc['body_pos']['femur_l'][0], state_desc['body_pos']['femur_l'][2]])
        right_heel = np.array([state_desc['body_pos']['calcn_r'][0], state_desc['body_pos']['calcn_r'][2]])
        left_heel = np.array([state_desc['body_pos']['calcn_l'][0], state_desc['body_pos']['calcn_l'][2]])

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
            reward += 0.1
        elif projection_right_heel > projection_left_heel:
            # print('crossing')
            reward -= 0.1



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
        # try to make shrink growth with amoubnt of coefficients
        # delete creatures that are less than minimum from previous population
        # sconstrain inital options to binary combinations of muscles activation
        # not minus small decrease in distance
        # may be dynamic fitness function
        # penalize for up knee higher than pelvis
        # increase population size and queue in thread pool

        # footstep reward (when made a new step)
        if self.footstep['new']:
            reward += 10 * self.d_reward['footstep']['del_t']
            reward += 10 * self.delta_of_last_step

            # footstep reward: so that solution does not avoid making footsteps
            # scaled by del_t, so that solution does not get higher rewards by making unnecessary (small) steps
            # reward_footstep_0 = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']

            # deviation from target velocity
            # the average velocity a step (instead of instantaneous velocity) is used
            # as velocity fluctuates within a step in normal human walking
            #reward_footstep_v = -self.reward_w['v_tgt']*(self.footstep['del_vx']**2)
            # reward_footstep_v = -self.d_reward['weight']['v_tgt']*np.linalg.norm(self.d_reward['footstep']['del_v'])/self.LENGTH0
            # reward_footstep_v = 20 * self.last_x


            # panalize effort
            reward += -self.d_reward['footstep']['effort']
            #
            # reward += 10 * np.exp(self.delta_of_last_step)
            #
            self.d_reward['footstep']['del_t'] = 0
            # self.d_reward['footstep']['del_v'] = 0
            self.d_reward['footstep']['effort'] = 0
            self.delta_of_last_step = 0


            # reward += reward_footstep_0

        # success bonus
        if not self.is_done() and (self.osim_model.istep >= self.time_limit): #and self.failure_mode is 'success':
            # retrieve reward (i.e. do not penalize for the simulation terminating in a middle of a step)
            #reward_footstep_0 = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']
            #reward += reward_footstep_0 + 100
            reward += 10

        return reward