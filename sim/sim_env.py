# this code is partially based on the code from https://github.com/stanfordnmbl/osim-rl/blob/master/osim/env/osim.py

import numpy as np
from sim.osim_model import OsimModel


class SimEnv:
    """
    Encapsulates technical details of OpenSim env and model.
    """

    act2mus = [0, 1, 4, 7, 3, 2, 5, 6, 8, 9, 10, 11, 12, 15, 18, 14, 13, 16, 17, 19, 20, 21]

    def __init__(self, mode, visualize, integrator_accuracy=0.001):
        self.osim_model = OsimModel(mode, visualize, integrator_accuracy)

        if mode == '2D':
            self.init_pose = np.array([
                0, # forward speed
                0, # rightward speed
                0.94, # pelvis height
                0*np.pi/180, # trunk lean
                0*np.pi/180, # [right] hip adduct
                -40 * np.pi / 180,  # hip flex
                -80 * np.pi / 180,  # knee extend
                0*np.pi/180, # ankle flex
                0*np.pi/180, # [left] hip adduct
                0*np.pi/180, # hip flex
                0*np.pi/180, # knee extend
                0*np.pi/180 # ankle flex
            ])
        else:
            self.init_pose = np.array([
                0, # forward speed
                0, # rightward speed
                0.94, # pelvis height
                0*np.pi/180, # trunk lean
                0*np.pi/180, # [right] hip adduct
                0*np.pi/180, # hip flex
                0*np.pi/180, # knee extend
                0*np.pi/180, # ankle flex
                0*np.pi/180, # [left] hip adduct
                0*np.pi/180, # hip flex
                0*np.pi/180, # knee extend
                0*np.pi/180 # ankle flex
            ])

    def reset(self):
        # initialize state
        self.osim_model.state = self.osim_model.model.initializeState()
        state = self.osim_model.get_state()
        QQ = state.getQ()
        QQDot = state.getQDot()
        for i in range(17):
            QQDot[i] = 0
        QQ[3] = 0 # x: (+) forward
        QQ[5] = 0 # z: (+) right
        QQ[1] = 0*np.pi/180 # roll
        QQ[2] = 0*np.pi/180 # yaw
        QQDot[3] = self.init_pose[0] # forward speed
        QQDot[5] = self.init_pose[1] # forward speed
        QQ[4] = self.init_pose[2] # pelvis height
        QQ[0] = -self.init_pose[3] # trunk lean: (+) backward
        QQ[7] = -self.init_pose[4] # right hip abduct
        QQ[6] = -self.init_pose[5] # right hip flex
        QQ[13] = self.init_pose[6] # right knee extend
        QQ[15] = -self.init_pose[7] # right ankle flex
        QQ[10] = -self.init_pose[8] # left hip adduct
        QQ[9] = -self.init_pose[9] # left hip flex
        QQ[14] = self.init_pose[10] # left knee extend
        QQ[16] = -self.init_pose[11] # left ankle flex

        state.setQ(QQ)
        state.setU(QQDot)
        self.osim_model.set_state(state)
        self.osim_model.model.equilibrateMuscles(self.osim_model.state)

        self.osim_model.state.setTime(0)
        self.osim_model.istep = 0

        self.osim_model.reset_manager()

        return self.get_state_desc()

    def step(self, action):
        action = [action[i] for i in self.act2mus]
        self.osim_model.actuate(action)
        self.osim_model.integrate()
        return self.get_state_desc()

    def get_state_desc(self):
        return self.osim_model.compute_state_desc()