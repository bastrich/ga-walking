import math
import numpy as np
import os
import opensim
import random


class OsimModel:
    stepsize = 0.01

    model = None
    state = None
    state0 = None
    joints = []
    bodies = []
    brain = None
    istep = 0

    state_desc_istep = None
    prev_state_desc = None
    state_desc = None

    maxforces = []
    curforces = []

    def __init__(self, model_path, visualize, integrator_accuracy=0.001):
        self.model = opensim.Model(model_path)
        self.model_state = self.model.initSystem()
        self.brain = opensim.PrescribedController()

        # Enable the visualizer
        self.model.setUseVisualizer(visualize)

        self.muscleSet = self.model.getMuscles()
        self.forceSet = self.model.getForceSet()
        self.bodySet = self.model.getBodySet()
        self.jointSet = self.model.getJointSet()
        self.markerSet = self.model.getMarkerSet()
        self.contactGeometrySet = self.model.getContactGeometrySet()

        # Add actuators as constant functions. Then, during simulations
        # we will change levels of constants.
        # One actuartor per each muscle
        for j in range(self.muscleSet.getSize()):
            func = opensim.Constant(1.0)
            self.brain.addActuator(self.muscleSet.get(j))
            self.brain.prescribeControlForActuator(self.muscleSet.get(j).getName(), func)

            self.maxforces.append(self.muscleSet.get(j).getMaxIsometricForce())
            self.curforces.append(1.0)

        self.model.addController(self.brain)
        self.model_state = self.model.initSystem()

        self.integrator_accuracy = integrator_accuracy

    def actuate(self, action):
        if np.any(np.isnan(action)):
            raise ValueError("NaN passed in the activation vector. Values in [0,1] interval are required.")

        action = np.clip(np.array(action), 0.0, 1.0)

        brain = opensim.PrescribedController.safeDownCast(self.model.getControllerSet().get(0))
        functionSet = brain.get_ControlFunctions()

        for j in range(functionSet.getSize()):
            func = opensim.Constant.safeDownCast(functionSet.get(j))
            func.setValue( float(action[j]) )

    def compute_state_desc(self):
        self.model.realizeAcceleration(self.state)

        res = {}

        ## Joints
        res["joint_pos"] = {}
        res["joint_vel"] = {}
        res["joint_acc"] = {}
        for i in range(self.jointSet.getSize()):
            joint = self.jointSet.get(i)
            name = joint.getName()
            # res["joint_pos_my"][name] = [joint.getTransformInGround(self.state).p()[i] for i in range(joint.numCoordinates())]
            res["joint_pos"][name] = [joint.get_coordinates(i).getValue(self.state) for i in range(joint.numCoordinates())]
            res["joint_vel"][name] = [joint.get_coordinates(i).getSpeedValue(self.state) for i in range(joint.numCoordinates())]
            res["joint_acc"][name] = [joint.get_coordinates(i).getAccelerationValue(self.state) for i in range(joint.numCoordinates())]

        ## Bodies
        res["body_pos"] = {}
        res["body_vel"] = {}
        res["body_acc"] = {}
        res["body_pos_rot"] = {}
        res["body_vel_rot"] = {}
        res["body_acc_rot"] = {}
        for i in range(self.bodySet.getSize()):
            body = self.bodySet.get(i)
            name = body.getName()
            res["body_pos"][name] = [body.getTransformInGround(self.state).p()[i] for i in range(3)]
            res["body_vel"][name] = [body.getVelocityInGround(self.state).get(1).get(i) for i in range(3)]
            res["body_acc"][name] = [body.getAccelerationInGround(self.state).get(1).get(i) for i in range(3)]

            res["body_pos_rot"][name] = [body.getTransformInGround(self.state).R().convertRotationToBodyFixedXYZ().get(i) for i in range(3)]
            res["body_vel_rot"][name] = [body.getVelocityInGround(self.state).get(0).get(i) for i in range(3)]
            res["body_acc_rot"][name] = [body.getAccelerationInGround(self.state).get(0).get(i) for i in range(3)]

        ## Forces
        res["forces"] = {}
        for i in range(self.forceSet.getSize()):
            force = self.forceSet.get(i)
            name = force.getName()
            values = force.getRecordValues(self.state)
            res["forces"][name] = [values.get(i) for i in range(values.size())]

        ## Muscles
        res["muscles"] = {}
        for i in range(self.muscleSet.getSize()):
            muscle = self.muscleSet.get(i)
            name = muscle.getName()
            res["muscles"][name] = {}
            res["muscles"][name]["activation"] = muscle.getActivation(self.state)
            res["muscles"][name]["fiber_length"] = muscle.getFiberLength(self.state)
            res["muscles"][name]["fiber_velocity"] = muscle.getFiberVelocity(self.state)
            res["muscles"][name]["fiber_force"] = muscle.getFiberForce(self.state)
            # We can get more properties from here http://myosin.sourceforge.net/2125/classOpenSim_1_1Muscle.html

        ## Markers
        res["markers"] = {}
        for i in range(self.markerSet.getSize()):
            marker = self.markerSet.get(i)
            name = marker.getName()
            res["markers"][name] = {}
            res["markers"][name]["pos"] = [marker.getLocationInGround(self.state)[i] for i in range(3)]
            res["markers"][name]["vel"] = [marker.getVelocityInGround(self.state)[i] for i in range(3)]
            res["markers"][name]["acc"] = [marker.getAccelerationInGround(self.state)[i] for i in range(3)]

        ## Other
        res["misc"] = {}
        res["misc"]["mass_center_pos"] = [self.model.calcMassCenterPosition(self.state)[i] for i in range(3)]
        res["misc"]["mass_center_vel"] = [self.model.calcMassCenterVelocity(self.state)[i] for i in range(3)]
        res["misc"]["mass_center_acc"] = [self.model.calcMassCenterAcceleration(self.state)[i] for i in range(3)]

        return res

    def get_state_desc(self):
        if self.state_desc_istep != self.istep:
            self.prev_state_desc = self.state_desc
            self.state_desc = self.compute_state_desc()
            self.state_desc_istep = self.istep
        return self.state_desc

    def reset_manager(self):
        self.manager = opensim.Manager(self.model)
        self.manager.setIntegratorAccuracy(self.integrator_accuracy)
        self.manager.initialize(self.state)

    def get_state(self):
        return opensim.State(self.state)

    def set_state(self, state):
        self.state = state
        self.istep = int(self.state.getTime() / self.stepsize) # TODO: remove istep altogether
        self.reset_manager()

    def integrate(self):
        # Define the new endtime of the simulation
        self.istep = self.istep + 1

        # Integrate till the new endtime
        self.state = self.manager.integrate(self.stepsize * self.istep)

class SimEnv():

    MASS = 75.16460000000001
    G = 9.80665

    LENGTH0 = 1

    dict_muscle = {
        'abd': 'HAB',
        'add': 'HAD',
        'iliopsoas': 'HFL',
        'glut_max': 'GLU',
        'hamstrings': 'HAM',
        'rect_fem': 'RF',
        'vasti': 'VAS',
        'bifemsh': 'BFSH',
        'gastroc': 'GAS',
        'soleus': 'SOL',
        'tib_ant': 'TA'
    }

    act2mus = [0, 1, 4, 7, 3, 2, 5, 6, 8, 9, 10, 11, 12, 15, 18, 14, 13, 16, 17, 19, 20, 21]
    # maps muscle order in action to muscle order in gait14dof22musc_20170320.osim
    # muscle order in action
    #    HAB, HAD, HFL, GLU, HAM, RF, VAS, BFSH, GAS, SOL, TA
    # muscle order in gait14dof22musc_20170320.osim
    #    HAB, HAD, HAM, BFSH, GLU, HFL, RF, VAS, GAS, SOL, TA
    #    or abd, add, hamstrings, bifemsh, glut_max, iliopsoas, rect_fem, vasti, gastroc, soleus, tib_ant

    INIT_POSE = np.array([
        0, # forward speed
        0, # rightward speed
        0.94, # pelvis height
        0*np.pi/180, # trunk lean
        0*np.pi/180, # [right] hip adduct
        # 0*np.pi/180, # hip flex
        # 0*np.pi/180, # knee extend
        -40 * np.pi / 180,  # hip flex
        -80 * np.pi / 180,  # knee extend
        0*np.pi/180, # ankle flex
        0*np.pi/180, # [left] hip adduct
        0*np.pi/180, # hip flex
        0*np.pi/180, # knee extend
        0*np.pi/180 # ankle flex
    ])

    observation_space = None
    osim_model = None
    verbose = False

    visualize = False

    def __init__(self, mode, visualize, integrator_accuracy=0.001):
        if mode == '3D':
            model_path = os.path.join(os.path.dirname(__file__), '../models/3d.osim')
        elif mode == '2D':
            model_path = os.path.join(os.path.dirname(__file__), '../models/2d.osim')
        else:
            raise ValueError('Invalid mode')

        self.visualize = visualize
        self.osim_model = OsimModel(model_path, self.visualize, integrator_accuracy)

        self.Fmax = {}
        self.lopt = {}
        for leg, side in zip(['r_leg', 'l_leg'], ['r', 'l']):
            self.Fmax[leg] = {}
            self.lopt[leg] = {}
            for MUS, mus in zip(    ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA'],
                                    ['abd', 'add', 'iliopsoas', 'glut_max', 'hamstrings', 'rect_fem', 'vasti', 'bifemsh', 'gastroc', 'soleus', 'tib_ant']):
                muscle = self.osim_model.muscleSet.get('{}_{}'.format(mus,side))
                # Fmax = muscle.getMaxIsometricForce()
                # lopt = muscle.getOptimalFiberLength()
                self.Fmax[leg][MUS] = muscle.getMaxIsometricForce()
                self.lopt[leg][MUS] = muscle.getOptimalFiberLength()

        self.last_x = 0
        self.delta_of_last_step = 0

        self.last_line_idx = None

    def reset(self):
        self.t = 0

        self.last_x = 0
        self.delta_of_last_step = 0

        # initialize state
        self.osim_model.state = self.osim_model.model.initializeState()
        init_pose = self.INIT_POSE
        state = self.osim_model.get_state()
        QQ = state.getQ()
        QQDot = state.getQDot()
        for i in range(17):
            QQDot[i] = 0
        QQ[3] = 0 # x: (+) forward
        QQ[5] = 0 # z: (+) right
        QQ[1] = 0*np.pi/180 # roll
        QQ[2] = 0*np.pi/180 # yaw
        QQDot[3] = init_pose[0] # forward speed
        QQDot[5] = init_pose[1] # forward speed
        QQ[4] = init_pose[2] # pelvis height
        QQ[0] = -init_pose[3] # trunk lean: (+) backward
        QQ[7] = -init_pose[4] # right hip abduct
        QQ[6] = -init_pose[5] # right hip flex
        QQ[13] = init_pose[6] # right knee extend
        QQ[15] = -init_pose[7] # right ankle flex
        QQ[10] = -init_pose[8] # left hip adduct
        QQ[9] = -init_pose[9] # left hip flex
        QQ[14] = init_pose[10] # left knee extend
        QQ[16] = -init_pose[11] # left ankle flex

        state.setQ(QQ)
        state.setU(QQDot)
        self.osim_model.set_state(state)
        self.osim_model.model.equilibrateMuscles(self.osim_model.state)

        self.osim_model.state.setTime(0)
        self.osim_model.istep = 0

        self.osim_model.reset_manager()

        self.last_line_idx = None

        return self.get_state_desc()

    def step(self, action):
        action = [action[i] for i in self.act2mus]

        self.osim_model.actuate(action)
        self.osim_model.integrate()

        return self.get_state_desc()



    def get_state_desc(self):
        return self.osim_model.get_state_desc()