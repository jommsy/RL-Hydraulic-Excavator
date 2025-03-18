# modification of dm_control/suite/manipulator/planar_manipulator.py
# ============================================================================

"""Excavator domain."""

import collections
from dm_control import mujoco
from dm_control.suite import base
from domain_task.rewards import custom_rewards as rewards
from dm_control.suite import manipulator
import os
import numpy as np
import matlab.engine
from domain_task.control_env import custom_control_env
import traceback
from dm_control.mujoco import wrapper
import domain_task.utils as utils
from abc import ABC, abstractmethod

_CONTROL_TIMESTEP = [0.08, 0.04, 0.02, 0.01, 0.005]  # (Seconds)
_STEP_LIMIT = 600
_JOINTS = ['cabin', 'boom', 'arm', 'bucket']
_DOF_ADDRESS = [0, 1, 2, 3]
_COMPONENT_LENGTH = {
    "o1a": 0.297059,
    "o1b": 1.181326,
    "o2c": 0.238248,
    "o2d": 1.396755,
    "gh": 0.234974,
    "gf": 0.970948,
    "o3f": 1.1254,
    "o3g": 0.157623,
    "o3e": 0.187843,
    "eh": 0.225615
}
_INIT_CYLINDER_LENGTH = [1.236, 1.288, 0.9] #[boom, arm, bucket]
_INIT_POS = [0, 1.236, 1.288, 0.9] #[boom, arm, bucket]
_INIT_ALPHA = [1.633416166871895, 1.019263853533036, 1.652802643862622] #[boom, arm, bucket]
_CONSTANT_THETA = [0.03, 2.925065]
_LENGTH_INDICES = [("o1b", "o1a"), ("o2c", "o2d"), ("gf", "gh")]
# for test
_CYLINDER_ROD_PAIRS = [('boom_cylinder', 'boom_rod'),
                    ('arm_cylinder', 'arm_rod'),
                    ('blink1', 'blink2')]
_SENSOR_POS = ['jointpos_swing', 'jointpos_boom', 'jointpos_arm', 'jointpos_bucket']
_SENSOR_VEL = ['jointvel_swing', 'jointvel_boom', 'jointvel_arm', 'jointvel_bucket']


def model_path():
    return os.path.join(os.path.dirname(__file__), "../assets/excavator.xml")

@manipulator.SUITE.add('custom')
def swing(init_mj_angle, fully_observable=True, random=None,
                     environment_kwargs=None):
    """Returns excavator move bucket to specified point before load task."""
    # compute according init pos: init_sim_pos
    disp_bias, init_sim_pos = utils.compute_init_sim_pos(init_mj_angle)
    physics = Physics.from_xml_path(model_path(), disp_bias, init_sim_pos)
    task = Swing(init_mj_angle, fully_observable=fully_observable, random=random)
    environment_kwargs = environment_kwargs or {}
    return custom_control_env(physics, task, step_limit=_STEP_LIMIT, **environment_kwargs)

@manipulator.SUITE.add('custom')
def digging(init_mj_angle, fully_observable=True, random=None,
                     environment_kwargs=None):
    """Returns excavator load soil task."""
    
    disp_bias, init_sim_pos = utils.compute_init_sim_pos(init_mj_angle)
    physics = Physics.from_xml_path(model_path(), disp_bias, init_sim_pos)
    task = Digging(init_mj_angle, fully_observable=fully_observable, random=random)
    environment_kwargs = environment_kwargs or {}
    return custom_control_env(physics, task, step_limit=_STEP_LIMIT, **environment_kwargs)

@manipulator.SUITE.add('custom')
def lifting(init_mj_angle, fully_observable=True, random=None,
                     environment_kwargs=None):
    """Returns excavator move bucket to specified point after load task"""
    disp_bias, init_sim_pos = utils.compute_init_sim_pos(init_mj_angle)
    physics = Physics.from_xml_path(model_path(), disp_bias, init_sim_pos)
    task = Lifting(init_mj_angle, fully_observable=fully_observable, random=random)
    environment_kwargs = environment_kwargs or {}
    return custom_control_env(physics, task, step_limit=_STEP_LIMIT, **environment_kwargs)

@manipulator.SUITE.add('custom')
def digging_soil(init_mj_angle, fully_observable=True, random=None,
                     environment_kwargs=None):
    """Returns excavator load soil task."""
    
    disp_bias, init_sim_pos = utils.compute_init_sim_pos(init_mj_angle)
    physics = Physics.from_xml_path(model_path(), disp_bias, init_sim_pos)
    task = DiggingSoil(init_mj_angle, fully_observable=fully_observable, random=random)
    environment_kwargs = environment_kwargs or {}
    return custom_control_env(physics, task, step_limit=_STEP_LIMIT, **environment_kwargs)

@manipulator.SUITE.add('custom')
def hauling(init_mj_angle, fully_observable=True, random=None,
                     environment_kwargs=None):
    """Returns excavator move bucket to specified point after load task"""
    disp_bias, init_sim_pos = utils.compute_init_sim_pos(init_mj_angle)
    physics = Physics.from_xml_path(model_path(), disp_bias, init_sim_pos)
    task = Hauling(init_mj_angle, fully_observable=fully_observable, random=random)
    environment_kwargs = environment_kwargs or {}
    return custom_control_env(physics, task, step_limit=_STEP_LIMIT, **environment_kwargs)



class SimulinkPlant:
    """ This class is used to connect to the Simulink Model and simulate the hydraulic system"""

    def __init__(self, disp_bias, init_sim_pos, model_name='hydraulic_model'):
        # The name of the Simulink Model (To be placed in the same directory as the Python Code)
        self.model_name = model_name
        self._disp_bias = disp_bias
        self._init_sim_pos = init_sim_pos

    def connectToMatlab(self):
        print("Starting matlab")
        """ TEST: use GUI matlab to monitor the simulation """
        process_id = matlab.engine.find_matlab()
        self.eng = matlab.engine.connect_matlab(process_id[0])

        """ real use """
        print("Connected to Matlab")
        work_dir = os.path.join(os.path.dirname(__file__), "../matlab_model")
        self.eng.cd(work_dir, nargout=0)
        try:
            self.eng.init_condition(matlab.double(self._disp_bias), matlab.double(self._init_sim_pos), nargout=0)
            self.eng.init_model(nargout=0)
        except Exception as e:
            print("Error: ", e)
            print("Closing MATLAB process due to error...")
            self.eng.quit()
        print("start model and set as fastrestart mode")

    def simulate(self, com_timestep, eng_time, u, loads):
        try:
            return self.eng.sim_step(com_timestep, eng_time, matlab.single(u), matlab.single(loads), nargout=2)
        except Exception as e:
            print("Error: ", e)
            print("Closing MATLAB process due to error...")
            self.eng.quit()
            
    def disconnect(self):
        self.eng.set_param(
            self.model_name, 'SimulationCommand', 'stop', nargout=0)
        self.eng.quit()

    def reset(self):
        self.eng.reset_model(nargout=0)
        
    def random_reset(self, disp_bias, init_sim_pos):
        self.eng.init_condition(matlab.double(disp_bias), matlab.double(init_sim_pos), nargout=0)
        self.eng.reset_model(nargout=0)


class Physics(mujoco.Physics):
    def __init__(self, data, disp_bias, init_sim_pos):
        # 调用父类的构造函数
        super().__init__(data)
        # 添加新的 self.eng 对象
        self._init_sim_pos = init_sim_pos
        self.eng = SimulinkPlant(disp_bias, init_sim_pos)
        self.eng.connectToMatlab()
        self.truncation = False
        self.lengths = _INIT_CYLINDER_LENGTH.copy()
        self.action = np.zeros(4)
        self.sim_pressure_diff = np.zeros(4) # [MPa]
        self.sim_pos = np.zeros(4)
        self.sim_vel = np.zeros(4)
        self._spoolPos = np.arange(0, 5.1, 0.1)  # [mm]
        self._BiPAreaSwing = np.concatenate(([1e-4, 1e-4], 3.6 * self._spoolPos[2:]))  # [mm^2]
        self._BiPAreaBoom = np.concatenate(([1e-4, 1e-4], 2.0 * self._spoolPos[2:]))   # [mm^2]
        self._BiPAreaArm = np.concatenate(([1e-4, 1e-4], 1.6 * self._spoolPos[2:]))    # [mm^2]
        self._BiPAreaBucket = np.concatenate(([1e-4, 1e-4], 1.5 * self._spoolPos[2:])) # [mm^2]
        self._pressure_drop = 17e5  # [Pa]
        self._oil_density = 870     # [kg/m^3]
        self._discharge_coef = 0.64
        
    @classmethod
    def from_xml_path(cls, file_path, disp_bias, init_sim_pos):
        model = wrapper.MjModel.from_xml_path(file_path)
        return cls.from_model(model, disp_bias, init_sim_pos)
    
    @classmethod
    def from_model(cls, model, disp_bias, init_sim_pos):
        data = wrapper.MjData(model)
        return cls(data, disp_bias, init_sim_pos)
    
    def joint_qpos(self, joint_names):
        """Returns joint angles."""
        return self.named.data.qpos[joint_names]
    
    def bounded_joint_qpos(self, joint_names):
        """Returns joint positions as (sin, cos) values."""
        joint_qpos = self.named.data.qpos[joint_names]
        return np.vstack([np.sin(joint_qpos), np.cos(joint_qpos)]).T

    def joint_qvel(self, joint_names):
        """Returns joint velocities."""
        return self.named.data.qvel[joint_names]

    def body_3d_xpose(self, body_names, orientation=True):
        """Returns positions and/or orientations of bodies."""
        if not isinstance(body_names, str):
            # Broadcast indices.
            body_names = np.array(body_names).reshape(-1, 1)
        pos = self.named.data.xpos[body_names, ['x', 'y', 'z']]
        if orientation:
            ori = self.named.data.xquat[body_names, ['qw', 'qx', 'qy', 'qz']]
            return np.hstack([pos, ori])
        else:
            return pos

    def site_distance(self, site1, site2):
        site1_to_site2 = np.diff(
            self.named.data.site_xpos[[site2, site1]], axis=0)
        return np.linalg.norm(site1_to_site2)

    def get_sensor_by_name(self, sensor_name):
        """Returns site 3D linear accelerations in global coordinate."""
        # sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        sensor_id = self.model.name2id(sensor_name, 'sensor')
        sensor_adr = self.model.sensor_adr[sensor_id]
        sensor_dim = self.model.sensor_dim[sensor_id]
        sensor_data = self.data.sensordata[sensor_adr:sensor_adr + sensor_dim]
        return sensor_data

    """
    17.11: test the synchronization between the Simulink and Mujoco
    """
    def test_sync(self, sim_pos, sim_vel):
        mj_distance_tmp = np.array([self.site_distance(pair[0], pair[1]) for pair in _CYLINDER_ROD_PAIRS])
        # set a site in g, the last paris of cylinder and rod is ('e', 'g')
        O3GE = self.cos_theorem(_COMPONENT_LENGTH['o3g'], mj_distance_tmp[2], _COMPONENT_LENGTH['o3e'], 'theta')
        EGH = self.cos_theorem(mj_distance_tmp[2], _COMPONENT_LENGTH['gh'], _COMPONENT_LENGTH['eh'], 'theta')
        HGF = _CONSTANT_THETA[1] - O3GE - EGH
        mj_distance_tmp[-1] = self.cos_theorem(_COMPONENT_LENGTH['gh'], _COMPONENT_LENGTH['gf'], HGF, 'length')
        mj_angle_cabin = self.named.data.sensordata['jointpos_swing']
        mj_distance = np.insert(mj_distance_tmp, 0, mj_angle_cabin)
        mj_disp =  mj_distance - _INIT_POS
        error_disp = mj_disp - sim_pos
        mj_vel = self.named.data.sensordata[_SENSOR_VEL]
        error_vel = mj_vel - sim_vel
        return mj_disp, mj_vel, error_disp, error_vel
                
    def before_step(self, action, com_timestep):
        """Send the control signal to simulink and get the force to mujoco."""
        self.action = action.copy()
        eng_time = np.round(self.data.time / self.model.opt.timestep) * self.model.opt.timestep + com_timestep
        
        torque_loads = np.round((self.data.qfrc_constraint[_DOF_ADDRESS] - self.data.qfrc_bias[_DOF_ADDRESS]).astype(np.float32), 8)
        force2sim = np.array(self.torque2force(torque_loads), dtype=np.float32) / 1e3
        min_values = np.array([-10.8, -71.0, -54, -54.0])  # 每个元素的最小值
        max_values = np.array([10.8, 55.0, 38.0, 38.0])     # 每个元素的最大值
        force2sim = np.array(np.round(np.clip(force2sim, min_values, max_values), 2), dtype=np.float32)
        
        sim_pos, sim_vel = self.eng.simulate(com_timestep, eng_time, action, force2sim)
        
        if sim_pos == matlab.double([]) or sim_vel == matlab.double([]):
            self.truncation = True
            print("get null matrix from simulink")
            return
        else:
            self.truncation = False
            sim_pos = np.array(sim_pos, dtype=np.float32).flatten()
            sim_vel = np.array(sim_vel, dtype=np.float32).flatten()
            alphas, omegas = self.pos_vel_2_alpha_omega(sim_pos, sim_vel)
            
            self.sim_pos = sim_pos.copy()
            self.sim_vel = sim_vel.copy()
            
            self.data.ctrl[0] = alphas[0] * 1000 # swing_pos
            self.data.ctrl[1] = omegas[0] * 1000 # swing_vel
            self.data.ctrl[2] = (_INIT_ALPHA[0] - alphas[1]) * 1000 # boom_pos
            self.data.ctrl[3] = -omegas[1] * 1000 # boom_vel
            self.data.ctrl[4] = (alphas[2] - _INIT_ALPHA[1]) * 1000 # arm_pos
            self.data.ctrl[5] = omegas[2] * 1000 # arm_vel
            self.data.ctrl[6] = (alphas[3] - _INIT_ALPHA[2]) * 1000 # bucket_pos
            if not np.isnan(omegas[3]):
                self.data.ctrl[7] = omegas[3] * 1000 # bucket_vel
                
        # # 03.17 test sync        
        # mj_disp, mj_vel, error_disp, error_vel = self.test_sync(sim_pos, sim_vel)
        # attach_xvel = self.get_sensor_by_name('attach_linvel')
        # attach_vel = np.sqrt(attach_xvel[0]**2 + attach_xvel[1]**2 + attach_xvel[2]**2) 
        # with open (f"test/sync/sync_0317.csv", 'a') as f:
        #     f.write(f"{self.data.time}, {attach_vel}, {sim_pos}, {mj_disp}, {error_disp}, {sim_vel}, {mj_vel}, {error_vel}\n")
            
    def _step_with_up_to_date_position_velocity(self, nstep: int = 1, soil_resistance=[0, 0, 0]) -> None:
        """Physics step with up-to-date position and velocity dependent fields."""
        # In the case of Euler integration we assume mj_step1 has already been
        # called for this state, finish the step with mj_step2 and then update all
        # position and velocity related fields with mj_step1. This ensures that
        # (most of) mjData is in sync with qpos and qvel. In the case of non-Euler
        # integrators (e.g. RK4) an additional mj_step1 must be called after the
        # last mj_step to ensure mjData syncing.
        
        mujoco.mj_applyFT(
                    self.model.ptr,
                    self.data.ptr,
                    soil_resistance,  # 力的大小
                    [0,0,0],  # 扭矩（这里不施加）
                    self.data.ptr.site_xpos[16],  # 应用力的位置
                    7,  # 物体ID
                    self.data.ptr.qfrc_applied,  # 应用力的数组
                )
        if self.model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4.value:
            mujoco.mj_step2(self.model.ptr, self.data.ptr)
            if nstep > 1:
                mujoco.mj_step(self.model.ptr, self.data.ptr, nstep-1)
        else:
            mujoco.mj_step(self.model.ptr, self.data.ptr, nstep)
        mujoco.mj_step1(self.model.ptr, self.data.ptr)
        mujoco.mju_zero(self.data.ptr.qfrc_applied)
        
    def step(self, nstep: int = 1, soil_resistance=[0, 0, 0]) -> None:
        """Advances the physics state by `nstep`s.

        Args:
        nstep: Optional integer, number of steps to take.

        The actuation can be updated by calling the `set_control` function first.
        """
        with self.check_invalid_state():
            if self.legacy_step:
                self._step_with_up_to_date_position_velocity(nstep, soil_resistance)
            else:
                mujoco.mj_step(self.model.ptr, self.data.ptr, nstep)
        
                
    def eng_reset(self):
        self.eng.reset()
        
    def eng_random_reset(self, disp_bias, init_sim_pos):
        self.eng.random_reset(disp_bias, init_sim_pos)
        
    def get_joint_id(self, joint_name):
        return self.model.name2id(joint_name, 'joint')
    
    def torque2force(self, torque_loads):
        lengths = self.lengths
        thetas = []
        force2sim = [torque_loads[0]]
        # compute theta of boom and arm
        for i, (length, (idx1, idx2)) in enumerate(zip(lengths, _LENGTH_INDICES)):
            theta = self.cos_theorem(length, _COMPONENT_LENGTH[idx1], _COMPONENT_LENGTH[idx2], 'theta')
            thetas.append(theta)
        # compute theta of bucket    
        theta_4 = thetas[2] - _CONSTANT_THETA[0]
        O3h = self.cos_theorem(lengths[2], _COMPONENT_LENGTH["o3f"], theta_4, 'length')
        sin_theta_3 = _COMPONENT_LENGTH["o3f"] * np.sin(theta_4) / O3h
        sin_thetas = np.sin(thetas)
        sin_thetas[-1] = sin_theta_3
        # compute torque2force
        moment_arms = np.array([_COMPONENT_LENGTH["o1b"], _COMPONENT_LENGTH["o2c"], O3h])
        for i, (sin_theta, moment_arm) in enumerate(zip(sin_thetas, moment_arms)):
            load = torque_loads[i+1] / (moment_arm * sin_theta)
            force2sim.append(load)
        return force2sim      
    
    def pos_vel_2_alpha_omega(self, rod_pos, rod_vel):
        alphas = [rod_pos[0]]
        # compute alpha of boom and arm
        lengths = self.lengths
        for i, length in enumerate(_INIT_CYLINDER_LENGTH):
            lengths[i] = length + rod_pos[i+1]

        for i, (length, (idx1, idx2)) in enumerate(zip(lengths, _LENGTH_INDICES)):
            alpha = self.cos_theorem(_COMPONENT_LENGTH[idx1], _COMPONENT_LENGTH[idx2], length, 'theta')
            alphas.append(alpha)
            
        O3GH = _CONSTANT_THETA[1] - alphas[3]
        O3h = self.cos_theorem(_COMPONENT_LENGTH["gh"], _COMPONENT_LENGTH["o3g"], O3GH, 'length')
        alpha = self.sin_theorem(_COMPONENT_LENGTH["gh"], O3h, O3GH) + self.cos_theorem(_COMPONENT_LENGTH["o3e"], O3h, _COMPONENT_LENGTH["eh"], 'theta')
        alphas[-1] = alpha
        omegas = self.vel2omega(rod_pos, rod_vel)
        return alphas, omegas
    
    @staticmethod
    def cos_theorem(a, b, c, flag):
        if flag == 'theta':
            cos_theta = (a**2 + b**2 - c**2) / (2 * a * b)
            if cos_theta < -1 or cos_theta > 1:
                raise ValueError(f"无效的 cos 值：{cos_theta}, 不能计算 arccos。")
            return np.arccos(cos_theta)
        elif flag == 'length':
            return np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(c))
    
    @staticmethod
    def sin_theorem(a, b, beta):
        return np.arcsin(a * np.sin(beta) / b)
    
    @staticmethod
    # TODO: process the omega and vel, protect from being interrupted by the singular invalid point
    def vel2omega(rod_pos, rod_vel):
        omegas = []
        omega_swing = rod_vel[0]
        omegas.append(omega_swing)
        omega_boom = (2.84962*(rod_pos[1] + 1.236)*rod_vel[1])/(1.0 - 1.0*(1.42481*(rod_pos[1] + 1.236)**2 - 2.1141)**2)**(1/2)
        omegas.append(omega_boom)
        omega_arm = (3.00504*(rod_pos[2] + 1.288)*rod_vel[2])/(1.0 - 1.0*(1.50252*(rod_pos[2] + 1.288)**2 - 3.01659)**2)**(1/2)
        omegas.append(omega_arm)
        term_6 = 10.0 * rod_pos[3] + 9.0
        term_2 = np.arccos(2.187 - 0.02192 * term_6**2) - 2.925
        cos_t2 = np.cos(term_2)
        sin_t2 = np.sin(term_2)
        t_1_t_3 = np.sqrt(10560 - (2.252 * term_6**2 - 224.7)**2) * (2.884 - 2.669 * cos_t2)**(3/2)
        term_4 = 2.669 * cos_t2 - 2.884
        expr_1 = term_6 * rod_vel[3] * (5.338 * cos_t2**2 - 5.769 * cos_t2 + 2.669 * sin_t2**2) * 1.004 * 10**(1.5)
        expr_2 = np.sqrt(1.989 * sin_t2**2 / term_4 + 1.0) * t_1_t_3
        expr_3 = sin_t2 * term_6 * rod_vel[3] * (2.669 * cos_t2 - 3.447) * 8.428 * 10**(0.5)
        expr_4 = np.sqrt((0.1967 * (2.669 * cos_t2 - 2.322)**2) / term_4 + 1.0) * t_1_t_3
        omega_bucket = expr_1 / expr_2 + expr_3 / expr_4
        omegas.append(omega_bucket)
        return omegas
    
    def flow_rate(self, action):
        spool_disp = np.abs(action) * 5.0  # [mm]
        BiPAreaSwing_interp = np.interp(spool_disp[0], self._spoolPos, self._BiPAreaSwing)  # [mm^2]
        BiPAreaBoom_interp = np.interp(spool_disp[1], self._spoolPos, self._BiPAreaBoom)    # [mm^2]
        BiPAreaArm_interp = np.interp(spool_disp[2], self._spoolPos, self._BiPAreaArm)      # [mm^2]
        BiPAreaBucket_interp = np.interp(spool_disp[3], self._spoolPos, self._BiPAreaBucket) # [mm^2]
        area = np.array([BiPAreaSwing_interp, BiPAreaBoom_interp, BiPAreaArm_interp, BiPAreaBucket_interp]) * 1e-6 # [m^2]
        flow_rate = self._discharge_coef * area * np.sqrt(2 * self._pressure_drop / self._oil_density) * 1e3 * 60  # [L/min]
        total_flow_rate = np.sum(flow_rate)
        return total_flow_rate

class BaseTask(base.Task, ABC):
    def __init__(self, init_mj_angle, fully_observable, random):
        if not hasattr(self, '_effector') or not hasattr(self, '_target'):
            raise NotImplementedError("Subclasses must implement '_effector' and '_target' properties.")
        self._fully_observable = fully_observable
        self._init_angles = init_mj_angle
        self._ground = np.array([0.0, -1.0], dtype=np.float32)
        self.previous_action = np.zeros(4)
        self.termination = False
        self.truncation = False
        self._visualize_reward = True
        self._theresold_target = [1000, 4, 1, 0.5, 0.1]
        
        self.old_attach_pos = None
        self.old_filling_ratio = 0.0
        self.old_attach_acc = None
        self.attach_pos = None
        self.bottom_pos = None
        self.soil_capacity = 0.15 #m^2
        self.filling_ratio = 0.0
        self.beta = 0.0
        self.termination_flag = 0
        self.prj_xy = None 
              
        self.soil_phi = 0.3 # 内摩擦角 0.3 ~0.8 [rad]
        self.soil_rho = 1400 # 密度 1400 ~ 1800 [kg/m^3]
        self.soil_C = 10 # 粘度系数 0 ~ 50 [kPa]
        self.soil_delta = 0.3 # 外摩擦角 0.2 ~ 0.4 [rad]
        self.soil_gamma = 0.5 # 滑裂面倾角 0.3 ~ 0.8 [rad]
        self.soil_Ca = 4 # 粘附度系数 0 ~ 10 [kPa]
              
        self.soil_w = 0.35 # 铲斗宽度 [m]
        self.soil_epsilon = None # 斗刃切入角
        self.soil_epsilon_c = None # 斗侧刃切入角
        self.soil_w_c = 0.02 # 斗侧刃宽度 [m] 
        self.soil_z = 0 # 土壤高度 [m]
        self.theta_c = 1.05 # 底板与斗侧夹角 [rad]
        self.g = 9.8 # 重力加速度 [m/s^2]
        
        super().__init__(random=random)

    @property
    @abstractmethod
    def _effector(self):
        """Abstract property for effector."""
        pass

    @property
    @abstractmethod
    def _target(self):
        """Abstract property for target."""
        pass

    @abstractmethod
    def get_reward(self, physics, action):
        """Compute task-specific reward components."""
        pass

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        pass

    def get_n_steps(self, physics):
        # distance_target = physics.site_distance(self._effector, self._target)
        site_xpos = physics.named.data.site_xpos[[self._effector, 'bucket_bottom']]
        distance_ground = min(site_xpos[0][2], site_xpos[1][2])
        # for i, value in enumerate(self._theresold_target):
        #     if distance_target < value:
        #         index = i
        index = 2
        if distance_ground < 0.3:
            index = max(index, 2)
        control_timestep = _CONTROL_TIMESTEP[index]
        com_timestep = control_timestep
        n_sub_steps = self.compute_n_steps(com_timestep, physics.timestep())
        n_sub_steps_control = self.compute_n_steps(control_timestep, com_timestep)
        return control_timestep, com_timestep, n_sub_steps, n_sub_steps_control

    @staticmethod
    def compute_n_steps(com_timestep, physics_timestep, tolerance=1e-8):
        if com_timestep < physics_timestep:
            raise ValueError(
                'Communicate timestep ({}) cannot be smaller than physics timestep ({}).'.
                format(com_timestep, physics_timestep))
        if abs((com_timestep / physics_timestep - round(
            com_timestep / physics_timestep))) > tolerance:
            raise ValueError(
                'Communicate timestep ({}) must be an integer multiple of physics timestep '
                '({})'.format(com_timestep, physics_timestep))
        return int(round(com_timestep / physics_timestep))

    def get_observation(self, physics):
        self.get_pre_observation(physics)
        obs = collections.OrderedDict()
        obs['joint_bounded_pos'] = np.round(physics.bounded_joint_qpos(_JOINTS).astype(np.float32), 8)
        obs['joint_vel'] = np.round(physics.joint_qvel(_JOINTS).astype(np.float32), 8)
        obs['bucket_ori'] = np.round(physics.body_3d_xpose('bucket').astype(np.float32), 8)
        obs['target_pos'] = np.round(physics.named.data.site_xpos[self._target], 8)
        obs['ground'] = self._ground
        obs['pre_action'] = self.previous_action
        return obs
    
    def get_pre_observation(self, physics):
        attach_xpos = physics.named.data.site_xpos['bucket_attach']
        bottom_xpos = physics.named.data.site_xpos['bucket_bottom']
        attach_pos = np.array([np.sign(attach_xpos[0]) * (np.sqrt(attach_xpos[0]**2 + attach_xpos[1]**2)), attach_xpos[2]], dtype=np.float32)
        bottom_pos = np.array([np.sign(bottom_xpos[0]) * (np.sqrt(bottom_xpos[0]**2 + bottom_xpos[1]**2)), bottom_xpos[2]], dtype=np.float32)
        self.attach_pos = attach_pos
        self.bottom_pos = bottom_pos
        if self.old_attach_pos is None:
            self.old_attach_pos = self.attach_pos.copy()
            if attach_pos[1] < 0.0:
                self.filling_ratio = np.random.uniform(0.0, abs(attach_pos[1]))
                self.old_filling_ratio = self.filling_ratio
        else:
            self.get_filling_ratio(attach_pos, bottom_pos)
        if not np.array_equal(self.old_attach_pos, self.attach_pos):
            v1 = self.attach_pos - bottom_pos
            v2 = self.attach_pos - self.old_attach_pos
            beta = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
            cross_product = np.cross(v1, v2)
            if cross_product > 0.0:
                beta = -beta
            self.beta = beta

    # calculate the soil volume
    def get_filling_ratio(self, attach_pos, bottom_pos):
        delta_x = attach_pos[0] - self.old_attach_pos[0]
        if attach_pos[1] < 0.0:
            delta_filling_ratio = (np.abs(attach_pos[1]) * np.abs(delta_x)) / self.soil_capacity
            if delta_x < 0.0:
                self.filling_ratio += delta_filling_ratio
            else:
                self.filling_ratio -= delta_filling_ratio
        elif attach_pos[1] >= 0.0:
            if attach_pos[1] - bottom_pos[1] < 0.0:
                self.filling_ratio -= 0.1* self.filling_ratio
            else:
                self.filling_ratio = self.filling_ratio
        self.filling_ratio = np.clip(self.filling_ratio, 0.0, 1.0)
        
    def TH0(a, b):
        if a ==0 and b == 0:
            return 0
        else:
            return (a * b) / (a +  b - a * b)

    @staticmethod
    def np_cot(x, eps=1e-10):
        return 1 / (np.tan(x) + eps)
        
    def get_termination(self, physics):
        """
        Truncation condition:
        if simulink can't converge in 1000 times wait, then return sim_pos = [[]] sim_vel = [[]], and end the episode.
        Termination condition:
        reach goal state or the episode is terminated.
        """
        if physics.truncation or self.truncation:
            physics.trubcation = False
            self.truncation = False
            return 1.0
        elif self.termination:
            self.termination = False
            # modify as None to close terminate
            # return 0.0
            return None
    
    def soil_resistance(self, physics):
        bucket_attach = physics.named.data.site_xpos['bucket_attach']
        bucket_bottom = physics.named.data.site_xpos['bucket_bottom']
        
        attach_radial = np.sign(bucket_attach[0]) * np.hypot(bucket_attach[0], bucket_attach[1])
        bottom_radial = np.sign(bucket_bottom[0]) * np.hypot(bucket_bottom[0], bucket_bottom[1])
        attach_pos = np.array([attach_radial, bucket_attach[2]], dtype=np.float32)
        bottom_pos = np.array([bottom_radial, bucket_bottom[2]], dtype=np.float32)
        
        attach_vel_vec = physics.get_sensor_by_name('attach_linvel')
        attach_vel = np.linalg.norm(attach_vel_vec)
        
        if self.prj_xy is None:
            self.prj_xy = np.arctan2(bucket_attach[1], bucket_attach[0])
        
        F1 = -self.filling_ratio * self.soil_capacity * self.soil_w * self.soil_rho * self.g
        
        if np.array_equal(self.old_attach_pos, attach_pos):
            return [0, 0, F1]
        
        z = self.soil_z - attach_pos[1]
        
        F2_XY = F2_Z = 0.0
        F3_XY = F3_Z = F4_XY = F4_Z = 0.0
        
        sin_phi = np.sin(self.soil_phi)
        cos_phi = np.cos(self.soil_phi)
        
        # 底板阻力
        if attach_pos[1] < self.soil_z or bottom_pos[1] < self.soil_z:
            numerator = 1 - sin_phi * np.cos(2 * self.beta)
            denominator = 1 - sin_phi
            F2 = (numerator / denominator) * (
                0.5 * z * self.soil_rho +
                self.soil_C / np.tan(self.soil_phi) +
                ((z * self.soil_rho - self.soil_w * self.soil_rho * np.sin(self.beta)) *
                (1 - sin_phi) / (1 + sin_phi))
            )

            delta_x = self.old_attach_pos[0] - attach_pos[0]
            delta_z = self.old_attach_pos[1] - attach_pos[1]
            alpha2 = np.arctan2(delta_z, delta_x)
            F2_XY = F2 * np.cos(alpha2)
            F2_Z = F2 * np.sin(alpha2)
        
        if attach_pos[1] < self.soil_z:
            if self.soil_epsilon is None:
                self.soil_epsilon = np.arctan2((bottom_pos[1] - attach_pos[1]), (bottom_pos[0] - attach_pos[0]))
                self.soil_epsilon_c = self.soil_epsilon + self.theta_c
            
            # 斗刃切削阻力
            angle_sum = self.soil_epsilon + self.soil_phi + self.soil_delta + self.soil_gamma
            F3 = (self.soil_w * z) / np.sin(angle_sum) * (
                np.sin(self.soil_gamma + self.soil_phi) * (
                    (self.soil_C + self.soil_Ca) / np.tan(self.soil_phi) +
                    (z * self.soil_rho * self.g * np.sin(self.soil_gamma + self.soil_epsilon)) /
                    (2 * np.sin(self.soil_gamma) * np.sin(self.soil_epsilon))
                ) +
                np.sin(2 * (self.soil_gamma + self.soil_phi)) * (
                    self.soil_C / np.tan(self.soil_gamma) + self.soil_Ca / np.tan(self.soil_epsilon)
                ) +
                self.soil_rho * min(abs(attach_vel), 2.5)**2 * cos_phi *
                np.sin(self.soil_epsilon) / np.sin(self.soil_epsilon + self.soil_gamma)
            )
            # 
            delta_x_bottom = bottom_pos[0] - attach_pos[0]
            delta_z_bottom = bottom_pos[1] - attach_pos[1]
            alpha3 = np.arctan2(delta_z_bottom, delta_x_bottom) - np.pi / 2
            F3_XY = F3 * np.cos(alpha3)
            F3_Z = F3 * np.sin(alpha3)
            
            # 斗侧切削阻力
            V_a = 0.5 * z**2 * (self.np_cot(self.soil_gamma) + self.np_cot(self.soil_epsilon_c))
            term1 = (2/3) * self.soil_rho * self.g * z * V_a * (1 - sin_phi)
            term2 = (self.soil_C * self.soil_w_c * z) / np.sin(self.soil_gamma)
            term3 = 2 * self.soil_C * V_a
            term4 = self.soil_rho * self.g * self.soil_w_c * V_a * np.sin(self.soil_gamma + self.soil_phi)
            term5 = self.soil_Ca * self.soil_w_c * z * np.sin(self.soil_epsilon_c + self.soil_phi + self.soil_gamma)
            
            F4 = (1 / np.sin(self.soil_epsilon_c + self.soil_phi + self.soil_gamma + self.soil_delta)) * (
                (term1 + term2 + term3) * cos_phi + term4 - term5
            )
            alpha4 = alpha3 + self.theta_c
            F4_XY = F4 * np.cos(alpha4)
            F4_Z = F4 * np.sin(alpha4)
        
        F_XY = F2_XY + F3_XY + F4_XY
        F_X = F_XY * np.cos(self.prj_xy)
        F_Y = F_XY * np.sin(self.prj_xy)
        F_Z = F1 + F2_Z + F3_Z + F4_Z

        self.old_attach_pos = attach_pos.copy()
        
        return [F_X, F_Y, F_Z]
    
    
class Swing(BaseTask):
    """A Move `Task`: move the bucket_attach site to dig_point site."""
    def __init__(self, init_mj_angle, fully_observable, random):
        """
        Args:
            init_mj_angle: Initial joint angles for the excavator.
            fully_observable: A `bool`, whether the observation should contain
                              all position and velocity information (always True).
            random: Optional random seed or instance for initializing randomness.
        """
        self._joint_lower_1 = np.array([-2.4, -0.511, -0.842, -0.894])
        self._joint_upper_1 = np.array([-2, -0.361, -0.692, -0.744])
        self._joint_lower_2 = np.array([2, 0.566, 1.56, 1.05])
        self._joint_upper_2 = np.array([2.4, 0.716, 1.71, 1.2])
        self._vel_lower = np.array([-1.428, -0.185, -0.214, -0.184])
        self._vel_upper = np.array([1.428, 0.143, 0.15, 0.129])
        self._cylinder_lower = np.array([-0.75 * np.pi, -0.15, -0.125, -0.15])
        self._cylinder_upper = np.array([0.75 * np.pi, 0.17, 0.33, 0.20])
        super().__init__(init_mj_angle, fully_observable, random)
        
    @property
    def _effector(self):
        return 'bucket_attach'
    
    @property
    def _target(self): 
        return 'dig_point'

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.old_attach_pos = None
        self.old_filling_ratio = 0.0
        self.old_attach_acc = None
        self.attach_pos = None
        self.bottom_pos = None
        self.filling_ratio = 0.0
        self.beta = 0.0
        
        # int_angles = np.array([0.45,
        #                        np.random.uniform(0, 0.2),
        #                        np.random.uniform(-0.1, 0.35),
        #                        np.random.uniform(-0.1, 0.3)])
        
        data = physics.named.data
        data.qpos[_JOINTS] = self._init_angles
        physics.eng_reset()
        super().initialize_episode(physics)
        
    def get_reward(self, physics, action, control_timestep=None, step_count = 0, num_timesteps=None):
        """ TODO: accumulative reward; consider add reward with respect to timestep
                higer reward with less time, indicating the agent to reduce the valve opening area when
            a cylinder arriving the position limit, to avoid the according load is too large to slowing down other cylinders.
            TODO: consider the update method of PPO, and if it's necessary to regularize the acummulative reward of an episode.
        """
        
        distance_target = physics.site_distance(self._effector, self._target)
        mj_vel = np.round(physics.joint_qvel(_JOINTS).astype(np.float32), 8)
        site_xpos_safe = physics.named.data.site_xpos[['bucket_attach', 'bucket_bottom']]
        total_flow_rate = physics.flow_rate(action)
        
        reward_target = rewards.is_close(0.1, 1.0, distance_target, sigmoid='long_tail')
        reward_constraint = rewards.joint_limit(0.05, 0.05, self._joint_upper_1, self._joint_lower_1, self._joint_upper_2, self._joint_lower_2, 
                                                np.round(physics.joint_qpos(_JOINTS).astype(np.float32), 8))
        reward_safe = rewards.is_collision(site_xpos_safe, (-1, 0), 0.15, 0.05, (-1, 0), 0.25, 0.05)
        reward_load = rewards.load_condition(self._vel_lower, self._vel_upper, self._cylinder_lower, self._cylinder_upper, 
                                             physics.sim_pos, physics.sim_vel, action)
        self.termination, reward_goal = rewards.p1_goal_state(distance_target, 0.05, mj_vel, 0.02, site_xpos_safe, 0.2)
        reward_flow = rewards.max_flow_rate(total_flow_rate, (0.0, 55.0), 2.5, 0.2)
        self.previous_action = action.copy()
        reward = 2 * reward_target + reward_flow + reward_constraint / 3 + reward_safe / 2 + reward_load / 2 + reward_goal
        return reward


class Digging(BaseTask):
    def __init__(self, init_mj_angle, fully_observable, random):
        self._init_angles = init_mj_angle
        self._joint_lower_1 = np.array([-2.4, -0.511, -0.842, -0.894])
        self._joint_upper_1 = np.array([0.5, -0.361, -0.692, -0.744])
        self._joint_lower_2 = np.array([0.6, 0.566, 1.56, 1.05])
        self._joint_upper_2 = np.array([2.4, 0.716, 1.71, 1.2])
        self._vel_lower = np.array([-1.428, -0.185, -0.214, -0.184])
        self._vel_upper = np.array([1.428, 0.143, 0.15, 0.129])
        self._cylinder_lower = np.array([-0.75 * np.pi, -0.15, -0.125, -0.15])
        self._cylinder_upper = np.array([0.75 * np.pi, 0.17, 0.33, 0.20])
        super().__init__(init_mj_angle, fully_observable, random)
        
    @property
    def _effector(self):
        return 'bucket_attach'
    
    @property
    def _target(self):
        return 'deep_point'
    
    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.old_attach_pos = None
        self.old_filling_ratio = 0.0
        self.old_attach_acc = None
        self.attach_pos = None
        self.bottom_pos = None
        self.filling_ratio = 0.0
        self.beta = 0.0
        
        # int_angles = np.array([0.45,
        #                        np.random.uniform(0, 0.2),
        #                        np.random.uniform(-0.1, 0.35),
        #                        np.random.uniform(-0.1, 0.3)])
        
        data = physics.named.data
        data.qpos[_JOINTS] = self._init_angles
        physics.eng_reset()
        super().initialize_episode(physics)

        
    def get_reward(self, physics, action, control_timestep=None, step_count = 0, num_timesteps=None):
        distance_target = physics.site_distance(self._effector, self._target)
        site_xpos_ori = physics.named.data.site_xpos[['bucket', 'bucket_attach']]
        site_xpos_safe = physics.named.data.site_xpos[['bucket_attach', 'bucket_bottom']]
        mj_vel = np.round(physics.joint_qvel(_JOINTS).astype(np.float32), 8)
        total_flow_rate = physics.flow_rate(action) 
        reward_target = rewards.is_close(0.1, 1.0, distance_target, sigmoid='long_tail')
        reward_bucket_ori = rewards.bucket_orientation(site_xpos_ori, (0.5, 0.7), 0.05, 0.1, flag='deep') - 1
        self.previous_action = action.copy()
        reward_constraint = rewards.joint_limit(0.05, 0.05, self._joint_upper_1, self._joint_lower_1, self._joint_upper_2, self._joint_lower_2, 
                                                np.round(physics.joint_qpos(_JOINTS).astype(np.float32), 8))
        reward_safe = rewards.is_collision(site_xpos_safe, (-1, -0.9), 0.1, 0.05, (-1, -0.9), 0.15, 0.05)
        reward_load = rewards.load_condition(self._vel_lower, self._vel_upper, self._cylinder_lower, self._cylinder_upper, 
                                             physics.sim_pos, physics.sim_vel, action)
        self.termination, reward_goal = rewards.p2_goal_state(distance_target, 0.1, mj_vel, 0.02, site_xpos_ori, (0.5, 0.7))
        reward_flow = rewards.max_flow_rate(total_flow_rate, (0.0, 55.0), 2.5, 0.2)
        reward = 3.5 * reward_target + reward_bucket_ori + reward_flow +\
            reward_constraint / 3 + reward_safe / 2 + reward_load / 2 + reward_goal
        return reward

class Lifting(BaseTask):
    def __init__(self, init_mj_angle, fully_observable, random):
        self._init_angles = init_mj_angle
        self._joint_lower_1 = np.array([-2.4, -0.511, -0.842, -0.894])
        self._joint_upper_1 = np.array([-2, -0.361, -0.692, -0.744])
        self._joint_lower_2 = np.array([2, 0.566, 1.56, 1.05])
        self._joint_upper_2 = np.array([2.4, 0.716, 1.71, 1.2])
        self._vel_lower = np.array([-1.428, -0.185, -0.214, -0.184])
        self._vel_upper = np.array([1.428, 0.143, 0.15, 0.129])
        self._cylinder_lower = np.array([-0.75 * np.pi, -0.15, -0.125, -0.15])
        self._cylinder_upper = np.array([0.75 * np.pi, 0.17, 0.33, 0.20])
        super().__init__(init_mj_angle, fully_observable, random)
        self.filling_ratio = 1.0
    
    @property
    def _effector(self):
        return 'bucket_bottom'
    
    @property
    def _target(self):
        return 'finish_point'

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.old_attach_pos = None
        self.old_filling_ratio = 1.0
        self.old_attach_acc = None
        self.attach_pos = None
        self.bottom_pos = None
        self.filling_ratio = 1.0
        self.beta = 0.0
        
        # int_angles = np.array([0.45,
        #                        np.random.uniform(0, 0.2),
        #                        np.random.uniform(-0.1, 0.35),
        #                        np.random.uniform(-0.1, 0.3)])
        
        data = physics.named.data
        data.qpos[_JOINTS] = self._init_angles
        physics.eng_reset()
        super().initialize_episode(physics)
               
    def get_reward(self, physics, action, control_timestep=None, step_count = 0, num_timesteps=None):
        distance_target = physics.site_distance(self._effector, self._target)
        site_xpos_ori = physics.named.data.site_xpos[['bucket', 'bucket_attach']]
        site_xpos_safe = physics.named.data.site_xpos[['bucket_attach', 'bucket_bottom']]
        mj_vel = np.round(physics.joint_qvel(_JOINTS).astype(np.float32), 8)
        total_flow_rate = physics.flow_rate(action)
        joint_qpos = np.round(physics.joint_qpos(_JOINTS).astype(np.float32), 8)
        
        reward_target = rewards.is_close(0.1, 1.0, distance_target, sigmoid='long_tail')
        reward_bucket_ori = rewards.bucket_orientation(site_xpos_ori, (-0.05, 0.1), 0.1, 0.05, flag='with_soil') - 1
        self.previous_action = action.copy()
        reward_constraint = rewards.joint_limit(0.05, 0.05, self._joint_upper_1, self._joint_lower_1, self._joint_upper_2, self._joint_lower_2, joint_qpos)
        reward_safe = rewards.is_collision(site_xpos_safe, (-1, -0.9), 0.15, 0.05, (-1, -0.9), 0.1, 0.05)
        reward_load = rewards.load_condition(self._vel_lower, self._vel_upper, self._cylinder_lower, self._cylinder_upper, 
                                             physics.sim_pos, physics.sim_vel, action)
        self.termination, reward_goal = rewards.p3_goal_state(distance_target, 0.1, mj_vel, 0.02, site_xpos_ori, (-0.05, 0.05))
        reward_flow = rewards.max_flow_rate(total_flow_rate, (0.0, 55.0), 2.5, 0.2)
        reward = 3.5 * reward_target + reward_bucket_ori + reward_flow + \
                reward_constraint / 3 + reward_safe / 2 + reward_load / 2 + reward_goal
        return reward


class DiggingSoil(BaseTask):
    def __init__(self, init_mj_angle, fully_observable, random):
        self._task_name = 'DiggingSoil'
        self._joint_lower_1 = np.array([-2.4, -0.511, -0.842, -0.894])
        self._joint_upper_1 = np.array([-2.0, -0.361, -0.692, -0.744])
        self._joint_lower_2 = np.array([2.0, 0.566, 1.56, 1.05])
        self._joint_upper_2 = np.array([2.4, 0.716, 1.71, 1.2])
        self._vel_lower = np.array([-1.428, -0.185, -0.214, -0.184])
        self._vel_upper = np.array([1.428, 0.143, 0.15, 0.129])
        self._cylinder_lower = np.array([-0.75 * np.pi, -0.15, -0.125, -0.15])
        self._cylinder_upper = np.array([0.75 * np.pi, 0.17, 0.33, 0.20])
        super().__init__(init_mj_angle, fully_observable, random)
    
    @property
    def _effector(self):
        return 'bucket_attach'
    
    @property
    def _target(self): 
        return 'dig_point'
        
    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.old_attach_pos = None
        self.old_filling_ratio = 0.0
        self.old_attach_acc = None
        self.attach_pos = None
        self.bottom_pos = None
        self.filling_ratio = 0.0
        self.beta = 0.0
        
        self.soil_phi = np.random.uniform(0.3, 0.8) # 内摩擦角 0.3 ~0.8 [rad]
        self.soil_rho = np.random.uniform(1400, 1800) # 密度 1400 ~ 1800 [kg/m^3]
        self.soil_C = np.random.uniform(0, 50) # 粘度系数 0 ~ 50 [kPa]
        self.soil_delta = np.random.uniform(0.2, 0.4) # 外摩擦角 0.2 ~ 0.4 [rad]
        self.soil_gamma = np.random.uniform(0.3, 0.8) # 滑裂面倾角 0.3 ~ 0.8 [rad]
        self.soil_Ca = np.random.uniform(0, 10) # 粘附度系数 0 ~ 10 [kPa]
 
        
        int_angles = np.array([0.45,
                               np.random.uniform(0, 0.2),
                               np.random.uniform(-0.1, 0.35),
                               np.random.uniform(-0.1, 0.3)])
        
        data = physics.named.data
        data.qpos[_JOINTS] = int_angles
        disp_bias, init_sim_pos = utils.compute_init_sim_pos(int_angles)
        physics.eng_random_reset(disp_bias, init_sim_pos)
        
    # _CONTROL_TIMESTEP = [0.08, 0.04, 0.02, 0.01, 0.005]
    def get_n_steps(self, physics):
        site_xpos = physics.named.data.site_xpos[[self._effector, 'bucket_bottom']]
        distance_ground = min(site_xpos[0][2], site_xpos[1][2])
        if self.old_filling_ratio <0.05:
            index = 1
        elif self.old_filling_ratio < 0.7:
            index = 3
        else: 
            index = 4
        if distance_ground < 0.3:
            index = max(index, 2)
        control_timestep = _CONTROL_TIMESTEP[index]
        com_timestep = control_timestep
        n_sub_steps = self.compute_n_steps(com_timestep, physics.timestep())
        n_sub_steps_control = self.compute_n_steps(control_timestep, com_timestep)
        # print("control_timestep: ", index)
        return control_timestep, com_timestep, n_sub_steps, n_sub_steps_control    
        
    def get_observation(self, physics):
        self.get_pre_observation(physics)
        obs = collections.OrderedDict()
        obs['joint_bounded_qpos'] = np.round(physics.bounded_joint_qpos(_JOINTS), 8).astype(np.float32)
        obs['joint_qvel'] = np.round(physics.joint_qvel(_JOINTS), 8).astype(np.float32)
        obs['bucket_xori'] = np.round(physics.body_3d_xpose('bucket'), 8).astype(np.float32)
        obs['filling_ratio'] = np.float32(np.round(self.filling_ratio, 8))
        obs['beta'] = np.float32(np.round(self.beta, 8))
        obs['ground'] = np.float32(self._ground)
        obs['pre_action'] = np.float32(self.previous_action)
        observation_arrays = [value.ravel() for value in obs.values()]
        observation = np.concatenate(observation_arrays).astype(np.float32)
        return observation
        
    def get_reward(self, physics, action, control_timestep=None, step_count = 0, num_timesteps=None):
        T1, T2, T3, T4 = False, False, False, False
        reward_T1, reward_T2, reward_T5, reward_T6, reward_T7, reward_T8 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        reward_R1, reward_R2, reward_R3, reward_R4, reward_R5 = 0.0, 0.0, 0.0, 0.0, 0.0
        k = 1 - np.exp(-0.01 * np.int8(num_timesteps / 1024))
        attach_xpos = physics.named.data.site_xpos['bucket_attach']    
        site_xpos_ori = physics.named.data.site_xpos[['bucket', 'bucket_attach']]
        joint_qpos = np.round(physics.joint_qpos(_JOINTS).astype(np.float32), 8)
        attach_xvel = physics.get_sensor_by_name('attach_linvel')
        mj_vel = np.round(physics.joint_qvel(_JOINTS).astype(np.float32), 8)
        """termination condition"""
        
        # T1: Full enough
        if self.filling_ratio > k * 0.68 +0.3:
            T1 = True

        # T2: Too close to the base
        if attach_xpos[0] < 2.0:
            T2 = True

        # T3: Edge high enough
        if attach_xpos[2] > 0.7 * k + 0.3:
            T3 = True
            
        # T4: Curled enough
        bucket_ori = site_xpos_ori[0][2] - site_xpos_ori[1][2]
        if 0 < bucket_ori < 0.12:
            T4 = True
            
        if (T1 or T2) and T3 and T4:
            self.termination_flag += 1
            if self.termination_flag > 10 + 100 * k:
                self.termination = True
                if T1 == True:
                    reward_T1 = 10.0 * np.exp((1 - step_count / _STEP_LIMIT) * 3)
                elif T2 == True:
                    reward_T2 = 5.0              
                
        """truncation condition"""
        
        # T5: push soil with plate
        if (self.attach_pos[1] < 0.0 or self.bottom_pos[1] < 0.0) and self.beta < 0.0:
            reward_T5 = -0.3
        
        # T6: empty bucket above soil
        if self.filling_ratio < 0.01 and self.attach_pos[1] > 0.0:
            reward_T6 = rewards.explore_boundary(self.attach_pos[1], (0.0, 1.0), 0.1, 0.05)

        # T7: Too deep or flat 
        if self.attach_pos[1] < 0.0:
            reward_T7 = rewards.explore_boundary(self.attach_pos[1], (-0.5, 0.0), 0.1, 0.05)
            
        """normal reward"""

        # R1: move down
        if self.filling_ratio < 0.05:
            reward_R1 = - 2.0 * attach_xvel[2]
            
        # R2: filling
        if not T2:
            reward_R2 = 200 * (self.filling_ratio - self.old_filling_ratio)  
        self.old_filling_ratio = self.filling_ratio

        # R3: move up
        if (T1 or T2) and self.attach_pos[1] <= 1.0:
            reward_R3 = 2.0 * attach_xvel[2]
            
        # R4: curling
        if (T1 or T2) and (not T4):
            if bucket_ori > 0.12:
                reward_R4 = 4 * mj_vel[3]
            else:
                reward_R4 = -4 * mj_vel[3]

        reward_constraint = rewards.joint_limit(0.05, 0.05, self._joint_upper_1, self._joint_lower_1, self._joint_upper_2, self._joint_lower_2, 
                                               np.round(physics.joint_qpos(_JOINTS).astype(np.float32), 8))
        reward_load = rewards.load_condition(self._vel_lower, self._vel_upper, self._cylinder_lower, self._cylinder_upper, 
                                              physics.sim_pos, physics.sim_vel, action)
        total_flow_rate = physics.flow_rate(action)
        reward_flow = rewards.max_flow_rate(total_flow_rate, (0.0, 60.0), 2.5, 0.2)
        reward = reward_T1 + reward_T2 + reward_T5 + \
                 (reward_T6 + reward_T7) / 3 +\
                 reward_R1 + reward_R2 + reward_R3 + 2 * reward_R4 + reward_R5 + \
                 reward_constraint / 3 + reward_load / 3 + reward_flow / 2
        return reward
        

class Hauling(BaseTask):
    def __init__(self, init_mj_angle, fully_observable, random):
        self._joint_lower_1 = np.array([-2.4, -0.511, -0.842, -0.894])
        self._joint_upper_1 = np.array([-2, -0.361, -0.692, -0.744])
        self._joint_lower_2 = np.array([2, 0.566, 1.56, 1.15])
        self._joint_upper_2 = np.array([2.4, 0.716, 1.71, 1.2])
        self._vel_lower = np.array([-1.428, -0.185, -0.214, -0.184])
        self._vel_upper = np.array([1.428, 0.143, 0.15, 0.129])
        self._cylinder_lower = np.array([-0.75 * np.pi, -0.15, -0.125, -0.15])
        self._cylinder_upper = np.array([0.75 * np.pi, 0.17, 0.33, 0.20])
        super().__init__(init_mj_angle, fully_observable, random)
        self.filling_ratio = 1.0

    @property
    def _effector(self):
        return 'bucket_bottom'
    
    @property
    def _target(self):
        return 'unload_point'

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.old_attach_pos = None
        self.old_filling_ratio = 1.0
        self.old_attach_acc = None
        self.attach_pos = None
        self.bottom_pos = None
        self.filling_ratio = 1.0
        self.beta = 0.0
        
        # int_angles = np.array([0.45,
        #                        np.random.uniform(0, 0.2),
        #                        np.random.uniform(-0.1, 0.35),
        #                        np.random.uniform(-0.1, 0.3)])
        
        data = physics.named.data
        data.qpos[_JOINTS] = self._init_angles
        physics.eng_reset()
        super().initialize_episode(physics)
                
    def get_reward(self, physics, action, control_timestep=None, step_count = 0, num_timesteps=None):
        distance_finish = physics.site_distance(self._effector, self._target)
        site_xpos_ori = physics.named.data.site_xpos[['bucket', 'bucket_attach']]
        site_xpos_safe = physics.named.data.site_xpos[['bucket_attach', 'bucket_bottom']]
        mj_vel = np.round(physics.joint_qvel(_JOINTS).astype(np.float32), 8)
        total_flow_rate = physics.flow_rate(action)
        
        reward_target = rewards.is_close(0.1, 1.5, distance_finish, sigmoid='long_tail')
        reward_bucket_ori = rewards.bucket_orientation(site_xpos_ori, (-0.05, 0.05), 0.05, 0.05, flag='with_soil') - 1
        self.previous_action = action.copy()
        
        reward_constraint = rewards.joint_limit(0.05, 0.05, self._joint_upper_1, self._joint_lower_1, self._joint_upper_2, self._joint_lower_2, 
                                                np.round(physics.joint_qpos(_JOINTS).astype(np.float32), 8))
        reward_safe = rewards.is_collision(site_xpos_safe, (-1, 0), 0.45, 0.05, (-1, 0), 0.35, 0.05)
        reward_load = rewards.load_condition(self._vel_lower, self._vel_upper, self._cylinder_lower, self._cylinder_upper, 
                                             physics.sim_pos, physics.sim_vel, action)
        self.termination, reward_goal = rewards.p3_goal_state(distance_finish, 0.1, mj_vel, 0.02, site_xpos_ori, (-0.05, 0.05))
        reward_flow = rewards.max_flow_rate(total_flow_rate, (0.0, 55.0), 2.5, 0.2)
        reward = 3 * reward_target + reward_bucket_ori + reward_flow + \
                 + reward_constraint / 3 + reward_safe / 2 + reward_load / 2 + reward_goal
        return reward