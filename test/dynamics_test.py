from dm_control.rl import control
import dm_env
import matlab.engine
import os
import mujoco
import cv2
import numpy as np
import time
import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
_CONTROL_TIMESTEP = .04  # (Seconds)
_COM_TIMESTEP = .02  # (Seconds)
_SIM_TIMESTEP = .005  # (Seconds)
_TIME_LIMIT = 20  # (Seconds)
_JOINTS = ['cabin', 'boom_cylinder', 'boom_rod', 'boom', 'arm_cylinder',
           'arm_rod', 'arm', 'bucket_cylinder', 'bucket_rod', 'blink_2',
           'blink_1', 'bucket']
_ACUTATOR_POS = ['swing_pos', 'boom_pos', 'arm_pos', 'bucket_pos']
_ACUTATOR_VEL = ['swing_vel', 'boom_vel', 'arm_vel', 'bucket_vel']
_HING_JOINTS = ['cabin', 'boom', 'arm', 'blink_2', 'blink_1', 'bucket']
_SLIDE_JOINTS = ['boom_cylinder', 'boom_rod', 'arm_cylinder',
                 'arm_rod', 'bucket_cylinder', 'bucket_rod']
_ALL_PROPS = frozenset(['ball', 'target_ball', 'cup',
                        'peg', 'target_peg', 'slot'])
_SENSORS_TORQUE = ['torque_boom', 'torque_arm', 'torque_bucket']
_DOF_ADDRESS = [0, 1, 2, 3]
# [O1a, O1b, O2c, O2d, gh, gf, O3f, O3g, O3e, eh]
_COMPONENT_LENGTH = [0.297059, 1.181326, 0.238248, 1.396755, 0.234974, 0.970948, 1.1254, 0.157623, 0.187843, 0.225615] 
_INIT_CYLINDER_LENGTH = [1.236, 1.288, 0.9] #[boom, arm, bucket]
_INIT_ALPHA = [1.633416166871895, 1.019263853533036, 1.652802643862622]
_CONSTANT_THETA = [0.03, 2.925065]
_CYLINDER_ROD_PAIRS = [('boom_cylinder', 'boom_rod'),
                    ('arm_cylinder', 'arm_rod'),
                    ('blink1', 'blink2')]
_LENGTH_INDICES = [(1, 0), (2, 3), (5, 4)]

class SimulinkPlant:
    """" This class is used to connect to the Simulink Model and simulate the hydraulic system"""

    def __init__(self, model_name='hydraulic_model_sensors'):
        # The name of the Simulink Model (To be placed in the same directory as the Python Code)
        self.model_name = model_name

    def connectToMatlab(self):
        print("Starting matlab")
        """ TEST: use GUI matlab to monitor the simulation """
        process_id = matlab.engine.find_matlab()
        self.eng = matlab.engine.connect_matlab(process_id[0])

        """ real use """
        # os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        # self.eng = matlab.engine.start_matlab("-softwareopengl")
        print("Connected to Matlab")
        # Load the model
        work_dir = os.path.join(os.path.dirname(__file__), "../matlab_model")
        self.eng.cd(work_dir, nargout=0)
        try:
            self.eng.init_model(nargout=0)
        except Exception as e:
            print("Error: ", e)
            print("Closing MATLAB process due to error...")
            self.eng.quit()
        print("start model and set as fastrestart mode")
    
    def simulate(self, eng_time, u, loads):
        try:
            return self.eng.test_sim_step(eng_time, matlab.single(u), matlab.single(loads), nargout=4)
        except Exception as e:
            print("Error: ", e)
            print("Closing MATLAB process due to error...")
            self.eng.quit()
    def disconnect(self):
        self.eng.set_param(self.model_name,'SimulationCommand','stop',nargout=0)
        self.eng.quit()
        
    def reset(self):
        self.eng.reset_model(nargout=0)   
        
def model_path():
    model_path = os.path.join(os.path.dirname(__file__), "../assets/excavator.xml")
    return model_path

def get_joint_id(model, joint_name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

def get_actuator_id(model, actuator_name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)

# sensor name is a 1*4 np array, output is 1*4 np array
def get_sensor_by_name(model, data, sensor_names):
    sensor_datas = []
    for sensor_name in sensor_names:
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        sensor_adr = model.sensor_adr[sensor_id]
        sensor_dim = model.sensor_dim[sensor_id]
        sensor_datas.append(data.sensordata[sensor_adr:sensor_adr + sensor_dim][0])
    return sensor_datas

def step_signal(t, a, period=1):
    assert np.abs(a) <= 1, "Amplitude must be less than 1.0"
    return a if (t % period) < (period / 2) else -a

def site_distance(model, data, site1, site2):
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) for site in [site1, site2]]
    site1_to_site2 = np.diff(data.site_xpos[site_ids], axis=0)
    return np.linalg.norm(site1_to_site2)

def get_sensor_by_name(model, data, sensor_name):
    sensor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    sensor_data = data.sensordata[sensor_adr:sensor_adr + sensor_dim]
    return sensor_data

"""
1. test torque and force transfer
"""
def test_t2f_second():
    sim = SimulinkPlant()
    sim.connectToMatlab()
    try:
        model = mujoco.MjModel.from_xml_path(model_path())
    except Exception as e:
        print(f"Error: {e}")

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)
    duration = 5
    framrate = 30
    num_steps = int(duration / model.opt.timestep)
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    frames = []
    
    mujoco.mj_resetData(model, data)
    mujoco.mj_step1(model, data)
    
    mj_distance_init = np.array([site_distance(model,data,pair[0],pair[1]) for pair in _CYLINDER_ROD_PAIRS])
    mj_distance_init = np.insert(mj_distance_init, 0, 0)
    mj_distance_last = mj_distance_init
    time_data = []
    sim_disp_data = []
    sim_vel_data = []
    mj_disp_data = []
    mj_vel_data = []
    error_disp_data = []
    error_vel_data = []

    # with open ("./test/t2f_test.csv", "a") as f:
    #         f.write("mujoco_time, action, transfered_force, sim_time, sim_force, sim_disp, mj_disp, error_disp, sim_vel, mj_vel, error_vel \n")  
    with open ("./test/t2f_test.csv", "a") as f:
        f.write("'mj_time 'qfrc_constraint', 'qfrc_bias', 'sim_time', 'force2sim', 'sim_force','torque2mj'\n")
    while data.time < duration:
        # control_signal = np.zeros(4, dtype=np.float32)
        control_signal = np.array(np.round(np.random.uniform(-1, 1, size=(1,4)),2), dtype=np.float32)
        for i in range(1):
            torque_load = np.zeros(4,dtype=np.float32)
            eng_time = (round(data.time / _COM_TIMESTEP) +1) * _COM_TIMESTEP
            qfrc_constraint = data.qfrc_constraint[_DOF_ADDRESS]
            qfrc_bias = data.qfrc_bias[_DOF_ADDRESS]
            torque_load[0] = qfrc_constraint[0] - qfrc_bias[0]
            torque_load[1] = qfrc_constraint[1] - qfrc_bias[1]
            torque_load[2] = qfrc_constraint[2] - qfrc_bias[2]
            torque_load[3] = qfrc_constraint[3] - qfrc_bias[3]
            # torque_load[3] = -data.qfrc_constraint[_DOF_ADDRESS[3]] + data.qfrc_bias[_DOF_ADDRESS[3]]
        
            _LENGTH_INDICES = [(1, 0), (2, 3), (4, 5)]
            lengths = [site_distance(model, data, pair[0], pair[1]) for pair in _CYLINDER_ROD_PAIRS]
            thetas = []
            force2sim = [torque_load[0]]
            for i, (length, (idx1, idx2)) in enumerate(zip(lengths, _LENGTH_INDICES)):
                theta = np.arccos((length**2 + _COMPONENT_LENGTH[idx1]**2 - _COMPONENT_LENGTH[idx2]**2) / 
                                (2 * length * _COMPONENT_LENGTH[idx1]))
                load = torque_load[i+1] / (_COMPONENT_LENGTH[idx1] * np.sin(theta))
                thetas.append(theta)
                force2sim.append(load)
            force2sim = np.array(force2sim, dtype=np.float32)
            sim_force, sim_disp, sim_vel, sim_time = sim.simulate(eng_time, control_signal, force2sim)
        
            sim_force = np.array(sim_force, dtype=np.float32)
            sim_disp = np.array(sim_disp, dtype=np.float32)
            sim_vel = np.array(sim_vel, dtype=np.float32)
            
            torque2mj = []
            torque2mj.append(sim_force[0][0])
            for i, (theta, (idx1, idx2)) in enumerate(zip(thetas, _LENGTH_INDICES)):
                torque = -sim_force[0][i+1] * np.sin(theta) * _COMPONENT_LENGTH[idx1]
                torque2mj.append(torque)
            for i, idx in enumerate(_DOF_ADDRESS):
                data.qfrc_applied[idx] = torque2mj[i]
            
            for i in range(4):       
                # with open ("./test/t2f_test.csv", "a") as f:
                #     f.write(f"{data.time}, {control_signal}, {force2sim}, {sim_time},{sim_force}, ")
                with open ("./test/t2f_test.csv", "a") as f:
                    f.write(f"{data.time}\n, {qfrc_constraint}\n, {qfrc_bias}\n, {sim_time}\n, {force2sim}\n, {sim_force}\n,{torque2mj}\n\n")
                    
                mj_distance_tmp = np.array([site_distance(model, data, pair[0], pair[1]) for pair in _CYLINDER_ROD_PAIRS])
                mj_angle_cabin = get_sensor_by_name(model, data, 'jointpos_swing')
                mj_distance = np.insert(mj_distance_tmp, 0, mj_angle_cabin)
                mj_disp =  mj_distance - mj_distance_init
                error_disp = mj_disp - sim_disp
                
                mj_vel = (mj_distance - mj_distance_last) / model.opt.timestep
                # mj_vel_cabin = get_sensor_by_name(model, data, 'jointvel_swing')
                # mj_vel = np.insert(mj_vel_tmp, 0, mj_vel_cabin)
                error_vel = mj_vel - sim_vel
                
                # with open ("./test/t2f_test.csv", "a") as f:
                #     f.write(f"{sim_disp}, {mj_disp}, {error_disp}, {sim_vel}, {mj_vel}, {error_vel}\n")
                    
                mj_distance_last = mj_distance
            
                time_data.append(data.time)
                sim_disp_data.append(sim_disp)
                sim_vel_data.append(sim_vel)
                mj_disp_data.append(mj_disp)
                mj_vel_data.append(mj_vel)
                error_disp_data.append(error_disp)
                error_vel_data.append(error_vel)
                
                mujoco.mj_step2(model, data)
                mujoco.mj_step1(model, data) 
        if len(frames) < data.time * framrate:
            renderer.update_scene(data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)
    sim.disconnect()

    time_data = np.array(time_data)
    n = time_data.shape[0]
    sim_disp_data = np.array(sim_disp_data).reshape(n,4)
    sim_vel_data = np.array(sim_vel_data).reshape(n,4)
    mj_disp_data = np.array(mj_disp_data).reshape(n,4)
    mj_vel_data = np.array(mj_vel_data).reshape(n,4)
    error_disp_data = np.array(error_disp_data).reshape(n,4)
    error_vel_data = np.array(error_vel_data).reshape(n,4)
    
    # height, width, layers = frames[0].shape
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    # out = cv2.VideoWriter('output.mp4', fourcc, framrate, (width, height))
    # if not out.isOpened():
    #     print("Error: Could not open video writer")
    #     exit()
    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("video", frame)
        if cv2.waitKey(int(1000 // framrate)) & 0xFF == ord('q'):
            break    
    cv2.waitKey(0)
    # out.release()
    cv2.destroyAllWindows()
    
    
    # Swing
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_data, sim_disp_data[:,0], label='sim_disp', linestyle='-', color='blue')
    plt.plot(time_data, mj_disp_data[:,0], label='mj_disp', linestyle='-', color='green')
    plt.plot(time_data, error_disp_data[:,0], label='error_disp', linestyle='--', color='orange')
    plt.legend()
    plt.title('Swing: Angel in Simulink, Angel in Mujoco and Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_data, sim_vel_data[:,0], label='sim_vel', linestyle='-', color='blue')
    plt.plot(time_data, mj_vel_data[:,0], label='mj_vel', linestyle='-', color='green')
    plt.plot(time_data, error_vel_data[:,0], label='error_vel', linestyle='--', color='orange')
    plt.legend()
    plt.title('Swing: Angular Velocity in Simulink, Angular Velocity in Mujoco and Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'second_swing_random_50Hz.png'))
    
    # Boom
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_data, sim_disp_data[:,1], label='sim_disp', linestyle='-', color='blue')
    plt.plot(time_data, mj_disp_data[:,1], label='mj_disp', linestyle='-', color='green')
    plt.plot(time_data, error_disp_data[:,1], label='error_disp', linestyle='--', color='orange')
    plt.legend()
    plt.title('Boom: Displacement in Simulink, Displacement in Mujoco and Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_data, sim_vel_data[:,1], label='sim_vel', linestyle='-', color='blue')
    plt.plot(time_data, mj_vel_data[:,1], label='mj_vel', linestyle='-', color='green')
    plt.plot(time_data, error_vel_data[:,1], label='error_vel', linestyle='--', color='orange')
    plt.legend()
    plt.title('Boom: Velocity in Simulink, Velocity in Mujoco and Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'second_boom_random_50Hz.png'))
    
    # Arm
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_data, sim_disp_data[:,2], label='sim_disp', linestyle='-', color='blue')
    plt.plot(time_data, mj_disp_data[:,2], label='mj_disp', linestyle='-', color='green')
    plt.plot(time_data, error_disp_data[:,2], label='error_disp', linestyle='--', color='orange')
    plt.legend()
    plt.title('Arm: Displacement in Simulink, Displacement in Mujoco and Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_data, sim_vel_data[:,2], label='sim_vel', linestyle='-', color='blue')
    plt.plot(time_data, mj_vel_data[:,2], label='mj_vel', linestyle='-', color='green')
    plt.plot(time_data, error_vel_data[:,2], label='error_vel', linestyle='--', color='orange')
    plt.legend()
    plt.title('Arm: Velocity in Simulink, Velocity in Mujoco and Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'second_arm_random_50Hz.png'))      
    
    # Bucket
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_data, sim_disp_data[:,3], label='sim_disp', linestyle='-', color='blue')
    plt.plot(time_data, mj_disp_data[:,3], label='mj_disp', linestyle='-', color='green')
    plt.plot(time_data, error_disp_data[:,3], label='error_disp', linestyle='--', color='orange')
    plt.legend()
    plt.title('Bucket: Displacement in Simulink, Displacement in Mujoco and Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_data, sim_vel_data[:,3], label='sim_vel', linestyle='-', color='blue')
    plt.plot(time_data, mj_vel_data[:,3], label='mj_vel', linestyle='-', color='green')
    plt.plot(time_data, error_vel_data[:,3], label='error_vel', linestyle='--', color='orange')
    plt.legend()
    plt.title('Bucket: Velocity in Simulink, Velocity in Mujoco and Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'second_bucket_random_50Hz.png'))



    """
    1.test the change of qfrc_constraint, qfrc_bias, qfrc_applied when combined with simulink
    """
def test_t2f_first():
    sim = SimulinkPlant()
    sim.connectToMatlab()
    try:
        model = mujoco.MjModel.from_xml_path(model_path())
    except Exception as e:
        print(f"Error: {e}")

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)
    duration = 1.9
    framrate = 30
    num_steps = int(duration / model.opt.timestep)
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    frames = []
    
    mujoco.mj_resetData(model, data)
    mujoco.mj_step1(model, data)
    

    time_data = []
    tq_constraint_data = []
    tq_bias_data = []
    tq_applied_data = []
    tq_load_data = []


    # with open ("./test/t2f_test.csv", "a") as f:
    #         f.write("mujoco_time, action, transfered_force, sim_time, sim_force, sim_disp, mj_disp, error_disp, sim_vel, mj_vel, error_vel \n")  
    with open ("./test/t2f_test.csv", "a") as f:
        f.write("'mj_time 'qfrc_constraint', 'qfrc_bias', 'sim_time', 'force2sim', 'sim_force','torque2mj'\n")

    while data.time < duration:
        control_signal = np.zeros(4, dtype=np.float32)
        for i in range(1):
            torque_load = np.zeros(4,dtype=np.float32)
            eng_time = (round(data.time / _COM_TIMESTEP) +1) * _COM_TIMESTEP
            qfrc_constraint = data.qfrc_constraint[_DOF_ADDRESS]
            qfrc_bias = data.qfrc_bias[_DOF_ADDRESS]
            torque_load[0] = qfrc_constraint[0] - qfrc_bias[0]
            torque_load[1] = qfrc_constraint[1] - qfrc_bias[1]
            torque_load[2] = qfrc_constraint[2] - qfrc_bias[2]
            torque_load[3] = qfrc_constraint[3] - qfrc_bias[3]
        
            _LENGTH_INDICES = [(1, 0), (2, 3), (4, 5)]
            lengths = [site_distance(model, data, pair[0], pair[1]) for pair in _CYLINDER_ROD_PAIRS]
            thetas = []
            force2sim = [torque_load[0]]
            for i, (length, (idx1, idx2)) in enumerate(zip(lengths, _LENGTH_INDICES)):
                theta = np.arccos((length**2 + _COMPONENT_LENGTH[idx1]**2 - _COMPONENT_LENGTH[idx2]**2) / 
                                (2 * length * _COMPONENT_LENGTH[idx1]))
                load = torque_load[i+1] / (_COMPONENT_LENGTH[idx1] * np.sin(theta))
                thetas.append(theta)
                force2sim.append(load)
            force2sim = np.array(force2sim, dtype=np.float32)
            sim_force, sim_disp, sim_vel, sim_time = sim.simulate(eng_time, control_signal, force2sim)
        
            sim_force = np.array(sim_force, dtype=np.float32)
            sim_disp = np.array(sim_disp, dtype=np.float32)
            sim_vel = np.array(sim_vel, dtype=np.float32)
            
            torque2mj = []
            torque2mj.append(sim_force[0][0])
            for i, (theta, (idx1, idx2)) in enumerate(zip(thetas, _LENGTH_INDICES)):
                torque = -sim_force[0][i+1] * np.sin(theta) * _COMPONENT_LENGTH[idx1]
                torque2mj.append(torque)
            for i, idx in enumerate(_DOF_ADDRESS):
                data.qfrc_applied[idx] = torque2mj[i]
            
            for i in range(4):        
                # with open ("./test/t2f_test.csv", "a") as f:
                #     f.write(f"{data.time}, {control_signal}, {force2sim}, {sim_time},{sim_force}, ")
                with open ("./test/t2f_test.csv", "a") as f:
                    f.write(f"{data.time}\n, {qfrc_constraint}\n, {qfrc_bias}\n, {sim_time}\n, {force2sim}\n, {sim_force}\n,{torque2mj}\n\n")
                    
                time_data.append(data.time)
                tq_constraint_data.append(qfrc_constraint)
                tq_bias_data.append(qfrc_bias)
                tq_load_data.append(torque_load)
                tq_applied_data.append(torque2mj)
                
                mujoco.mj_step2(model, data)
                mujoco.mj_step1(model, data) 
                if len(frames) < data.time * framrate:
                    renderer.update_scene(data, scene_option=scene_option)
                    pixels = renderer.render()
                    frames.append(pixels)
    sim.disconnect()

    time_data = np.array(time_data)
    n = time_data.shape[0]
    tq_constraint_data = np.array(tq_constraint_data).reshape(n,4)
    tq_bias_data = np.array(tq_bias_data).reshape(n,4)
    tq_load_data = -1*np.array(tq_load_data).reshape(n,4)
    tq_applied_data = np.array(tq_applied_data).reshape(n,4)

    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("video", frame)
        if cv2.waitKey(int(1000 // framrate)) & 0xFF == ord('q'):
            break    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
   # Swing
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, tq_constraint_data[:,0], label='tq_cons', linestyle='-', color='blue')
    plt.plot(time_data, tq_bias_data[:,0], label='tq_bias', linestyle='-', color='green')
    plt.plot(time_data, tq_applied_data[:,0], label='tq_app', linestyle='--', color='orange')
    plt.plot(time_data, tq_load_data[:,0], label='minus_tq_load', linestyle='--', color='red')
    plt.legend()
    plt.title('Swing')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), './qfrc_result/first_swing.png'))
    
    # Boom
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, tq_constraint_data[:,1], label='tq_cons', linestyle='-', color='blue')
    plt.plot(time_data, tq_bias_data[:,1], label='tq_bias', linestyle='-', color='green')
    plt.plot(time_data, tq_applied_data[:,1], label='tq_app', linestyle='--', color='orange')
    plt.plot(time_data, tq_load_data[:,1], label='minus_tq_load', linestyle='--', color='red')
    plt.legend()
    plt.title('Boom')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), './qfrc_result/first_boom.png'))
    
    # Arm
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, tq_constraint_data[:,2], label='tq_cons', linestyle='-', color='blue')
    plt.plot(time_data, tq_bias_data[:,2], label='tq_bias', linestyle='-', color='green')
    plt.plot(time_data, tq_applied_data[:,2], label='tq_app', linestyle='--', color='orange')
    plt.plot(time_data, tq_load_data[:,2], label='minus_tq_load', linestyle='--', color='red')
    plt.legend()
    plt.title('Arm')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), './qfrc_result/first_arm.png'))  
    
    # Bucket
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, tq_constraint_data[:,3], label='tq_cons', linestyle='-', color='blue')
    plt.plot(time_data, tq_bias_data[:,3], label='tq_bias', linestyle='-', color='green')
    plt.plot(time_data, tq_applied_data[:,3], label='tq_app', linestyle='--', color='orange')
    plt.plot(time_data, tq_load_data[:,3], label='minus_tq_load', linestyle='--', color='red')
    plt.legend()
    plt.title('Bucket')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), './qfrc_result/first_bucket.png'))
    
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

def cos_theorem(a, b, c, flag):
    if flag == 'theta':
        return np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
    elif flag == 'length':
        return np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(c))
    
def sin_theorem(a, b, beta):
    return np.arcsin(a * np.sin(beta) / b)

def torque2force(torque_loads, lengths):
    thetas = []
    force2sim = [torque_loads[0]]
    # compute theta of boom and arm
    for i, (length, (idx1, idx2)) in enumerate(zip(lengths, _LENGTH_INDICES)):
        theta = cos_theorem(length, _COMPONENT_LENGTH[idx1], _COMPONENT_LENGTH[idx2], 'theta')
        thetas.append(theta)
    # compute theta of bucket    
    theta_4 = thetas[2] - _CONSTANT_THETA[0]
    O3h = cos_theorem(lengths[2], _COMPONENT_LENGTH[6], theta_4, 'length')
    sin_theta_3 = _COMPONENT_LENGTH[6] * np.sin(theta_4) / O3h
    sin_thetas = np.sin(thetas)
    sin_thetas[-1] = sin_theta_3
    # compute torque2force
    moment_arms = np.array([_COMPONENT_LENGTH[1], _COMPONENT_LENGTH[2], O3h])
    for i, (sin_theta, moment_arm) in enumerate(zip(sin_thetas, moment_arms)):
        load = torque_loads[i+1] / (moment_arm * sin_theta)
        force2sim.append(load)
    return force2sim

def pos_vel_2_alpha_omega(rod_pos, rod_vel):
    alphas = [rod_pos[0]]
    # compute alpha of boom and arm
    lengths = []
    for i, length in enumerate(_INIT_CYLINDER_LENGTH):
        lengths.append(length + rod_pos[i+1])

    for i, (length, (idx1, idx2)) in enumerate(zip(lengths, _LENGTH_INDICES)):
        alpha = cos_theorem(_COMPONENT_LENGTH[idx1], _COMPONENT_LENGTH[idx2], length, 'theta')
        alphas.append(alpha)
        
    O3GH = _CONSTANT_THETA[1] - alphas[3]
    O3h = cos_theorem(_COMPONENT_LENGTH[4], _COMPONENT_LENGTH[7], O3GH, 'length')
    alpha = sin_theorem(_COMPONENT_LENGTH[4], O3h, O3GH) + cos_theorem(_COMPONENT_LENGTH[8], O3h, _COMPONENT_LENGTH[9], 'theta')
    alphas[-1] = alpha
    omegas = vel2omega(rod_pos, rod_vel)
    return alphas, omegas, lengths
"""
1. torque -> force -> simulink -> pos -> angle -> mujoco
"""
def test_t2f_third():
    sim = SimulinkPlant()
    sim.connectToMatlab()
    try:
        model = mujoco.MjModel.from_xml_path(model_path())
    except Exception as e:
        print(f"Error: {e}")

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)
    duration = 10
    framrate = 30
    num_steps = int(duration / model.opt.timestep)
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    frames = []
    
    mujoco.mj_resetData(model, data)
    mujoco.mj_step1(model, data)
    
    mj_distance_init = _INIT_CYLINDER_LENGTH
    mj_distance_init = np.insert(mj_distance_init, 0, 0)
    mj_distance_last = mj_distance_init
    time_data = []
    sim_disp_data = []
    sim_vel_data = []
    mj_disp_data = []
    mj_vel_data = []
    error_disp_data = []
    error_vel_data = []
    
    tq_constraint_data = []
    tq_bias_data = []
    tq_applied_data = []
    tq_load_data = []
    tq_sim_data = []
    
    O3g	= 0.157623
    gf	= 0.970948
    gh	= 0.234974
    eh	= 0.225615
    O3e	= 0.187843
    lengths = _INIT_CYLINDER_LENGTH

    # with open ("./test/t2f_test.csv", "a") as f:
    #         f.write("mujoco_time, action, transfered_force, sim_time, sim_force, sim_disp, mj_disp, error_disp, sim_vel, mj_vel, error_vel \n")  
    with open ("./test/t2f_test.csv", "a") as f:
        f.write("'mj_time 'qfrc_constraint', 'qfrc_bias', 'sim_time', 'force2sim', 'sim_force','torque2mj'\n")
    while data.time < duration:
        # control_signal = np.zeros(3, dtype=np.float32)
        control_signal = np.array(np.round(np.random.uniform(-1, 1, size=(1,4)),2), dtype=np.float32)
        # control_signal = np.insert(control_signal, 1, 1)
        for i in range(1):
            eng_time = (round(data.time / _COM_TIMESTEP) +1) * _COM_TIMESTEP
            
            torque_loads = np.zeros(4,dtype=np.float32)
            qfrc_constraint = data.qfrc_constraint[_DOF_ADDRESS]
            qfrc_bias = data.qfrc_bias[_DOF_ADDRESS]
            torque_loads[0] = qfrc_constraint[0] - qfrc_bias[0]
            torque_loads[1] = qfrc_constraint[1] - qfrc_bias[1]
            torque_loads[2] = qfrc_constraint[2] - qfrc_bias[2]
            torque_loads[3] = qfrc_constraint[3] - qfrc_bias[3]
            with open ("./test/t2f_third.csv", "a") as f:
                f.write(f"{qfrc_bias}, {qfrc_constraint}\n")
            force2sim = np.array(torque2force(torque_loads, lengths), dtype=np.float32)
            sim_force, sim_pos, sim_vel, sim_time = sim.simulate(eng_time, control_signal, force2sim)
        
            sim_pos = np.array(sim_pos, dtype=np.float32).flatten()
            sim_vel = np.array(sim_vel, dtype=np.float32).flatten()
            
            alphas, omegas, lengths = pos_vel_2_alpha_omega(sim_pos, sim_vel)

            data.ctrl[get_actuator_id(model, _ACUTATOR_POS[0])] = alphas[0] * 1000
            data.ctrl[get_actuator_id(model, _ACUTATOR_VEL[0])] = omegas[0] * 1000
            data.ctrl[get_actuator_id(model, _ACUTATOR_POS[1])] = (_INIT_ALPHA[0] - alphas[1]) * 1000
            data.ctrl[get_actuator_id(model, _ACUTATOR_VEL[1])] = omegas[1] * 1000
            data.ctrl[get_actuator_id(model, _ACUTATOR_POS[2])] = (alphas[2] - _INIT_ALPHA[1]) * 1000
            data.ctrl[get_actuator_id(model, _ACUTATOR_VEL[2])] = omegas[2] * 1000
            data.ctrl[get_actuator_id(model, _ACUTATOR_POS[3])] = (alphas[3] - _INIT_ALPHA[2]) * 1000
            data.ctrl[get_actuator_id(model, _ACUTATOR_VEL[3])] = omegas[3] * 1000
            
            for i in range(4):      
                mj_distance_tmp = np.array([site_distance(model, data, pair[0], pair[1]) for pair in _CYLINDER_ROD_PAIRS])
                # set a site in g, the last paris of cylinder and rod is ('e', 'g')
                O3GE = cos_theorem(O3g, mj_distance_tmp[2], O3e, 'theta')
                EGH = cos_theorem(mj_distance_tmp[2], gh, eh, 'theta')
                HGF = _CONSTANT_THETA[1] - O3GE - EGH
                mj_distance_tmp[-1] = cos_theorem(gh, gf, HGF, 'length')
                mj_angle_cabin = get_sensor_by_name(model, data, 'jointpos_swing')
                mj_distance = np.insert(mj_distance_tmp, 0, mj_angle_cabin)
                mj_disp =  mj_distance - mj_distance_init
                error_disp = mj_disp - sim_pos
                mj_vel = (mj_distance - mj_distance_last) / model.opt.timestep
                error_vel = mj_vel - sim_vel
                
                mj_distance_last = mj_distance
            
                time_data.append(data.time)
                sim_disp_data.append(sim_pos)
                sim_vel_data.append(sim_vel)
                mj_disp_data.append(mj_disp)
                mj_vel_data.append(mj_vel)
                error_disp_data.append(error_disp)
                error_vel_data.append(error_vel)
                tq_constraint_data.append(qfrc_constraint)
                tq_bias_data.append(qfrc_bias)
                tq_load_data.append(force2sim)
                tq_sim_data.append(sim_force)
                
                mujoco.mj_step2(model, data)
                mujoco.mj_step1(model, data) 

        if len(frames) < data.time * framrate:
            renderer.update_scene(data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)
    sim.disconnect()

    time_data = np.array(time_data)
    n = time_data.shape[0]
    sim_disp_data = np.array(sim_disp_data).reshape(n,4)
    sim_vel_data = np.array(sim_vel_data).reshape(n,4)
    mj_disp_data = np.array(mj_disp_data).reshape(n,4)
    mj_vel_data = np.array(mj_vel_data).reshape(n,4)
    error_disp_data = np.array(error_disp_data).reshape(n,4)
    error_vel_data = np.array(error_vel_data).reshape(n,4)
    tq_constraint_data = np.array(tq_constraint_data).reshape(n,4)
    tq_bias_data = np.array(tq_bias_data).reshape(n,4)
    tq_load_data = np.array(tq_load_data).reshape(n,4)
    tq_sim_data = np.array(tq_sim_data).reshape(n,4)

    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("video", frame)
        if cv2.waitKey(int(1000 // framrate)) & 0xFF == ord('q'):
            break    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Swing
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, tq_constraint_data[:,0], label='tq_cons', linestyle='-', color='blue')
    plt.plot(time_data, tq_bias_data[:,0], label='tq_bias', linestyle='-', color='green')
    plt.plot(time_data, tq_load_data[:,0], label='fr_load', linestyle='--', color='red')
    plt.plot(time_data, tq_sim_data[:,0], label='tq_sim', linestyle='--', color='orange')
    plt.legend()
    plt.title('Swing')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), './tracking_result_ground/third_qfrc_swing_margin.png'))
    
    # Boom
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, tq_constraint_data[:,1], label='tq_cons', linestyle='-', color='blue')
    plt.plot(time_data, tq_bias_data[:,1], label='tq_bias', linestyle='-', color='green')
    plt.plot(time_data, tq_load_data[:,1], label='fr_load', linestyle='--', color='red')
    plt.plot(time_data, tq_sim_data[:,1], label='tq_sim', linestyle='--', color='orange')
    plt.legend()
    plt.title('Boom')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), './tracking_result_ground/third_qfrc_boom_margin.png'))
    
    # Arm
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, tq_constraint_data[:,2], label='tq_cons', linestyle='-', color='blue')
    plt.plot(time_data, tq_bias_data[:,2], label='tq_bias', linestyle='-', color='green')
    plt.plot(time_data, tq_load_data[:,2], label='fr_load', linestyle='--', color='red')
    plt.plot(time_data, tq_sim_data[:,2], label='tq_sim', linestyle='--', color='orange')
    plt.legend()
    plt.title('Arm')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), './tracking_result_ground/third_qfrc_arm_margin.png'))  
    
    # Bucket
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, tq_constraint_data[:,3], label='tq_cons', linestyle='-', color='blue')
    plt.plot(time_data, tq_bias_data[:,3], label='tq_bias', linestyle='-', color='green')
    plt.plot(time_data, tq_load_data[:,3], label='fr_load', linestyle='--', color='red')
    plt.plot(time_data, tq_sim_data[:,3], label='tq_sim', linestyle='--', color='orange')
    plt.legend()
    plt.title('Bucket')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), './tracking_result_ground/third_qfrc_bucket_margin.png'))
    
    
    # Swing
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_data, sim_disp_data[:,0], label='sim_disp', linestyle='-', color='blue')
    plt.plot(time_data, mj_disp_data[:,0], label='mj_disp', linestyle='-', color='green')
    plt.plot(time_data, error_disp_data[:,0], label='error_disp', linestyle='--', color='orange')
    plt.legend()
    plt.title('Swing: Angel in Simulink, Angel in Mujoco and Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_data, sim_vel_data[:,0], label='sim_vel', linestyle='-', color='blue')
    plt.plot(time_data, mj_vel_data[:,0], label='mj_vel', linestyle='-', color='green')
    plt.plot(time_data, error_vel_data[:,0], label='error_vel', linestyle='--', color='orange')
    plt.legend()
    plt.title('Swing: Angular Velocity in Simulink, Angular Velocity in Mujoco and Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), './tracking_result_ground/third_swing_50Hz_margin.png'))
    
    # Boom
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_data, sim_disp_data[:,1], label='sim_disp', linestyle='-', color='blue')
    plt.plot(time_data, mj_disp_data[:,1], label='mj_disp', linestyle='-', color='green')
    plt.plot(time_data, error_disp_data[:,1], label='error_disp', linestyle='--', color='orange')
    plt.legend()
    plt.title('Boom: Displacement in Simulink, Displacement in Mujoco and Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_data, sim_vel_data[:,1], label='sim_vel', linestyle='-', color='blue')
    plt.plot(time_data, mj_vel_data[:,1], label='mj_vel', linestyle='-', color='green')
    plt.plot(time_data, error_vel_data[:,1], label='error_vel', linestyle='--', color='orange')
    plt.legend()
    plt.title('Boom: Velocity in Simulink, Velocity in Mujoco and Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), './tracking_result_ground/third_boom_50Hz_margin.png'))
    
    # Arm
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_data, sim_disp_data[:,2], label='sim_disp', linestyle='-', color='blue')
    plt.plot(time_data, mj_disp_data[:,2], label='mj_disp', linestyle='-', color='green')
    plt.plot(time_data, error_disp_data[:,2], label='error_disp', linestyle='--', color='orange')
    plt.legend()
    plt.title('Arm: Displacement in Simulink, Displacement in Mujoco and Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_data, sim_vel_data[:,2], label='sim_vel', linestyle='-', color='blue')
    plt.plot(time_data, mj_vel_data[:,2], label='mj_vel', linestyle='-', color='green')
    plt.plot(time_data, error_vel_data[:,2], label='error_vel', linestyle='--', color='orange')
    plt.legend()
    plt.title('Arm: Velocity in Simulink, Velocity in Mujoco and Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), './tracking_result_ground/third_arm_50Hz_margin.png'))      
    
    # Bucket
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_data, sim_disp_data[:,3], label='sim_disp', linestyle='-', color='blue')
    plt.plot(time_data, mj_disp_data[:,3], label='mj_disp', linestyle='-', color='green')
    plt.plot(time_data, error_disp_data[:,3], label='error_disp', linestyle='--', color='orange')
    plt.legend()
    plt.title('Bucket: Displacement in Simulink, Displacement in Mujoco and Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_data, sim_vel_data[:,3], label='sim_vel', linestyle='-', color='blue')
    plt.plot(time_data, mj_vel_data[:,3], label='mj_vel', linestyle='-', color='green')
    plt.plot(time_data, error_vel_data[:,3], label='error_vel', linestyle='--', color='orange')
    plt.legend()
    plt.title('Bucket: Velocity in Simulink, Velocity in Mujoco and Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), './tracking_result_ground/third_bucket_50Hz_margin.png'))

        
if __name__ == "__main__":
    test_t2f_third()