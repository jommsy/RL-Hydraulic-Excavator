from dm_control.mujoco import Physics
import matplotlib.pyplot as plt
import os
import time
import mujoco
import mujoco.viewer
import numpy as np
import cv2
import matplotlib
from sympy import im
matplotlib.use('Agg')

_SENSORS_TORQUE = ['torque_boom', 'torque_arm', 'torque_bucket']
_COMPONENT_LENGTH = [0.297059, 1.181326, 0.238248, 1.396755, 0.234974, 0.970948]
_DOF_ADDRESS = [0, 1, 2, 3]
_ACUTATOR_POS = ['swing_pos', 'boom_pos', 'arm_pos', 'bucket_pos']
_ACUTATOR_VEL = ['swing_vel', 'boom_vel', 'arm_vel', 'bucket_vel']
_JOINTS = ['cabin', 'boom', 'arm', 'bucket']

def model_path():
    model_path = os.path.join(os.path.dirname(
        __file__), "../assets/excavator.xml")
    return model_path


def site_distance(physics, site1, site2):
    site1_to_site2 = np.diff(
        physics.named.data.site_xpos[[site2, site1]], axis=0)
    site_distance = np.linalg.norm(site1_to_site2)
    print(site_distance)

def get_sensors_by_name(model, data, sensor_names):
    sensor_datas = []
    for sensor_name in sensor_names:
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        sensor_adr = model.sensor_adr[sensor_id]
        sensor_dim = model.sensor_dim[sensor_id]
        sensor_datas.append(data.sensordata[sensor_adr:sensor_adr + sensor_dim])
    return sensor_datas

def get_sensor_by_name(model, data, sensor_name):
    sensor_data = []
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    sensor_data.append(data.sensordata[sensor_adr:sensor_adr + sensor_dim])
    return sensor_data

def get_actuator_id(model, actuator_name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)

def sine_signal(t, a, period=1):
    if t < 0:
        return -1
    else:
        return a * np.sin(2 * np.pi * t / period)
    
def step_signal(t, a, period=1):
    return a if (t % period) < (period / 2) else -a


def parse_nv_id():
    try:
        model = mujoco.MjModel.from_xml_path(model_path())
    except Exception as e:
        print(f"Error: {e}")

    data = mujoco.MjData(model)
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        dof_index = model.jnt_dofadr[i]
        dof_type = model.jnt_type[i]
        with open('./test/model_data.csv', 'a') as f:
            f.write(
                f'Joint name: {joint_name}, DOF index: {dof_index}, DOF type: {dof_type}\n')    
            
"""
test which dimension of senor data is needed.
result: the sensor data dimension is consistent with the rotation axis
"""           
def t2f_direction():
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
    time_data = []
    data1 = []
    data2 = []
    data3 = []
    actuator = get_actuator_id(model, "arm_motor")
    mujoco.mj_resetData(model, data)
    mujoco.mj_step1(model, data)
    start_time = time.time()
    data.ctrl[get_actuator_id(model, "boom_motor")] = -0.5
    while data.time < duration:
        data.ctrl[actuator] = sine_signal(data.time, 0.5, period=3)
        sensor_data = get_sensor_by_name(model, data, "torque_arm")
        mujoco.mj_step(model, data)
        if len(frames) < data.time * framrate:
            renderer.update_scene(data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)
        with open ("./test/t2f_test.csv", "a") as f:
            f.write(f"{sensor_data}\n")
        time_data.append(data.time)
        data1.append(sensor_data[0][0])
        data2.append(sensor_data[0][1])
        data3.append(sensor_data[0][2])

    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("video", frame)
        if cv2.waitKey(int(1000 // framrate)) & 0xFF == ord('q'):
            break    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.figure(figsize=(10, 6))
    plt.plot(time_data, data1, label='data1', linestyle='-', color='blue')
    plt.plot(time_data, data2, label='data2', linestyle='-', color='green')
    plt.plot(time_data, data3, label='data3', linestyle='--', color='orange')
    plt.legend()
    plt.title('Swing: Angel Signal, Joint Angel, and Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), 't2f.png'))    
    
    
def t2f():
    physics = Physics.from_xml_path(model_path())
    torques = physics.named.data.sensordata[_SENSORS_TORQUE]
    torque_boom = torques[1]
    torque_arm = torques[4]
    torque_bucket = torques[7]
    length_boom = site_distance(physics, 'boom_cylinder', 'boom_rod')
    length_arm = site_distance(physics, 'arm_cylinder', 'arm_rod')
    length_bucket = site_distance(physics, 'bucket_cylinder', 'bucket_rod')
    theta1 = np.arccos((length_boom**2 + _COMPONENT_LENGTH[1]**2 - _COMPONENT_LENGTH[0]**2) / (2 * length_boom * _COMPONENT_LENGTH[1]))
    theta2 = np.arccos((length_arm**2 + _COMPONENT_LENGTH[2]**2 - _COMPONENT_LENGTH[3]**2) / (2 * length_arm * _COMPONENT_LENGTH[2]))
    theta3 = np.arccos((length_bucket**2 + _COMPONENT_LENGTH[4]**2 - _COMPONENT_LENGTH[5]**2) / (2 * length_bucket * _COMPONENT_LENGTH[4]))
    load_boom = torque_boom / (_COMPONENT_LENGTH[1] * np.sin(theta1))
    load_arm = torque_arm / (_COMPONENT_LENGTH[2] * np.sin(theta2))
    load_bucket = torque_bucket / (_COMPONENT_LENGTH[4] * np.sin(theta3))

def site_id():
    try:
        model = mujoco.MjModel.from_xml_path(model_path())
    except Exception as e:
        print(f"Error: {e}")
    
    data = mujoco.MjData(model)
    mujoco.mj_step1(model,data)
    site_names = ["boom_cylinder", "boom_rod"]
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in site_names]

    # 获取 site 的位置
    site_xpos = data.site_xpos[[0,1]]
    print(site_xpos)


            
            
"""
impunity: when combine with matlab, the load (qfrc_constraint + qfrc_bias) is too large
when use qfrc_applied = qfrc_constraint + qfrc_bias, the model will be chaos instead of kepping still
1. test how the qfrc_constraint and qfrc_bias changed when using qfrc_applied = qfrc_constraint + qfrc_bias
"""
def test_qfrc():
    try:
        model = mujoco.MjModel.from_xml_path(model_path())
    except Exception as e:
        print(f"Error: {e}")

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)
    duration = 2
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
    torque_load = np.zeros(4)

    while data.time < duration:
        for i in range(1):
            torque_constraint = data.qfrc_constraint[_DOF_ADDRESS]
            torque_bias = data.qfrc_bias[_DOF_ADDRESS]
            torque_load[0] = torque_bias[0] - torque_constraint[0]
            torque_load[1] = torque_bias[1] - torque_constraint[1]
            torque_load[2] = torque_bias[2] - torque_constraint[2]
            torque_load[3] = torque_bias[3] - torque_constraint[3]
            
            # torque_load[0] = -(torque_bias[0] + torque_constraint[0])
            # torque_load[1] = -(torque_bias[1] + torque_constraint[1])
            # torque_load[2] = -(torque_bias[2] + torque_constraint[2])
            # torque_load[3] = -(torque_bias[3] + torque_constraint[3])
            
            # data.qfrc_applied[_DOF_ADDRESS[0]] = torque_load[0]
            # data.qfrc_applied[_DOF_ADDRESS[1]] = torque_load[1]
            # data.qfrc_applied[_DOF_ADDRESS[2]] = torque_load[2]
            # data.qfrc_applied[_DOF_ADDRESS[3]] = torque_load[3]
            
            # data.qfrc_applied[_DOF_ADDRESS] = [0.0, -3514.5123723536317, -263.3502101282814, 103.40054824990601]

            
            torque_applied = data.qfrc_applied[_DOF_ADDRESS]
            
            for i in range(1):        
                mujoco.mj_step2(model, data)
                mujoco.mj_step1(model, data) 
            
                time_data.append(data.time)
                tq_constraint_data.append(torque_constraint)
                tq_bias_data.append(torque_bias)
                tq_applied_data.append(torque_applied)
                
        # progress_bar.update(data.time - progress_bar.n)
        if len(frames) < data.time * framrate:
            renderer.update_scene(data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)

    time_data = np.array(time_data)
    n = time_data.shape[0]
    tq_constraint_data = np.array(tq_constraint_data).reshape(n,4)
    tq_bias_data = np.array(tq_bias_data).reshape(n,4)
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
    plt.legend()
    plt.title('Swing')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), './qfrc_result/mj_swing.png'))
    
    # Boom
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, tq_constraint_data[:,1], label='tq_cons', linestyle='-', color='blue')
    plt.plot(time_data, tq_bias_data[:,1], label='tq_bias', linestyle='-', color='green')
    plt.plot(time_data, tq_applied_data[:,1], label='tq_app', linestyle='--', color='orange')
    plt.legend()
    plt.title('Boom')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), './qfrc_result/mj_boom.png'))
    
    # Arm
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, tq_constraint_data[:,2], label='tq_cons', linestyle='-', color='blue')
    plt.plot(time_data, tq_bias_data[:,2], label='tq_bias', linestyle='-', color='green')
    plt.plot(time_data, tq_applied_data[:,2], label='tq_app', linestyle='--', color='orange')
    plt.legend()
    plt.title('Arm')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), './qfrc_result/mj_arm.png'))  
    
    # Bucket
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, tq_constraint_data[:,3], label='tq_cons', linestyle='-', color='blue')
    plt.plot(time_data, tq_bias_data[:,3], label='tq_bias', linestyle='-', color='green')
    plt.plot(time_data, tq_applied_data[:,3], label='tq_app', linestyle='--', color='orange')
    plt.legend()
    plt.title('Bucket')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), './qfrc_result/mj_bucket.png'))
 
 
 
""" 
test the tracking of gievn joint position with position and velocity controller for single joint.
"""

def get_sensor_by_id(model, data, sensor_id):
    sensor_name = sensor_id
    sensor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    sensor_data = data.sensordata[sensor_adr:sensor_adr + sensor_dim]
    return sensor_data

def test_gain_single():

    try:
        model = mujoco.MjModel.from_xml_path(model_path())
    except Exception as e:
        print(f"Error: {e}")

    time_data = []
    
    # pos_data_swing = []
    # joint_data_pos_swing = [] 
    # error_data_pos_swing = []
    # vel_data_swing = []
    # joint_data_vel_swing = []
    # error_data_vel_swing= []
    
    # pos_data_boom = []
    # joint_data_pos_boom= [] 
    # error_data_pos_boom = []
    # vel_data_boom = []
    # joint_data_vel_boom = []
    # error_data_vel_boom = []
    
    # pos_data_arm = []
    # joint_data_pos_arm = [] 
    # error_data_pos_arm = []
    # vel_data_arm = []
    # joint_data_vel_arm = []
    # error_data_vel_arm = []
    
    pos_data_bucket = []
    joint_data_pos_bucket = [] 
    error_data_pos_bucket = []
    vel_data_bucket = []
    joint_data_vel_bucket = []
    error_data_vel_bucket = []
    
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)
    dt = model.opt.timestep

    duration = 5
    total_steps = int(duration / dt)
    framrate = 30

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    frames = []
    mujoco.mj_resetData(model, data)
    
    data.ctrl[get_actuator_id(model, 'swing_pos')] = 0
    data.ctrl[get_actuator_id(model, 'boom_pos')] = -350
    data.ctrl[get_actuator_id(model, 'arm_pos')] = 450
    
    for step in range(total_steps):
        t = step * dt
        # pos_signal_swing = sine_signal(t, 1, period=8)
        # pos_signal_boom = sine_signal(t, 0.1, period=2)
        # pos_signal_arm = sine_signal(t, 0.15, period=2) - 0.1
        pos_signal_bucket = step_signal(t, 560, period=2) -160
        # data.ctrl[get_actuator_id(model, 'swing_pos_pid')] = pos_signal_swing
        # data.ctrl[get_actuator_id(model, 'boom_pos_pid')] = pos_signal_boom
        # data.ctrl[get_actuator_id(model, 'arm_pos_pid')] = pos_signal_arm
        data.ctrl[get_actuator_id(model, 'bucket_pos')] = pos_signal_bucket
        
        mujoco.mj_step(model, data)
        # jointpos_swing = get_sensor_by_id(model, data, 'jointpos_swing')
        # jointpos_boom_rod = get_sensor_by_id(model, data, 'jointpos_boom_rod')
        # jointpos_arm_rod = get_sensor_by_id(model, data, 'jointpos_arm_rod')
        jointpos_bucket = get_sensor_by_id(model, data, 'jointpos_bucket') * 400

        # jointvel_swing = get_sensor_by_id(model, data, 'jointvel_swing')
        # jointvel_boom_rod = get_sensor_by_id(model, data, 'jointvel_boom_rod')
        # jointvel_arm_rod = get_sensor_by_id(model, data, 'jointvel_arm_rod')
        jointvel_bucket = get_sensor_by_id(model, data, 'jointvel_bucket')
                
        # error_pos_swing = pos_signal_swing - jointpos_swing[0]
        # error_pos_boom = pos_signal_boom - jointpos_boom_rod[0]
        # error_pos_arm = pos_signal_arm - jointpos_arm_rod[0]
        error_pos_bucket = pos_signal_bucket - jointpos_bucket[0]

        time_data.append(t)
        # pos_data_swing.append(pos_signal_swing)
        # pos_data_boom.append(pos_signal_boom)
        # pos_data_arm.append(pos_signal_arm)
        pos_data_bucket.append(pos_signal_bucket)
        
        # error_data_pos_swing.append(error_pos_swing)
        # error_data_pos_boom.append(error_pos_boom)
        # error_data_pos_arm.append(error_pos_arm)
        error_data_pos_bucket.append(error_pos_bucket)
        
        # joint_data_pos_swing.append(jointpos_swing[0])  # scalar sensor
        # joint_data_pos_boom.append(jointpos_boom_rod[0])  # scalar sensor
        # joint_data_pos_arm.append(jointpos_arm_rod[0])  # scalar sensor
        joint_data_pos_bucket.append(jointpos_bucket[0])  # scalar sensor
        
        # joint_data_vel_swing.append(jointvel_swing[0])  # scalar sensor
        # joint_data_vel_boom.append(jointvel_boom_rod[0])  # scalar sensor
        # joint_data_vel_arm.append(jointvel_arm_rod[0])  # scalar sensor
        joint_data_vel_bucket.append(jointvel_bucket[0])  # scalar sensor

        if len(frames) < data.time * framrate:
            renderer.update_scene(data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)

    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("video", frame)
        if cv2.waitKey(int(1000 // framrate)) & 0xFF == ord('q'):
            break

    # plot position
    plt.figure(figsize=(10, 6))
    
    # plt.plot(time_data, pos_data_swing, label='Pos Signal', linestyle='-', color='blue')
    # plt.plot(time_data, error_data_pos_swing, label='Pos Error Signal', linestyle='-', color='green')
    # plt.plot(time_data, joint_data_pos_swing, label='Joint Postion Sensor', linestyle='--', color='orange')
    # plt.legend()
    # plt.title('Swing: Pos Signal, Joint Pos, and Actuator Pos Over Time')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Values')
    # plt.grid(True)
    # plt.savefig(os.path.join(os.path.dirname(__file__), 'sine_pos_sw.png'))

    # plt.plot(time_data, pos_data_boom, label='Pos Signal', linestyle='-', color='blue')
    # plt.plot(time_data, error_data_pos_boom, label='Pos Error Signal', linestyle='-', color='green')
    # plt.plot(time_data, joint_data_pos_boom, label='Joint Postion Sensor', linestyle='--', color='orange')
    # plt.legend()
    # plt.title('Boom: Pos Signal, Joint Pos, and Actuator Pos Over Time')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Values')
    # plt.grid(True)
    # plt.savefig(os.path.join(os.path.dirname(__file__), 'sine_pos_bm.png'))

    # plt.plot(time_data, pos_data_arm, label='Pos Signal', linestyle='-', color='blue')
    # plt.plot(time_data, error_data_pos_arm, label='Pos Error Signal', linestyle='-', color='green')
    # plt.plot(time_data, joint_data_pos_arm, label='Joint Postion Sensor', linestyle='--', color='orange')
    # plt.legend()
    # plt.title('Arm: Pos Signal, Joint Pos, and Actuator Pos Over Time')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Values')
    # plt.grid(True)
    # plt.savefig(os.path.join(os.path.dirname(__file__), 'sine_pos_am.png'))
    
    plt.plot(time_data, pos_data_bucket, label='Pos Signal', linestyle='-', color='blue')
    plt.plot(time_data, error_data_pos_bucket, label='Pos Error Signal', linestyle='-', color='green')
    plt.plot(time_data, joint_data_pos_bucket, label='Joint Postion Sensor', linestyle='--', color='orange')
    print(joint_data_pos_bucket[0])
    plt.legend()
    plt.title('Bucket: Pos Signal, Joint Pos, and Actuator Pos Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'step_pos_bk.png'))
    
    # plot velocity
    plt.figure(figsize=(10, 6))
    
    # plt.plot(time_data, joint_data_vel_swing, label='Joint Velocity Sensor', linestyle='--', color='orange')
    # plt.legend()
    # plt.title('Swing Vel Over Time')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Values')
    # plt.grid(True)
    # plt.savefig(os.path.join(os.path.dirname(__file__), 'sine_vel_sw.png'))
    
    # plt.plot(time_data, joint_data_vel_boom, label='Joint Velocity Sensor', linestyle='--', color='orange')
    # plt.legend()
    # plt.title('Boom Vel Over Time')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Values')
    # plt.grid(True)
    # plt.savefig(os.path.join(os.path.dirname(__file__), 'sine_vel_bm.png'))
    
    # plt.plot(time_data, joint_data_vel_arm, label='Joint Velocity Sensor', linestyle='--', color='orange')
    # plt.legend()
    # plt.title('Arm Vel Over Time')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Values')
    # plt.grid(True)
    # plt.savefig(os.path.join(os.path.dirname(__file__), 'sine_vel_am.png'))
    
    plt.plot(time_data, joint_data_vel_bucket, label='Joint Velocity Sensor', linestyle='--', color='orange')
    plt.legend()
    plt.title('Bucket Vel Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'step_vel_bk.png'))
    
def test_site(site1, site2):
    model = mujoco.MjModel.from_xml_path(model_path())
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) for site in [site1, site2]]
    site1_to_site2 = np.diff(data.site_xpos[site_ids], axis=0)
    print(np.linalg.norm(site1_to_site2))
    # print(data.site_xpos[1][2])    
    
def test_actuator_id():
    model = mujoco.MjModel.from_xml_path(model_path())
    for _, (name1, name2) in enumerate(zip(_ACUTATOR_POS, _ACUTATOR_VEL)):
        print(get_actuator_id(model, name1))
        print(get_actuator_id(model, name2))

def test_swing_wrap_angle():

    try:
        model = mujoco.MjModel.from_xml_path(model_path())
    except Exception as e:
        print(f"Error: {e}")

    time_data = []
    
    pos_data_swing = []
    joint_data_pos_swing = [] 
    error_data_pos_swing = []
    vel_data_swing = []
    joint_data_vel_swing = []
    error_data_vel_swing= []
    
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)
    dt = model.opt.timestep

    duration = 5
    total_steps = int(duration / dt)
    framrate = 30

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    frames = []
    mujoco.mj_resetData(model, data)
    
    data.ctrl[get_actuator_id(model, 'swing_vel')] = 0.1
    
    for step in range(total_steps):
        t = step * dt
        pos_signal_swing = (sine_signal(t, 0.75*np.pi, period=4)) * 1000

        data.ctrl[get_actuator_id(model, 'swing_pos')] = pos_signal_swing
        mujoco.mj_step(model, data)
        jointpos_swing = get_sensor_by_id(model, data, 'jointpos_swing') * 1000
        jointvel_swing = get_sensor_by_id(model, data, 'jointvel_swing') * 1000
        error_pos_swing = pos_signal_swing - jointpos_swing[0]
        time_data.append(t)
        pos_data_swing.append(pos_signal_swing)
        error_data_pos_swing.append(error_pos_swing)
        joint_data_pos_swing.append(jointpos_swing[0])  # scalar sensor
        joint_data_vel_swing.append(jointvel_swing[0])  # scalar sensor

        if len(frames) < data.time * framrate:
            renderer.update_scene(data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)

    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("video", frame)
        if cv2.waitKey(int(1000 // framrate)) & 0xFF == ord('q'):
            break

    plt.figure(figsize=(10, 6))
    
    plt.plot(time_data, pos_data_swing, label='Pos Signal', linestyle='-', color='blue')
    plt.plot(time_data, error_data_pos_swing, label='Pos Error Signal', linestyle='-', color='green')
    plt.plot(time_data, joint_data_pos_swing, label='Joint Postion Sensor', linestyle='--', color='orange')
    plt.legend()
    plt.title('Swing: Pos Signal, Joint Pos, and Actuator Pos Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'swing_angle_wrap_test_pos.png'))
    
    # plot velocity
    plt.figure(figsize=(10, 6))
    
    plt.plot(time_data, joint_data_vel_swing, label='Joint Velocity Sensor', linestyle='--', color='orange')
    plt.legend()
    plt.title('Swing Vel Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'swing_angle_wrap_test_vel.png'))

def test_site_xpos(names):
    model = mujoco.MjModel.from_xml_path(model_path())
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in names]
    for id in site_ids:
        print(data.site_xpos[id])
        
        
        
def test_applyFT():
    xml = """
    <mujoco>
        <compiler autolimits="true"/>

        <option timestep="0.002" integrator="RK4"/>

        <asset>
            <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
            <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
        </asset>

        <worldbody>
            <geom name="floor" type="plane" size="3 3 .01" material="grid"/>
            <light pos="0 0 3"/>

            <body name="box1" pos="0.0 0 0.5">
                <freejoint/>
                <geom name="box1" type="box" size=".05 .05 .05" pos="0 0 0" mass="1"/>
            </body>

        </worldbody>    
    </mujoco>    
    """

    # 根据XML字符串创建MjModel对象
    m = mujoco.MjModel.from_xml_string(xml)
    # 创建MjData对象，用于存储模拟数据
    d = mujoco.MjData(m)
    print(d.xipos)
    # 初始化一个标志，用于控制初始阶段的行为
    first = True

    # 启动一个被动的图形界面查看器
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # 设置自动关闭查看器的时间限制
        start = time.time()
        while viewer.is_running() and time.time() - start < 30:
            # 记录当前步骤开始的时间
            step_start = time.time()

            # 在模拟开始后的前两秒内对物体施加向上的力
            if time.time() - start < 50:
                mujoco.mj_applyFT(
                    m,
                    d,
                    [0 ,2 , 9.81],  # 力的大小
                    [0,0,0],  # 扭矩（这里不施加）
                    [0 ,0 ,0],  # 应用力的位置
                    1,  # 物体ID
                    d.qfrc_applied,  # 应用力的数组
                )
            
                
            # 更新物理模拟状态
            mujoco.mj_step(m, d)
            print(d.xipos)
            mujoco.mju_zero(d.qfrc_applied)
            # 同步查看器，更新图形界面显示
            viewer.sync()

            # 简易的时间控制，确保每个模拟步骤与设定的时间步长一致
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
      
def test_applyFT_excavator():
    # 根据XML字符串创建MjModel对象
    m = mujoco.MjModel.from_xml_path(model_path())
    # 创建MjData对象，用于存储模拟数据
    d = mujoco.MjData(m)
    
    # 获取bucket的id
    bucket_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bucket")
    #获取bucket_site的id
    bucket_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "bucket_attach")
    print(bucket_id)
    print(bucket_site_id)
    # 初始化一个标志，用于控制初始阶段的行为
    first = True

    # 启动一个被动的图形界面查看器
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # 设置自动关闭查看器的时间限制
        start = time.time()
        while viewer.is_running() and time.time() - start < 130:
            # 记录当前步骤开始的时间
            step_start = time.time()

            # 在模拟开始后的前两秒内对物体施加向上的力
            if time.time() - start < 130:
                mujoco.mj_applyFT(
                    m,
                    d,
                    [0.1 ,0 , 0.1],  # 力的大小
                    [0,0,0],  # 扭矩（这里不施加）
                    d.site_xpos[16],  # 应用力的位置
                    7,  # 物体ID
                    d.qfrc_applied,  # 应用力的数组
                )
            
                
            # 更新物理模拟状态
            mujoco.mj_step(m, d)
            print(d.qfrc_applied)
            # mujoco.mju_zero(d.qfrc_applied)
            # 同步查看器，更新图形界面显示
            viewer.sync()

            # 简易的时间控制，确保每个模拟步骤与设定的时间步长一致
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                      
if __name__ == "__main__":
    test_applyFT()