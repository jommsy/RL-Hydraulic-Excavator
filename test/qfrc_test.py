# test the tracking of actual joint velocity and input actuator veloctiy
from dm_control.mujoco import Physics
import matplotlib.pyplot as plt
import os
import mujoco
import numpy as np
import cv2
import matplotlib
from sympy import im
matplotlib.use('Agg')

_CONTROL_JOINTS = ['cabin', 'boom', 'arm', 'blink_2'] 

def model_path():
    model_path = os.path.join(os.path.dirname(
        __file__), "../assets/excavator.xml")
    return model_path


def flatten_array(nested_array):
    """
    将嵌套的数组展平成一维数组。

    参数:
    nested_array (list): 可能嵌套的数组

    返回:
    list: 展平的一维数组
    """
    flat_list = []
    for item in nested_array:
        if isinstance(item, list):  # 如果item是list，递归调用flatten_array
            flat_list.extend(flatten_array(item))
        else:
            flat_list.append(item)  # 否则直接添加到flat_list
    return flat_list

# get the sensor data by sensor id


def get_sensor_by_id(model, data, sensor_id):
    sensor_name = sensor_id
    sensor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    sensor_data = data.sensordata[sensor_adr:sensor_adr + sensor_dim]
    return sensor_data

# map the address of the actuator by id


def get_actuator_id(model, actuator_name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)


def get_joint_id(model, joint_name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)


def get_equality_constraint_ids(model, body_name):
    constraint_ids = []
    body_ids = []
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    for i in range(model.eq_data.shape[0]):
        obj1 = model.eq_obj1id[i]
        obj2 = model.eq_obj2id[i]
        body_ids.append(obj1)
        body_ids.append(obj2)
        if obj1 == body_id or obj2 == body_id:
            constraint_ids.append(i)
    with open('z-excavator_sim/test/model_data.csv', 'a') as f:
        f.write(f'equality_constrains_bodys_id:, {body_ids}\n\n')
    return constraint_ids

def get_body_id(model):
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        with open('z-excavator_sim/test/model_data.csv', 'a') as f:
            f.write(f'Body ID: {i}, Name: {body_name}\n')

# 定义解析 efc_id 的函数
def parse_efc_id(model, data):
    results = []
    for i, efc_id in enumerate(data.efc_id):
        # 获取约束类型
        efc_type = data.efc_type[efc_id]

        type_str = ''
        if efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY:
            type_str = 'Equality Constraint'
        elif efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF:
            type_str = 'Friction DOF'
        elif efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON:
            type_str = 'Friction Tendon'
        elif efc_type == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT:
            type_str = 'Joint Limit'
        elif efc_type == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON:
            type_str = 'Tendon Limit'
        elif efc_type == mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS:
            type_str = 'Contact Frictionless'
        elif efc_type == mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL:
            type_str = 'Contact Pyramidal'
        elif efc_type == mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC:
            type_str = 'Contact Elliptic'
        else:
            type_str = 'Unknown'

        result = {
            'efc_id': efc_id,
            'efc_index': i,
            'type': type_str,
            'efc_force': data.efc_force[i]
        }
        results.append(result)

    # 打印解析结果
    for res in results:
        print(f"EFC Index: {res['efc_index']}, EFC ID: {res['efc_id']}, Type: {res['type']}, "
              f"EFC Force: {res['efc_force']}")


# parse nv_id
def parse_nv_id(model, data):
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        dof_index = model.jnt_dofadr[i]
        dof_type = model.jnt_type[i]
        with open('z-excavator_sim/test/model_data.csv', 'a') as f:
            f.write(
                f'Joint name: {joint_name}, DOF index: {dof_index}, DOF type: {dof_type}\n')


# parse efc_id
def parse_efc_id(model, data):
    get_body_id(model)
    get_equality_constraint_ids(model, 'boom')
    for i in range(data.nefc):
        efc_id = data.efc_id[i]
        efc_type = data.efc_type[efc_id]
        efc_force = data.efc_force[efc_id]
        efc_frictionloss = data.efc_frictionloss[efc_id]
        constraint_state = data.efc_state[efc_id]
        with open('z-excavator_sim/test/model_data.csv', 'a') as f:
            f.write(
                f'EFC ID: {efc_id}, EFC Type: {efc_type}, EFC Force: {efc_force}, '
                f'EFC Friction: {efc_frictionloss}, Constraint State: {constraint_state}\n')


def get_constraint_forces(model, data, constraint_ids):
    forces = []
    for cid in constraint_ids:
        print(data.efc_id)
        parse_efc_id(model, data)
        efc_id = data.efc_id[cid]
        force = data.efc_force[efc_id]
        forces.append(force)
    return forces


def get_model_data_properties(model, data):
    with open('z-excavator_sim/test/model_data_0.csv', 'a') as f:
        f.write(f'qfrc_applied:, {data.qfrc_applied}\n')
        f.write(f'qfrc_passive:, {data.qfrc_passive}\n')
        f.write(f'qfrc_bias:, {data.qfrc_bias}\n')
        f.write(f'qfrc_constraint:, {data.qfrc_constraint}\n \n')
    # parse_nv_id(model, data)
    # parse_efc_id(model, data)


def step_signal(t, a, period=1):
    return a if (t % period) < (period / 2) else -a


def sine_signal(t, a, period=1):
    return a * np.sin(2 * np.pi * t / period)

""" 
test the qfrc_applied
"""

def test_external_force():
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
    dt = model.opt.timestep
    frames = []
    mujoco.mj_resetData(model, data)
    dof_address = [model.jnt_dofadr[get_joint_id(model, joint)] for joint in _CONTROL_JOINTS]
    print(dof_address)
    
    for step in range(num_steps):
        t = step * dt
        mujoco.mj_step1(model, data)
        mujoco.mj_step2(model, data)
        force_constraint = data.qfrc_constraint[dof_address]
        force_bias = data.qfrc_bias[dof_address]
        force_applied = force_constraint + force_bias
        
        data.qfrc_applied[dof_address] = force_applied

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

if __name__ == '__main__':
    test_external_force()
