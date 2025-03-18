from dm_control.mujoco import Physics
import numpy as np
import os
import mujoco


_JOINTS = ['cabin', 'boom', 'arm', 'bucket']
_CYLINDER_ROD_PAIRS = [('boom_cylinder', 'boom_rod'),
                    ('arm_cylinder', 'arm_rod'),
                    ('blink1', 'blink2')]
_CONSTANT_THETA = [0.03, 2.925065]
_INIT_CYLINDER_LENGTH = [1.236, 1.288, 0.9]
_INIT_SIM_POS = [0.45, 0.45, 0.25]

def model_path():
    return os.path.join(os.path.dirname(__file__), "../assets/excavator.xml")

def site_distance(physics, site1, site2):
	site1_to_site2 = np.diff(
		physics.named.data.site_xpos[[site2, site1]], axis=0)
	return np.linalg.norm(site1_to_site2)

def cos_theorem(a, b, c, flag):
	if flag == 'theta':
		cos_theta = (a**2 + b**2 - c**2) / (2 * a * b)
		if cos_theta < -1 or cos_theta > 1:
			raise ValueError(f"无效的 cos 值：{cos_theta}, 不能计算 arccos。")
		return np.arccos(cos_theta)
	elif flag == 'length':
		return np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(c))
        
def compute_init_sim_pos(init_mj_angles):
	O3g	= 0.157623
	gf	= 0.970948
	gh	= 0.234974
	eh	= 0.225615
	O3e	= 0.187843
    
	physics = Physics.from_xml_path(model_path())
	data = physics.named.data
	data.qpos[_JOINTS] = init_mj_angles
	mujoco.mj_step1(physics.model.ptr, physics.data.ptr)
	mj_distance_tmp = np.array([site_distance(physics, pair[0], pair[1]) for pair in _CYLINDER_ROD_PAIRS])
	# set a site in g, the last paris of cylinder and rod is ('e', 'g')
	O3GE = cos_theorem(O3g, mj_distance_tmp[2], O3e, 'theta')
	EGH = cos_theorem(mj_distance_tmp[2], gh, eh, 'theta')
	HGF = _CONSTANT_THETA[1] - O3GE - EGH
	mj_distance_tmp[-1] = cos_theorem(gh, gf, HGF, 'length')
	# mj_angle_cabin = get_sensor_by_name(model, data, 'jointpos_swing')
	# mj_distance = np.insert(mj_distance_tmp, 0, mj_angle_cabin)
	disp_bias =  mj_distance_tmp - _INIT_CYLINDER_LENGTH
	init_sim_pos =  np.round((_INIT_SIM_POS + disp_bias), 5)
	disp_bias = np.round(np.insert(disp_bias, 0, init_mj_angles[0]), 5)
	return disp_bias, init_sim_pos