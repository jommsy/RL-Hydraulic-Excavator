import hydra
from utils.parser import parse_cfg
from wrappers.dmc2gym import make_env
from utils.seed import set_seed
from stable_baselines3 import A2C, PPO, TD3, DDPG, SAC
from datetime import datetime
from gymnasium.wrappers import record_video
import cv2
from dm_control.mujoco.engine import Camera
import matplotlib.pyplot as plt
import os

_SENSOR_POS = ['jointpos_swing', 'jointpos_boom', 'jointpos_arm', 'jointpos_bucket']
_SENSOR_VEL = ['jointvel_swing', 'jointvel_boom', 'jointvel_arm', 'jointvel_bucket']
def grabFrame(env):
    # Get RGB rendering of env
    camera = Camera(env.physics, height=1080, width=1920, camera_id=-1)
    camera._render_camera.distance = 15
    rgbArr = camera.render()
    return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)
        
@hydra.main(config_name='config', config_path='.', version_base=None)   
def show_result(cfg: dict):
    hydra.core.utils.configure_log(None)
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    cfg.task = 'digging_soil'
    cfg.seed = 45
    env = make_env(cfg, "human")
    
    # video_name = f"logs/PPO/hauling_3.mp4"
    # frame = grabFrame(env)
    # height = 1080
    # width = 1920
    # video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
    
    time_data = []
    distance_data = []
    # model = PPO.load(f"{cfg.log_dir}/../{cfg.task}_1.zip")  # timesteps = 200.000
    model = PPO.load(f"{cfg.log_dir}/../ds_2.zip")  # timesteps = 200.000
    obs, info = env.reset()
    for i in range(250):
        time = env._physics.data.time
        action, _ = model.predict(obs, deterministic=True)
        with open (f"{cfg.log_dir}/../{cfg.task}_action.csv", 'a') as f:
            f.write(f"{time}, {action}\n")
        obs, reward, terminated, truncated, info = env.step(action)
        # frame = grabFrame(env)
        # video.write(frame)
    
if __name__ == '__main__':
    show_result()


