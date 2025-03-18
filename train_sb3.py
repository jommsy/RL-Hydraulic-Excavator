import hydra
from utils.parser import parse_cfg
from wrappers.dmc2gym import make_env
from utils.seed import set_seed
from stable_baselines3 import A2C, PPO, TD3, DDPG, SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
import threading
from pynput import keyboard
from datetime import datetime
import torch

env_lock = threading.Lock()
alt_pressed = False
   
@hydra.main(config_name='config', config_path='.', version_base=None)   
def train(cfg: dict):
    global env
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    """ TRAIN WITH STABLE-BASELINES3 """
    with env_lock:
        env = make_env(cfg, render="human")
    print('env created')
    with env_lock:
        env.reset()
    eval_callback = EvalCallback(env,
                                best_model_save_path=f"{cfg.log_dir}",
                                log_path=f"{cfg.log_dir}",
                                deterministic=True,
                                render=False) 
    checkpoint_callback = CheckpointCallback(
                                save_freq=10000,
                                save_path=f"{cfg.model_log_dir}",
                                name_prefix="rl_model",
                                save_replay_buffer=True,
                                save_vecnormalize = True
                                )
    callback = CallbackList([eval_callback, checkpoint_callback])
    
    
    # model = PPO('MlpPolicy', 
    #             env, 
    #             policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    #             n_epochs=5,
    #             verbose=1, 
    #             clip_range=0.30,
    #             tensorboard_log=f"{cfg.tb_log_dir}")
    
    # model.learn(total_timesteps=cfg.timesteps, callback=callback, tb_log_name=cfg.tb_log_name, progress_bar=True)
    # model.save(f"{cfg.log_dir}/{'model_' + cfg.task + '_' + cfg.policy}")
    # print('model saved')

    
    model = PPO.load(f"{cfg.log_dir}/0317_0107/rl_model_450000_steps.zip", env=env, tensorboard_log=f"{cfg.log_dir}/0317_0107_1")
    # model = PPO.load(f"{cfg.log_dir}/best_model.zip", env=env, tensorboard_log=f"{cfg.log_dir}/0220_2130_1")
    model.learn(total_timesteps=cfg.timesteps, callback=callback, reset_num_timesteps=False, progress_bar=True)
    model.save(f"{cfg.log_dir}/{'model_' + cfg.task + '_' + cfg.policy}")
    print('model saved')
    
    # old_model = PPO.load(f"{cfg.log_dir}/0221_1523/rl_model_50000_steps.zip", env=env)
    # new_model = PPO('MlpPolicy', 
    #                 env, 
    #                 policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    #                 n_epochs=5, 
    #                 verbose=1, 
    #                 clip_range=0.3,
    #                 tensorboard_log=f"{cfg.tb_log_dir}")
    # new_model.set_parameters(old_model.get_parameters())
    # new_model.learn(total_timesteps=cfg.timesteps, callback=callback, progress_bar=True)
    # new_model.save(f"{cfg.log_dir}/{'model_' + cfg.task + '_' + cfg.policy}")
    # print('model saved') 

def on_press(key):
    global alt_pressed
    if key == keyboard.Key.alt_l:
        alt_pressed = True
    if hasattr(key, 'char') and key.char == 'v' and alt_pressed:
        with env_lock:
            if env.render_mode == 'human':
                env.render_mode = 'none'
                env.viewer.close()
            else:
                env.render_mode = 'human'
        print("Render mode changed to:", env.render_mode)

def on_release(key):
    global alt_pressed
    # 当 Alt 键被释放时，重置 alt_pressed
    if key == keyboard.Key.alt_l:
        alt_pressed = False
    
if __name__ == '__main__':
    # TODO: manage and track the whole change and training process with yaml or wandb or ...
    train_thread = threading.Thread(target=train)
    train_thread.start()
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()   
    train_thread.join()
    listener.join()


