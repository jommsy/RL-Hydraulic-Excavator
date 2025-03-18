from dm_control import suite
import dm_env
from dm_env import StepType, specs
from shimmy.dm_control_compatibility import DmControlCompatibilityV0
import numpy as np
from domain_task import task
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)


class ActionRoundWrapper(dm_env.Environment):
	def __init__(self, env, num_round):
		self._env = env
		self._num_round = num_round

	def step(self, action):
		action = np.round(action, self._num_round)
		return self._env.step(action)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._env.action_spec()

	def reset(self):
		return self._env.reset()

	def __getattr__(self, name):
		return getattr(self._env, name)


def make_env(cfg, render=None):
    """
    Make DMControl environment.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    domain, task = cfg.domain, cfg.task
    if (domain, task) not in suite.ALL_TASKS:
        raise ValueError('Unknown task:', task)
    env = suite.load(domain,
                     task,
                     task_kwargs={'random': cfg.seed, 
                                  'init_mj_angle': cfg.init_mj_angle[cfg.task]},
                     environment_kwargs={'num_actions': cfg.num_actions[cfg.task]},
                     visualize_reward=False)

    env = ActionRoundWrapper(env, 3)
    env = DmControlCompatibilityV0(env, render)
    # env = Monitor(env)
    # env = DummyVecEnv([lambda : env])
    return env
