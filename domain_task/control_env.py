from dm_control.rl import control
import dm_env
import numpy as np
from dm_env import specs
import collections

class custom_control_env(control.Environment):
    """this class is used to create a control environment for the hydraulic system"""

    def __init__(self,
                 physics,
                 task,
                 num_actions = 4,
                 step_limit = None,
                 flat_observation=False,
                 legacy_step: bool = True):

        self._task = task
        self._physics = physics
        self._physics.legacy_step = legacy_step
        self._flat_observation = flat_observation
        self._step_limit = step_limit
        self._step_count = 0
        self._reset_next_step = True
        self._num_actions = num_actions
        self.num_timesteps = 0
        
    def reset(self):
        """Starts a new episode and returns the first `TimeStep`."""
        self._reset_next_step = False
        self._step_count = 0
        with self._physics.reset_context():
            self._task.initialize_episode(self._physics)

        observation = self._task.get_observation(self._physics)
        if self._flat_observation:
            observation = control.Task.flatten_observation(observation)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            discount=None,
            observation=observation)
 
    def step(self, action):
        """Updates the environment using the action and returns a `TimeStep`."""
        # global start_time
        if self._reset_next_step:
            return self.reset()
        self.num_timesteps += 1
        control_timestep, com_timestep, n_sub_steps, n_sub_steps_control = self._task.get_n_steps(self._physics)
        
        if action.shape[0] == 3:
            action = np.insert(action, 0, 0.0)
            
        for i in range(n_sub_steps_control):
            soil_resistance = self._task.soil_resistance(self._physics)
            self._physics.before_step(action, com_timestep)
            self._physics.step(n_sub_steps, soil_resistance)
        self._task.after_step(self._physics)    
        observation = self._task.get_observation(self._physics)    
        reward = self._task.get_reward(self._physics, action, control_timestep, self._step_count, self.num_timesteps)
        if self._flat_observation:
            observation = control.Task.flatten_observation(observation)
            
        self._step_count += 1
        
        if self._step_count >= self._step_limit:
            discount = 1.0
        else:
            discount = self._task.get_termination(self._physics)

        episode_over = discount is not None

        if episode_over:
            self._reset_next_step = True
            return dm_env.TimeStep(
                dm_env.StepType.LAST, reward, discount, observation)
        else:
            return dm_env.TimeStep(dm_env.StepType.MID, reward, 1.0, observation)
        
    
    def action_spec(self):
        """Returns a `BoundedArraySpec` matching the `physics` actuators."""
        task_num_actions = self._num_actions
        minima = np.full(task_num_actions, fill_value=-1, dtype=np.float32)
        maxima = np.full(task_num_actions, fill_value=1, dtype=np.float32)
        return specs.BoundedArray(
            shape=(task_num_actions,), dtype=np.float32, minimum=minima, maximum=maxima)

    #  TODO: if it's necessary to consider the limit of vel, using clipping, if observation_spec() is needed.
    
    def observation_spec(self):
        """Returns the observation specification for this environment.

        Infers the spec from the observation, unless the Task implements the
        `observation_spec` method.

        Returns:
        An dict mapping observation name to `ArraySpec` containing observation
        shape and dtype.
        """
        try:
            return self._task.observation_spec(self._physics)
        except NotImplementedError:
            observation = self._task.get_observation(self._physics)
            if self._flat_observation:
                observation = control.flatten_observation(observation)
        if isinstance(observation, dict):
            result = collections.OrderedDict()
            for key, value in observation.items():
                result[key] = specs.Array(value.shape, value.dtype, name=key)
            return result
        elif isinstance(observation, np.ndarray):
            return specs.Array(observation.shape, observation.dtype)
