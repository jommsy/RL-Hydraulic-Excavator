import numpy as np
from dm_control.utils import rewards

class custom_rewards:
    def is_close(c, margin, distance, sigmoid):
        return rewards.tolerance(distance, (0, c), margin, sigmoid=sigmoid)

    def p1_goal_state(distance, dis_goal, vel, vel_goal, site_xpos, ori_goal):
        """
        goal_state: 1. the distance between the bucket_attach and dig_point < dis_goal
                2. the velocity of every joint < vel_goal
                3. bucket oriention, z_bottom - z_attach > ori_goal
        """
        termination = False
        bucket_ori_z = site_xpos[1][2] - site_xpos[0][2]
        if distance < dis_goal and np.all(np.abs(vel) < vel_goal) and bucket_ori_z > ori_goal:
            termination = True
            reward = 10.0
            return termination, reward
        else:
            reward = 0.0
            return termination, reward
        
    def p2_goal_state(distance, dis_goal, vel, vel_goal, site_xpos, bounds):
        """
        goal_state: 1. the distance between the bucket_attach and deep_point < dis_goal
                2. the velocity of every joint < vel_goal
                3. bucket oriention
        """  
        # ['bucket', 'bucket_attach']
        termination = False
        in_bounds = False
        lower, upper = bounds
        bucket_ori_z = site_xpos[0][2] - site_xpos[1][2]
        if site_xpos[0][0] **2 + site_xpos[0][1] **2 > site_xpos[1][0] **2 + site_xpos[1][1] **2:
            in_bounds = np.logical_and(lower <= bucket_ori_z, bucket_ori_z <= upper)  
        if distance < dis_goal and np.all(np.abs(vel) < vel_goal) and in_bounds:
            termination = True
            reward = 10.0
            return termination, reward
        else:
            reward = 0.0
            return termination, reward
        
    def p3_goal_state(distance, dis_goal, vel, vel_goal, site_xpos, ori_bounds):
        """
        goal_state: 1. the distance between the bucket_attach and deep_point < dis_goal
                2. the velocity of every joint < vel_goal
                3. bucket oriention
        """  
        #  ['bucket', 'bucket_attach']
        termination = False
        bucket_ori_z = site_xpos[0][2] - site_xpos[1][2]
        lower, upper = ori_bounds
        is_bounds = np.logical_and(lower <= bucket_ori_z, bucket_ori_z <= upper)
        if distance < dis_goal and np.all(np.abs(vel) < vel_goal) and is_bounds:
            termination = True
            reward = 10.0
            return termination, reward
        else:
            reward = 0.0
            return termination, reward

    def joint_limit(c, v, joint_upper_1, joint_lower_1, joint_upper_2, joint_lower_2, joint_qpos):
        reward_constraint = np.zeros(4)
        for i in range(4):
            qpos = joint_qpos[i]
            if qpos < joint_upper_1[i]:
                margin = (joint_upper_1[i] - joint_lower_1[i]) / 2
                reward_constraint[i] = rewards.tolerance(
                    qpos, (joint_lower_1[i], joint_lower_1[i] + c), margin, sigmoid='hyperbolic', value_at_margin=v)
            elif qpos > joint_lower_2[i]:
                margin = (joint_upper_2[i] - joint_lower_2[i]) / 2
                reward_constraint[i] = rewards.tolerance(
                    qpos, (joint_upper_2[i] - c, joint_upper_2[i]), margin, sigmoid='hyperbolic', value_at_margin=v)
            else:
                reward_constraint[i] = 0
        reward = -np.sum(reward_constraint) / 4
        return reward

    def is_collision(site_xpos, bounds_attach, margin_attach, v_attach, bounds_bottom, margin_bottom, v_bottom):
        attach_xpos_z = site_xpos[0][2]
        bottom_xpos_z = site_xpos[1][2]
        # choose margin = (upper - lower) / 2
        reward_attach = rewards.tolerance(
            attach_xpos_z, bounds_attach, margin_attach, sigmoid='hyperbolic', value_at_margin=v_attach)
        reward_bottom = rewards.tolerance(
            bottom_xpos_z, bounds_bottom, margin_bottom, sigmoid='hyperbolic', value_at_margin=v_bottom)
        reward = -(reward_attach + reward_bottom) / 2
        return reward

    def load_condition(vel_lower, vel_upper, cylinder_lower, cylinder_upper, sim_pos, sim_vel, action):
        # reward for negative load
        reward_neg_loads = np.zeros(4)
        reward_terminate = np.zeros(4)
        vel_max = [3.14, 1.0, 1.0, 1.0]
        c = 0.2
        v = 0.05
        for i in range(4):
            vel = sim_vel[i]
            if vel < vel_lower[i]:
                margin = (vel_lower[i] + vel_max[i]) / 2
                reward_neg_loads[i] = rewards.tolerance(
                    vel, (-vel_max[i], -vel_max[i] + c), margin, sigmoid='hyperbolic', value_at_margin=v)
            elif vel > vel_upper[i]:
                margin = (vel_max[i] - vel_upper[i]) / 2
                reward_neg_loads[i] = rewards.tolerance(
                    vel, (vel_max[i] - c, vel_max[i]), margin, sigmoid='hyperbolic', value_at_margin=v)

        # reward for cylinder terminate at the end (position limit) has been considered in joint_limit
        for i in range(4):
            if sim_pos[i] < (cylinder_lower[i] + 0.01) or sim_pos[i] > (cylinder_upper[i] - 0.01):
                in_bound = np.logical_and(-0.02 <=
                                          action[i], action[i] <= 0.02)
                reward_terminate[i] = np.where(
                    in_bound, 0.0, np.abs(action[i]) - 0.02)

        reward = - (np.sum(reward_neg_loads) +
                    np.sum(reward_terminate)) / 4
        return reward

    def action_limit(action, prev_action, control_timestep, physics_timestep, rate_limit, flag):
        # TODO: the rate the has been limited in Simulink, move it to reward function
        action_limit = rate_limit * physics_timestep
        if flag == 'normal':
            rewards_action = np.zeros(4)
            for i in range(4):
                action_change = abs(action[i] - prev_action[i]) / (control_timestep / physics_timestep)
                if action_change > action_limit:
                    rewards_action[i] = np.clip((action_change - action_limit), 0, 1)
            reward = -np.sum(rewards_action) / 4
            return reward
        elif flag == 'no_swing':
            rewards_action = np.zeros(4)
            if abs(action[0]) > 0.02:
                rewards_action[0] = 1 * 4
            for i in range(3):
                action_change = abs(action[i+1] - prev_action[i+1]) / (control_timestep / physics_timestep)
                if action_change > action_limit:
                    rewards_action[i+1] = np.clip((action_change - action_limit), 0, 1)
            reward = -np.sum(rewards_action) / 4
            return reward
        
    def no_swing(angle_diff, bounds, margin, v):
        return rewards.tolerance(angle_diff, bounds, margin, sigmoid='hyperbolic', value_at_margin=v)
        
    def bucket_orientation(site_xpos, bounds, margin, v, flag):
        # site_xpos_pair = [bucket, bucket_attach]
        bucket_ori_z = site_xpos[0][2] - site_xpos[1][2]
        # for touch_point
        if flag == 'touch':
            if site_xpos[0][0] **2 + site_xpos[0][1] **2 < site_xpos[1][0] **2 + site_xpos[1][1] **2:
                return rewards.tolerance(
                bucket_ori_z, bounds, margin, sigmoid='long_tail', value_at_margin=v)
            else:
                return 0
        elif  flag == 'deep':
            if site_xpos[0][0] **2 + site_xpos[0][1] **2 > site_xpos[1][0] **2 + site_xpos[1][1] **2:
                return rewards.tolerance(bucket_ori_z, bounds, margin, sigmoid='hyperbolic', value_at_margin=v)
            else:
                return 0
        elif flag == 'with_soil':
            return rewards.tolerance(bucket_ori_z, bounds, margin, sigmoid='hyperbolic', value_at_margin=v)
        
        
    def max_flow_rate(total_flow_rate, bounds, margin, v):
        return (rewards.tolerance(total_flow_rate, bounds, margin, sigmoid='hyperbolic', value_at_margin=v) - 1)
    
    def explore_boundary(attach_height, bounds, margin, v):
        return (rewards.tolerance(attach_height, bounds, margin, sigmoid='hyperbolic', value_at_margin=v) - 1)