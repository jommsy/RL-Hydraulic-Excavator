## 08.15

### Changlog

1. modify the abs block for command ---> fix logistic error
2. add dead zone in valve vector
3. modify sample time of Tm, spoolPos, loads
4. add rate limit for action, to solve transient initialization consistency problem
5. add site and modify the excavator.py 
6. again the transient consistency problem, change the rate limit from 2 to 1.5

### Bug

1. can't use `global env` for press listener ---> compare with the previous file
2. problem while the second episode
3. sim/real time test???

## 08.16

### Changlog

1. wrap the action, round to 3 digits
2. Bug 2. is because of tEnd in Simulink

## 08.21

### Changelog

1. add PS-Abs for spool pos command in Simulink
2. searching for Linux VM of **AWS**, Huawei, too expensive
3. fix the solid file path of Simulink, to promise the use in different system
4. add progressbar
5. compare the simulation speed, see `bwhpc-single.webm` and `local-run.webm` ---> **guess the biggest factor affecting sim speed is frequently I/O W/R**
```bash
# bwhpc command
$ salloc -p single -n 40 -t 24:00:00 --mem 16000
$ module load math/matlab
$ xvfb-run matlab -nodesktop -nosplash
```
TODO: test with gpu
### Bug

1. it's still possible that can't simulate 0.02s in 4s wallclock time, leading the null pointer of `pos` which may lead to the BAD_MJQACC problem

## 08.22

### Changelog

1. test with hydraulic model without sensors ---> no significantly speed increase
2. detect the solver for yesterday's bug
- try set the minimum step size as 1e-6  ---> can't work, even worse
- try dea solver ---> simulation suspend
- set the abs tolerance as 1e-3 ---> no obvious effect
3. model change
- discrete block motivates continuous block ---> exchange the position of gain and rate limiter
- many zero crossing points for max, abs block in subsystem of load ---> change the constant pressure in max load detection system as 2bar
	change the spool pos constant value in command system as 0.1mm
- **add open dynamics of 0.01s for compensator valve ---> works better for the bug**

### 

## 08.25

### Changelog

1. when test in bwcenter, the problem can't get pos appeared again --->
``` python
	if no pos in 200 times wait
		simulink return []

	if pos_ctrl == [] in all com_steps
		terminate = TRUE
```
2. pass also the vel of cylinder in simulink to python as observation
3. add `show_result.py` to show the model result,

### Problem

1. after set the judgement of termination, there is no error bug, but find the ep_mean_reward keep as 12.3 for very long
---> consider how to set the discount if terminate because of synchronization
2. consider how to use parallel process to speed up simulation
3. modify the `config.yaml` file

## 09.09
1. modify the para of bucket, whose areas are set as 2000, 2000 wrongly
2. test the computed and sensed force. The purpose of motion transfer block is synchronize the motion of 2-D simscpe and 3-D multibody. The point is to consider the friction and prevented reversed force at the end of cylinder. The sensed force is the cumulative force.

## 09.12
$$
M\dot{v} + c = \tau + J^Tf  \rightArrow M\dot{v} = \tau - (c - J^Tf)
$$
this is the dynamic equation, so the load equals $(c-J^Tf)$, so in force_ctrl, it's ok, in pos_ctrl, we should set the equality constraint point as the slider joint instead of the revolution joint between `boom_rod` and `boom` 

## 20.09
1. add solflimit and solimplimit of joints / 放大约束，使边界在 multibody 中实现
2. add force range of actuators

## 30.09
1. add friction for swing
2. improve the preload force of cylinder from 20 to 800s
3. add solref solimp condim attribute for joints in .xml
4. add range limit for joints in xml
5. increase the relief pressure to 5000 bar
6. TODO: constraint force is too huge because of the constraint computation is based on $a_ref$ instead of given force, and there is elastic force
7. TODO: the problem of synchronization of boom
8. TODO: modify the kp gain of boom position actuator

## 01.10
1. Integration Code
2. TODO: consider the viscous and contact para of bucket, ground, box and particle
3. set `self.cam.distance = 15` to modify the initial render frame
4. problem: invalid value of `cos_theorem`

## 02.10
1. change the MultiInputPolicy to SignalInputPolicy

## 04.10
1. modify the computation method of tq_load 
2. modify the consistence of rotation axis in mujoco, data.ctrl and slide axis in simulink
3. add the solimp and solref parameters for range limit in xml, contact between bucket and ground, contact between bucket and particle

## 08.10
1. problem: the velocity in simulink is very large (more than 10 m/s) when the rate of force change is very large, such as step
2. fix: the reason is that when the load change between positive load and negative load, the filler check valve can't fill the vacuums in cylinder --->
   so add transfer function for force load, and increase the max opening area (100 -> 1000), decrease the max opening pressure difference of filler check valve(1 -> 0.5)

## 09.10
1. use simulink transfer func instead od physical transfer func, to avoid the computation suspend
2. refine reward function, considering negative load and the cylinder reaches the end (which will lead to oscillation of velocity of other cylinders)
3. modify the saturation of load force to simulink according `p_max * area_section`
4. modify the position limit interval of bucket, solve the nan problem
5. modify the sign problem angular velocity of boom, note to be consistence of angular and liner velocity

## 10.10
1. modify reward function of collision
   ![old](images/image.png)
   ![new](images/image-1.png)

## 14.10 
1. there is some special values of load which will suspend the simulation (such as 1981) -> change the type of load from float32 to int32
2. maybe because the angle of mj corresponds to the cylinder position that is not consistent with the settings in sim
3. modify the init_sim_pos code, only to modify the cfg.task

## 27.10
change gym.env.mujoco.mujoco_rendering.BaseRender.scn from mujoco.MjvScene(self.model, 1000) to mujoco.MjvScene(self.model, 3000)

## 12.11

remove 'bucket_ori' from observation, add pressure difference in obs

