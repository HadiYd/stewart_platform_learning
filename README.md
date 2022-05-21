# stewart_platfrorm_learning
Set of tools and environments to implement Deep Reinforcement Learning (DRL) algorithms on Stewart Platfrom by parametric simulation in Gazebo and ROS.


## Cloning the project as a workspace
```
git clone https://github.com/HadiYd/stewart_platform_learning.git
```

## Build the controller plugin for controlling joints and changing the PID values.
> **Plugin credit by:** [ros_sdf](https://github.com/daniel-s-ingram/ros_sdf) with modification of adding PID section to the code.
```
cd src/stewart_platform/plugin
mkdir build
cd build
cmake ../
make 
```
## Installing openai_ros and building the project
```
cd stewart_platform_learning/src
git clone https://bitbucket.org/theconstructcore/openai_ros.git
cd stewart_platform_learning
catkin build
source devel/setup.bash
rosdep install openai_ros
```

## Spawn the Stewart platform in Gazebo using launch file
```
roslaunch stewart_platform stewart.launch 
```

In case of an error in the subsequent launches, kill the previous running Gazebo server by:
```
killall -9 gzserver
```

## running the deep reinforcement learning training scripts
I use wandb to log all the rewards and performance metrics. First pip install it the create a free account. 
```
pip install wandb

wandb login
```
> **DRL algorithms credit by:** [Deep Reinforcement Learning in TensorFlow2](https://github.com/marload/DeepRL-TensorFlow2)

### DDPG

**Paper** [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)<br>
**Author** Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra<br>
**Method** OFF-Policy / Temporal-Diffrence / Model-Free<br>
**Action** Continuous<br>


#### running the DDPG algorithm
```
rosrun stewart_platform DDPG_Continuous.py 
```

### A3C

**Paper** [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)<br>
**Author** Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu<br>
**Method** ON-Policy / Temporal-Diffrence / Model-Free<br>
**Action** Discrete, Continuous<br>

#### running the DDPG algorithm
```
rosrun stewart_platform A3_algorithm_training.py 
```

### PPO

**Paper** [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)<br>
**Author** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov<br>
**Method** ON-Policy / Temporal-Diffrence / Model-Free<br>
**Action** Discrete, Continuous<br>

### running the DDPG algorithm
```
rosrun stewart_platform PPO_Continuous.py 
```