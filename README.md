# stewart_platfrorm_learning
Set of tools and environments to implement Deep Reinforcement Learning (DRL) algorithms on Stewart Platfrom by parametric simulation in Gazebo and ROS.


## Cloning the project as a workspace
```
git clone https://github.com/HadiYd/stewart_platform_learning.git
```

## Build the controller plugin for controlling joints and changing the PID values.
> credit: https://github.com/daniel-s-ingram/ros_sdf with modification of adding PID section to the code.
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