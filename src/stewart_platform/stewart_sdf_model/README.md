# Model Stewart Platform in SDF format

ROS uses URDF format to represent a robot. However, URDF uses a tree structure which does not allow closed loops structures like a Stewart platform. 

However, using SDFormat which uses a graph structure you can represent closed kinematics.


Generaly, we use macro based approach to write more easy to read type of URDFs. In a way that we first write robot.xacro and then we use xacro tool to parse and build robot.urdf file. 

However, we don't have xacro tool to write such sdf macro files. One solution is using embedded ruby (erb). We can encode robots parameters in an embedded ruby template file. But most python users may find it hard. Therefore, I wrote a python class to generate paramteric SDF files, specifically for the stewart platform!
