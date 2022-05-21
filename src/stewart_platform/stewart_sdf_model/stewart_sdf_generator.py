
from sdf_generator import CreateRobotSDF
import math
from piston_balls_pose import balls_link_pose,  piston_link_pose

# Define base platform parameters
base_height = 0.125
base_radius = 2.0   
base_mass = 1000.0

# Define top platform parameters
platform_height = 0.1  
platform_radius =  0.8*base_radius 
platform_mass = 0.1 

# Define top and bottom balls parameters 
ball_radius = 0.1
platform_balls_radius = 0.05

# Define attachment angles between balls of base and moiving platform
attachment_angle_top = 30   
attachment_angle_bottom = 60

# Define distance between base and moving platform
base_platform_distance = 2
base_plat_dis = base_platform_distance  - 0.5*base_height - 0.5*platform_height - platform_balls_radius -ball_radius



# Now define piston (cylinder and shaft) radius , length and pose based on the above parameters
piston_radius = 0.5*ball_radius
piston_cylinder_link_pose , piston_length = piston_link_pose(base_radius, platform_radius,attachment_angle_bottom,attachment_angle_top,base_plat_dis,ball_radius) 



"""
Initialize stewart platform model and add control plugin 
"""
# Creat model object 
stewart_model = CreateRobotSDF()
# add plugin
stewart_model.add_plugin("joint_controller", "libjoint_controller.so")

"""
First, Define all Joints:

1- prismatic joints - 6 numbers - type = prismatic
2- bottom ball joints - 6 numbers - type = universal
3- piston bottom pitch joints - 6 numbers - type = revolute
4- top ball joints - 6 numbers -type = revolute2
5- piston top pitch joints - 6 numbers - type = revolute

"""
# define prismatic joints , total number 6 legs
p_p_joint_vel_limit = str(1)
p_p_joint_eff_limit = str(500)
p_p_joint_day_damping = str(3)
p_p_joint_axis_lower_limit = "0.1"
p_p_joint_axis_upper_limit = str(0.8*piston_length)

for i in range(1,7):
    stewart_model.add_joint(f"piston{i}_prismatic_joint",'prismatic' ,f"piston{i}_cylinder_link", f"piston{i}_shaft_link", pose="0 0 0 0 0 0", axis_xyz="0 0 1", axis_limit_lower_param=p_p_joint_axis_lower_limit,axis_limit_upper_param=p_p_joint_axis_upper_limit,axis_limit_velocity_param=p_p_joint_vel_limit,axis_limit_effort_param=p_p_joint_eff_limit,axis_dynamics_damping_param=p_p_joint_day_damping)

# define all joints movement limit 
lower_limit_angle = str(math.radians(-30)) # -30 degree
upper_limit_angle = str(math.radians(30))  # 30 degree
velocity_limit_on_balls = str(1)


# define bottom ball joints
for i in range(1,7):
    stewart_model.add_joint(f"bottom_ball{i}_joint","revolute","base_link",f"bottom_ball{i}_link",pose="0 0 0 0 0 0", axis_xyz="0 1 0", axis_limit_lower_param=lower_limit_angle,axis_limit_upper_param=upper_limit_angle,axis_limit_velocity_param=velocity_limit_on_balls)

# define piston bottom pitch joints
piston_bottom_pose = "0 0 "+str(-0.5*piston_length)+" 0 0 0"

for i in range(1,7):
    stewart_model.add_joint(f"piston{i}_bottom_pitch_joint","revolute2",f"bottom_ball{i}_link",f"piston{i}_cylinder_link",pose=piston_bottom_pose, axis_xyz="1 0 0", axis_limit_lower_param=lower_limit_angle,axis_limit_upper_param=upper_limit_angle)


# define top ball joints
for i in range(1,7):
    stewart_model.add_joint(f"top_ball{i}_joint","revolute","platform_link",f"top_ball{i}_link",pose="0 0 0 0 0 0", axis_xyz="0 1 0", axis_limit_lower_param=lower_limit_angle,axis_limit_upper_param=upper_limit_angle,axis_limit_velocity_param=velocity_limit_on_balls,axis_name_2="axis2",axis_name_2_xyz="0 0 1")


# define piston top pitch joints
piston_top_pose = "0 0 " + str(0.5*piston_length) +" 0 0 0"

for i in range(1,7):
    stewart_model.add_joint(f"piston{i}_top_pitch_joint","universal",f"top_ball{i}_link",f"piston{i}_shaft_link",pose=piston_top_pose, axis_xyz="1 0 0", axis_limit_lower_param=lower_limit_angle,axis_limit_upper_param=upper_limit_angle)



"""
Second, Define all Links:

1- base link - type = cylinder
2- platform link - type = cylinder
3- bottom ball links - 6 numbers - type = sphere
4- top ball links - 6 numbers - type = sphere
5- cylinder link - 6 numbers - type = cylinder
6- shaft link - 6 numbers - type = cylinder

"""
# add base link 
base_link_pose = "0 0 " + str(0.5*base_height) +" 0 0 0"
stewart_model.add_link("base_link",base_link_pose, 'cylinder',mass=base_mass,radius= base_radius,length=base_height,material_script_uri_param="file://media/materials/scripts/gazebo.material",material_script_name_param="Gazebo/Trunk") # Trunk Farina gole

# add platform link
platform_link_pose = "0 0 " + str(0.5*base_height+base_platform_distance) +" 0 0 0"  #-0.5*platform_height
stewart_model.add_link("platform_link",platform_link_pose, 'cylinder',mass=platform_mass,radius= platform_radius,length=platform_height,material_script_uri_param="file://media/materials/scripts/gazebo.material",material_script_name_param="Gazebo/Footway")

# add bottom ball links pose
_, bottom_balls_link_pose = balls_link_pose(base_radius,attachment_angle_bottom, base_height,0)
for i in range(1,7):
    stewart_model.add_link(f"bottom_ball{i}_link",bottom_balls_link_pose[f"ball{i}_link_pose"],geometry='sphere', mass=0.01,radius=ball_radius,length=0.1,material_script_uri_param="file://media/materials/scripts/gazebo.material",material_script_name_param="Gazebo/Gold")

# add top ball links pose
top_ball_height = 0.5*base_height+base_platform_distance - platform_height
_, top_balls_link_pose = balls_link_pose(platform_radius,attachment_angle_top,top_ball_height,0)

for i in range(1,7):
    stewart_model.add_link(f"top_ball{i}_link",top_balls_link_pose[f"ball{i}_link_pose"],geometry='sphere', mass=0.01,radius=platform_balls_radius,length=0.1,material_script_uri_param="file://media/materials/scripts/gazebo.material",material_script_name_param="Gazebo/White")


for i in range(1,7):
    stewart_model.add_link(f"piston{i}_cylinder_link",piston_cylinder_link_pose[f"piston{i}_link_pose"],geometry='cylinder', mass=0.1,radius=piston_radius,length=piston_length,material_script_uri_param="file://media/materials/scripts/gazebo.material",material_script_name_param="Gazebo/DarkGrey")

# define piston shaft link pose 
# note: piston shaft pose is equal to piston cylinder link pose

for i in range(1,7):
    stewart_model.add_link(f"piston{i}_shaft_link", piston_cylinder_link_pose[f"piston{i}_link_pose"],geometry='cylinder', mass=0.1,radius=0.5*piston_radius,length=piston_length,material_script_uri_param="file://media/materials/scripts/gazebo.material",material_script_name_param="Gazebo/Black")


# finally, save the model in sdf format
stewart_model.save_model("stewart_sdf")