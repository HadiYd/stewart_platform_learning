import math
from tkinter import font
import matplotlib.pyplot as plt

def balls_link_pose(radi, theta, z , pitch):
    radi -= 0.1* radi
    balls_link_pose = {}
    for i in [1,3,5]:
        lambda_i = (i*math.pi)/3 - math.radians(theta)/2
        balls_link_pose[f'ball{i}_link_pose']   = [radi* math.cos(lambda_i),      radi* math.sin(lambda_i),      z, 0 , pitch, 0]
        lambda_i_plus = lambda_i + math.radians(theta)
        balls_link_pose[f'ball{i+1}_link_pose'] = [radi* math.cos(lambda_i_plus), radi* math.sin(lambda_i_plus), z, 0 , pitch, 0]
        
    for i in range(1,7):
        yaw = math.atan2(balls_link_pose[f'ball{i}_link_pose'][1],balls_link_pose[f'ball{i}_link_pose'][0])
        balls_link_pose[f'ball{i}_link_pose'][5] = yaw
    
    balls_link_pose_str = {k: ' '.join(map(str, list(v))) for (k,v) in balls_link_pose.items()} 

    return balls_link_pose , balls_link_pose_str

        
def plot_attachment_points(radi,theta):
    radi -= 0.1* radi
    
    balls_link_pose = {}
    for i in [1,3,5]:
        lambda_i = (i*math.pi)/3 - math.radians(theta)/2
        balls_link_pose[f'ball{i}_link_pose']   = [radi* math.cos(lambda_i),      radi* math.sin(lambda_i)]
        lambda_i_plus = lambda_i + math.radians(theta)
        balls_link_pose[f'ball{i+1}_link_pose'] = [radi* math.cos(lambda_i_plus), radi* math.sin(lambda_i_plus)]

    x_values = [item[0] for item in list(balls_link_pose.values())]
    y_values = [item[1] for item in list(balls_link_pose.values())]

    # piston_link_pose()
    
    circle=plt.Circle((0,0),BASE_RADIUS, color = "grey",fill=False)
    plt.gca().add_patch(circle)

    plt.scatter(x_values,y_values)
    # Loop for annotation of all points
    for i in range(len(x_values)):
        plt.annotate(f"point_{i+1}", (x_values[i], y_values[i] + 0.2),size=15)
    #plt.text(-1,0,f"Angle between two points = {Tetha_angle} deg")
    lim_limit = 1.2 * radi
    plt.xlim(-lim_limit,lim_limit)
    plt.xlabel("X axis")
    plt.ylabel("Y axis")

    plt.ylim(-lim_limit,lim_limit)


def piston_link_pose(radi_b,radi_p,theta_b,theta_p, base_platform_distance, base_height=0.125, ball_radius=0.1, platform_balls_radius=0.05):

    base_balls_link_poses ,_= balls_link_pose(radi_b, theta_b,0,0)
    platform_balls_link_poses,_ = balls_link_pose(radi_p, theta_p,0,0)

    radi_b -= 0.1*radi_b
    radi_p -= 0.1*radi_p


    pistons_link_pose = {}

    for i in range(1,7):

        x_p_i, y_p_i = platform_balls_link_poses[f"ball{i}_link_pose"][0] , platform_balls_link_poses[f"ball{i}_link_pose"][1]
        x_b_i, y_b_i = base_balls_link_poses[f"ball{i}_link_pose"][0]     , base_balls_link_poses[f"ball{i}_link_pose"][1]
        
        piston_length_i_proj = math.sqrt((x_p_i - x_b_i)**2 + (y_p_i - y_b_i)**2)
        yaw_piston_i = math.atan2(y_p_i - y_b_i,x_p_i - x_b_i)

        piston_length_i = math.sqrt( piston_length_i_proj**2 + base_platform_distance**2)
        pitch_piston_i  = math.radians(90) - math.atan2(base_platform_distance,piston_length_i_proj)  

        pistons_link_pose[f'piston{i}_link_pose'] = [1/2*(x_p_i+x_b_i), 1/2*(y_p_i+y_b_i), base_height+ ball_radius + base_platform_distance/2, 0 , pitch_piston_i , yaw_piston_i ]  
    
    pistons_link_pose = {k: ' '.join(map(str, list(v))) for (k,v) in pistons_link_pose.items()}


    return pistons_link_pose , piston_length_i




def attachment_points_position(radi, theta, height):
    radi -= 0.1* radi
    balls_link_pose = {}

    for i in [1,3,5]:
        lambda_i = (i*math.pi)/3 - math.radians(theta)/2
        balls_link_pose[f'ball{i}_link_pose']   = [radi* math.cos(lambda_i),      radi* math.sin(lambda_i),      height]
        lambda_i_plus = lambda_i + math.radians(theta)
        balls_link_pose[f'ball{i+1}_link_pose'] = [radi* math.cos(lambda_i_plus), radi* math.sin(lambda_i_plus), height]
        

    return list(balls_link_pose.values())



if __name__=="__main__":

    Tetha_angle = 60
    BASE_RADIUS = 2
    height = 2
    pitch_angle = 80


    platform_radius = 1.5
    teta_plat = 30


    dist_plat_base = 2

    base_balls_link_pose, base_balls_link_poses_str = balls_link_pose(BASE_RADIUS,Tetha_angle,height, 0) 
    plat_balls_link_pose, plat_balls_link_poses_str = balls_link_pose(platform_radius,teta_plat,0, 0) 


    x_base_1 = base_balls_link_pose['ball1_link_pose'][0]
    y_base_1 = base_balls_link_pose['ball1_link_pose'][1]

    x_platform_1 = plat_balls_link_pose['ball1_link_pose'][0]
    y_platform_1 = plat_balls_link_pose['ball1_link_pose'][1]


    piston_link_poses , pis_leng = piston_link_pose(BASE_RADIUS,platform_radius, Tetha_angle,teta_plat,dist_plat_base)
    
    fig= plt.figure(figsize=(10,9))
    plot_attachment_points(platform_radius,teta_plat)

    plot_attachment_points(BASE_RADIUS,Tetha_angle)
    plt.text(-0.25,-0.25,f"platform_points_angle = {teta_plat} deg",color='darkblue',size=15)
    plt.text(-0.25,0.25,f"Base_points_angle = {Tetha_angle} deg",color='orange',size=15)
    plt.scatter([0],[0],color="r")
    # plt.plot([x_base_1,x_platform_1],[y_base_1,y_platform_1],'k-')
    for i in range(1,7):
        plt.plot([base_balls_link_pose[f'ball{i}_link_pose'][0],plat_balls_link_pose[f'ball{i}_link_pose'][0]],
        [base_balls_link_pose[f'ball{i}_link_pose'][1],plat_balls_link_pose[f'ball{i}_link_pose'][1]],'k-')

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("X (m)", fontsize=15)
    plt.ylabel("Y (m)", fontsize=15)

    circle_1=plt.Circle((0,0),platform_radius, color = "grey",fill=False)
    plt.gca().add_patch(circle_1)
    plt.tight_layout()
    # plt.savefig("attachment_points.pdf",dpi=200)
    plt.show()
    print("base_balls_link_pose: \n",[ i[:2] + [1] for i in list(base_balls_link_pose.values())])

    final_atach = attachment_points_position(BASE_RADIUS,Tetha_angle,height)

    print("final_atach: \n",final_atach)

