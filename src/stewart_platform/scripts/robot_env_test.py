#!/usr/bin/env python
import rospy
from stewart_env import StewartEnv
import math


def callback_1(event):
    pass
    # move_end = hadi_stewart.move_end_effector(x,y,z,roll, pitch,yaw)

if __name__ == '__main__':
    rospy.init_node("stewart_robot_env_test")

    hadi_stewart = StewartEnv()
    print("stewart object env created !")
    # rate = rospy.Rate(1)

    # x, y ,z , roll, pitch , yaw = 0,0,1,0,0,0
    # rospy.Timer(rospy.Duration(1/10), callback_1)

    # Z change
    while not rospy.is_shutdown():
        pass
        # print("Sleep 1")
        # rate.sleep()
        # z=0.5
        # print("Sleep 2")
        # rate.sleep()
        # z=1
        
        # print("Sleep 3")
        
        # rate.sleep()
        # z=1.5
        # print("Sleep 4")
        # rate.sleep()
        # print("Sleep pitch 1")
        # rate.sleep()
        # pitch = math.radians(30)
        # print("Sleep pitch 2")
        # rate.sleep()
        # pitch = math.radians(-30)
        # print("Sleep pitch 3")
        # rate.sleep()
        # pitch = math.radians(-15)
        # print("Sleep pitch 4")
        # rate.sleep()
        # pitch = math.radians(15)
        # print("Sleep pitch 5")
        # rate.sleep()
        # pitch = 0
    
