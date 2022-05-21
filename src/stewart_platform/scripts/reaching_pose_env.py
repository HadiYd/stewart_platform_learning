#! /usr/bin/env python

import rospy
from gym import spaces

from gym.envs.registration import register
import numpy as np

import stewart_env
import math

max_episode_steps = 200 # Can be any Value 

register(
        id='StewartPose-v0',
        entry_point='reaching_pose_env:PoseSetEnv',
        max_episode_steps=max_episode_steps,
    )


class PoseSetEnv(stewart_env.StewartEnv):
    def __init__(self):

        print ("Entered stewart pose Env")
        
        self.get_params()          

        self.action_space =  spaces.Box( low   = np.array([ self.p_range[0], self.i_range[0], self.d_range[0]]),
                                         high  = np.array([ self.p_range[1], self.i_range[1], self.d_range[1]]), 
                                         dtype = np.float32)

        self.observation_space = spaces.Box(low  = np.array([self.x_range[0], self.y_range[0], self.z_range[0], -self.max_roll_angle, -self.max_pitch_angle, -self.min_yaw_angle_diff]),
                                            high = np.array([self.x_range[1], self.y_range[1], self.z_range[1],  self.max_roll_angle,  self.max_pitch_angle,  self.min_yaw_angle_diff ]), 
                                            dtype=np.float32)

        super(PoseSetEnv, self).__init__()

        # Simple reaching Task definition:
        self.ee_goal_pose = self.move_end_effector(self.reach_x,
                                                   self.reach_y,
                                                   self.reach_z,
                                                   self.reach_roll,
                                                   self.reach_pitch,
                                                   self.reach_yaw)
        
        print("The Goal is to reach: ", self.ee_goal_pose)
        

    def get_params(self):
        """
        get configuration parameters

        """

        # reaching task parameters ( we want the end effector to reach the below pose as a task )
        self.reach_x = rospy.get_param("/reaching_task/reach_x")
        self.reach_y = rospy.get_param("/reaching_task/reach_y")
        self.reach_z = rospy.get_param("/reaching_task/reach_z")
        self.reach_roll = math.radians(rospy.get_param('/reaching_task/reach_roll'))
        self.reach_pitch = math.radians(rospy.get_param('/reaching_task/reach_pitch'))
        self.reach_yaw = math.radians(rospy.get_param('/reaching_task/reach_yaw'))
   

        # get PID value ranges - Action space range
        self.p_range = rospy.get_param("/stewart/p_range_value")
        self.i_range = rospy.get_param("/stewart/i_range_value")
        self.d_range = rospy.get_param("/stewart/d_range_value")



        # get the observation spaces limit values and task goals values 
        self.x_range                = rospy.get_param('/stewart/x_range')
        self.y_range                = rospy.get_param('/stewart/y_range')
        self.z_range                = rospy.get_param('/stewart/z_range')

        self.max_roll_angle       = math.radians(rospy.get_param('/stewart/max_roll_angle'))
        self.max_pitch_angle      = math.radians(rospy.get_param('/stewart/max_pitch_angle'))
        self.max_yaw_angle        = math.radians(rospy.get_param('/stewart/max_yaw_angle'))

        self.max_speed_end_eff    = rospy.get_param('/stewart/max_speed_end_eff')
        self.max_dis_from_goal    = rospy.get_param('/stewart/max_dis_from_goal')
        self.min_dis_from_goal    = rospy.get_param('/stewart/min_dis_from_goal')

        self.min_pitch_angle_diff = math.radians(rospy.get_param('/stewart/min_pitch_angle_diff'))
        self.min_roll_angle_diff  = math.radians(rospy.get_param('/stewart/min_roll_angle_diff'))
        self.min_yaw_angle_diff   = math.radians(rospy.get_param('/stewart/min_yaw_angle_diff'))

        self.min_speed_to_settle   = rospy.get_param('/stewart/min_speed_to_settle')

        # get the reward values
        self.reach_goal_reward    = rospy.get_param('/stewart/reach_goal_reward')
        self.go_very_far_reward   = rospy.get_param('/stewart/go_very_far_reward')
        self.immobility_anxiety   = rospy.get_param('/stewart/immobility_anxiety')
        self.very_far_quadratic   = rospy.get_param('/stewart/very_far_quadratic')
        
    

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        # self.set_pid_values(1000,0.01,100)
        # self.set_poistion_joints(self.ini_joints)
        # self.set_pid_values(self.proper, self.integr, self.deriv)
        return True



    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        """
        rospy.logdebug("Init Env Variables...")
        self.cumulated_reward = 0.0
        rospy.logdebug("Init Env Variables...END")


    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        self.p =      action[0] 
        self.integr = action[1]  
        self.deriv =  action[2] 

        self.set_pid_values(self.p, self.integr, self.deriv )


    def _get_obs(self):
        """
        It returns the Position and Orientaion of the EndEffector as observation.

        Also we could consider the velocity of the endEffector coming from Twist!

        """
        self.gazebo.unpauseSim()
        
        self.ee_pose  = self.get_end_effector_pose()  # get the pose of the end effector
        self.ee_twist = self.get_end_effector_twist() # get the speed of the end effector
        
        # avrage speed of the end effector
        self.speed_of_end_eff = 1/3 *(abs(self.ee_twist.linear.x) +  abs(self.ee_twist.linear.y) + abs(self.ee_twist.linear.z))
                
        # Distance between current position and goal 
        self.dist_to_goal = np.sqrt( (self.ee_goal_pose.linear.x - self.ee_pose.linear.x)**2 +
                                     (self.ee_goal_pose.linear.y - self.ee_pose.linear.y)**2 +
                                     (self.ee_goal_pose.linear.z - self.ee_pose.linear.z)**2 )
        
        # difference between paramteres of current and goal oriantation
        self.roll_diff   =  self.ee_goal_pose.angular.x - self.ee_pose.angular.x 
        self.pitch_diff  =  self.ee_goal_pose.angular.y - self.ee_pose.angular.y
        self.yaw_diff    =  self.ee_goal_pose.angular.z - self.ee_pose.angular.z



        pose_observation = [ 
                    self.ee_pose.linear.x,
                    self.ee_pose.linear.y,
                    self.ee_pose.linear.z,
                    self.ee_pose.angular.x,
                    self.ee_pose.angular.y,
                    self.ee_pose.angular.z
                        ]


        self.pose_diff = [ 
                            self.dist_to_goal, 
                            self.roll_diff,
                            self.pitch_diff,
                            self.yaw_diff

                             ]

        # rospy.logdebug("Observations==>"+str(pose_observation))

        return  pose_observation
    

    
    def get_elapsed_time(self):
        """
        Returns the elapsed time since the beginning of the simulation
        Then maintains the current time as "previous time" to calculate the elapsed time again
        """
        current_time = rospy.get_time()
        dt = self.sim_time - current_time
        self.sim_time = current_time
        return dt

    def _is_done(self, observations):
        """
        If the latest Action didnt succeed, it means that tha position asked was imposible therefore the episode must end.
        It will also end if it reaches its goal.
        """

        # Did the movement fail in the set action?
        self.done_fail = self.dist_to_goal > self.max_dis_from_goal  # 1.3

        # if the end effector reached only the goal position
        done_xyz_dis   = self.dist_to_goal <= self.min_dis_from_goal  # 0.05

        # if all three oriantation values are reached
        done_roll  = abs(self.roll_diff )  <  self.min_roll_angle_diff  # 5 degree
        done_pitch = abs(self.pitch_diff) <  self.min_pitch_angle_diff
        done_yaw   = abs(self.yaw_diff)   <  self.min_yaw_angle_diff

        # and if the speed of the end effector is close to zero (NO oscillation)
        done_speed = self.speed_of_end_eff < self.min_speed_to_settle # speed below 0.4 -> sttlement

        
        # if all goals are satisfied
        self.done_sucess = (done_xyz_dis and done_roll and done_pitch and done_yaw and done_speed)

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>      done_fail="+str(self.done_fail)+", done_sucess="+str(self.done_sucess))

        # done anyway : 
        done = self.done_fail or self.done_sucess

        return done

    ################## Qualitative REWARD #############################
    ### In case you do not want to use quadratic reward function, you may use the below custom reward function.

    # def _compute_reward(self, observations, done):
    #     """
    #     Reward moving to the goal
    #     Punish movint to unreachable positions
    #     """ 

    #     if self.done_fail:
    #         reward = self.go_very_far_reward  # We punish if it goes far off the goal
            
    #     elif self.done_sucess:
    #         reward = self.reach_goal_reward  # If it reaches all 4 goals

    #     elif  self.min_dis_from_goal < self.distance_to_goal:  # if it doesnt move we introduce some immobility punishment
    #         reward = self.immobility_anxiety 

    #     else:
    #         reward = 0.0001

    #     return reward

    ################## QUADRETIC REWARD #############################

    def _compute_reward(self, observations, done):
        """
        Quadratic reward   r = - s*Q*sT
        """ 
        
        # get only the diff values of observation matrix
        S = np.array(self.pose_diff)
        Q = np.diagflat([[1,1], [1,1]])      #  Q = np.diagflat([[10,10], [10,10]]) 


        reward =  - S @ Q @ np.transpose(S) 

        if self.done_fail:
            reward = self.very_far_quadratic  # We punish very very much if it goes far off the goal
            

        return reward


