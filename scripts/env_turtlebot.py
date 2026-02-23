#! /usr/bin/env python

# This file actually is supposed to get connect the robot to the ddpg network

import rospy
# Script becomes a ros node...need to subscibe to the scan topic to obtain the lidar scans and need to publish velocities (cmd_vel)

import rospkg
# To find the maps (file paths)

import tf
# Handles transfor,

from std_msgs.msg import String
# String: string version of Ros message

from geometry_msgs.msg import Twist, Point, Quaternion
# Twist: ros message that contains lin and ang velocities that gets published to the cmd_vel topic
# Point: ros message that contains the position of a point(x,y,z)
# Quaternion: ros message that represents an orientation in free space in quaternion form (x,y,z,w)

from math import radians, copysign, sqrt, pow, pi, atan2
from tf.transformations import euler_from_quaternion
# Import math stuff

import threading
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState 
# The threading module provides a way to run multiple threads (smaller units of a process) concurrently within a single process.
# The line from gazebo_msgs.msg import ModelStates is used to broadcast
# the current pose and twist (linear and angular velocity) of all models in the simulation world frame. 


from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import LaserScan
import time
# 1st line: It is used to change the position (pose) or velocity (twist) of a model (robot or object) 
# inside the Gazebo simulation programmatically, used to teleporting a robot to a starting position, resetting a simulation.
# 2nd line:  It allows your Python script to read data from 2D Laser Range-finders (LiDAR) in the simulation.
# allows subscribing to /scan topic to recieve data for obstacle avoidance
# 3rd line: allows to work with time


import numpy as np
import math
import random

from std_srvs.srv import Empty

# Need this class to ensure the robot Action (movement) and Observation (lidar scans) are perfectly synced.
class InfoGetter(object):
    def __init__(self):
        #event that will block until the info is received
        self._event = threading.Event()
        #attribute for storing the rx'd message
        self._msg = None

    def __call__(self, msg):
        #Uses __call__ so the object itself acts as the callback
        #save the data, trigger the event
        self._msg = msg
        self._event.set()

    def get_msg(self, timeout=None):
        """Blocks until the data is rx'd with optional timeout
        Returns the received message
        """
        self._event.wait(timeout)
        return self._msg




class GameState:

    def __init__(self):
        self.talker_node = rospy.init_node('talker', anonymous=True)
        self.pose_ig = InfoGetter()
        self.laser_ig = InfoGetter()
        self.collision_ig = InfoGetter()
        

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.position = Point()
        self.move_cmd = Twist()

        self.pose_info = rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_ig)
        self.laser_info = rospy.Subscriber("/scan", LaserScan, self.laser_ig)
        # changed the topic---such a simple thing, took too long to debug, wrong topic was the reason of going round in circles
        # self.bumper_info = rospy.Subscriber("/mobile_base/events/bumper", BumperEvent, self.processBump)


        # tf
        self.tf_listener = tf.TransformListener()
        self.odom_frame = 'odom'
        try:
            self.tf_listener.waitForTransform(self.odom_frame, 'base_footprint', rospy.Time(), rospy.Duration(1.0))
            self.base_frame = 'base_footprint'
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            try:
                self.tf_listener.waitForTransform(self.odom_frame, 'base_link', rospy.Time(), rospy.Duration(1.0))
                self.base_frame = 'base_link'
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo("Cannot find transform between odom and base_link or base_footprint")
                rospy.signal_shutdown("tf Exception")
        (self.position, self.rotation) = self.get_odom()

  
        self.rate = rospy.Rate(100) # 100hz

        # Create a Twist message and add linear x and angular z values
        self.move_cmd = Twist()
        self.move_cmd.linear.x = 0.6 #linear_x
        self.move_cmd.angular.z = 0.2 #angular_z

        # crush default value
        self.crash_indicator = 0

        # observation_space and action_space
        self.state_num = 28              
        self.action_num = 2
        self.observation_space = np.empty(self.state_num)
        self.action_space = np.empty(self.action_num)
        # self.state_input1_space =  np.empty(1)
        # self.state_input2_space =  np.empty(1)

        self.laser_reward = 0
        # set target position
        self.target_x = 6
        self.target_y = 5

        # set turtlebot index in gazebo world
        self.model_index = 10 #25

        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

    


    def get_odom(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
            rotation = euler_from_quaternion(rot)

        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return

        return (Point(*trans), rotation[2])


    def shutdown(self):
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)


    def print_odom(self):
        while True:
            (self.position, self.rotation) = self.get_odom()
            print("position is %s, %s, %s, ", self.position.x, self.position.y, self.position.z)
            print("rotation is %s, ", self.rotation)


    def reset(self):

        # for maze:
        self.target_x = np.random.uniform(-9, 9)  
        self.target_y = np.random.uniform(-9, 9)

        # Safety Check: If target is too close to center (< 1m), pick again
        while abs(self.target_x) < 1.0 and abs(self.target_y) < 1.0:
             self.target_x = np.random.uniform(-9, 9)
             self.target_y = np.random.uniform(-9, 9)

        
        '''for corridor
        self.target_x = (np.random.random()-0.5)*5 + 14*index_x
        self.target_y = (np.random.random()-0.5)*3
        random_turtlebot_y = (np.random.random())*5 #+ index_turtlebot_y
        '''


        self.crash_indicator = 0

        state_msg = ModelState()    
        state_msg.model_name = 'turtlebot3_waffle_pi'
        state_msg.pose.position.x = 0.0
        state_msg.pose.position.y = 0.0 #random_turtlebot_y
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 1

        
        # Pink ball that denotes the target in each episode
        state_target_msg = ModelState()    
        state_target_msg.model_name = 'unit_sphere_0_0' #'unit_sphere_0_0' #'unit_box_1' #'cube_20k_0'
        state_target_msg.pose.position.x = self.target_x
        state_target_msg.pose.position.y = self.target_y
        state_target_msg.pose.position.z = 0.0
        state_target_msg.pose.orientation.x = 0
        state_target_msg.pose.orientation.y = 0
        state_target_msg.pose.orientation.z = 0
        state_target_msg.pose.orientation.w = 1


        #rospy.wait_for_service('gazebo/reset_simulation')
        # try:
        #     self.reset_proxy()
        # except (rospy.ServiceException) as e:
        #     print("gazebo/reset_simulation service call failed")

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg )
            resp_target = set_state( state_target_msg )

        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
        
        self.move_cmd.linear.x = 0
        self.move_cmd.angular.z = 0
        self.pub.publish(self.move_cmd)
        
        time.sleep(0.2)

        initial_state = np.ones(self.state_num)
        #initial_state[self.state_num-2] = 0
        initial_state[self.state_num-1] = 0
        initial_state[self.state_num-2] = 0
        initial_state[self.state_num-3] = 0
        initial_state[self.state_num-4] = 0

        self.move_cmd.linear.x = 0
        self.move_cmd.angular.z = 0
        self.pub.publish(self.move_cmd)
        time.sleep(1)
        self.pub.publish(self.move_cmd)
        self.rate.sleep()


        return initial_state


    def turtlebot_is_crashed(self, laser_values, range_limit):
        self.laser_crashed_value = 0
        self.laser_crashed_reward = 0

        for i in range(len(laser_values)):
            if (laser_values[i] < 2*range_limit):
                self.laser_crashed_reward = -80
            if (laser_values[i] < range_limit):
                self.laser_crashed_value = 1
                self.laser_crashed_reward = -200
                self.reset()
                time.sleep(1)
                break
        return self.laser_crashed_reward


    def game_step(self, time_step=0.1, linear_x=0.8, angular_z=0.3):


        start_time = time.time()
        record_time = start_time
        record_time_step = 0
        self.move_cmd.linear.x = linear_x*0.26      # Scaling the linear velocity 
        self.move_cmd.angular.z = angular_z*0.5     # Scaling the angular velocity by an ideal facor of 2 (max angular speed)
                                                    # but in reality can be too fast for the lidar to process
                                                     
        self.rate.sleep()


        (self.position, self.rotation) = self.get_odom()
        turtlebot_x_previous = self.position.x
        turtlebot_y_previous = self.position.y


        while (record_time_step < time_step) and (self.crash_indicator==0):
            self.pub.publish(self.move_cmd)
            self.rate.sleep()
            record_time = time.time()
            record_time_step = record_time - start_time

        (self.position, self.rotation) = self.get_odom()
        turtlebot_x = self.position.x
        turtlebot_y = self.position.y
        angle_turtlebot = self.rotation

        # make input, angle between the turtlebot and the target
        angle_turtlebot_target = atan2(self.target_y - turtlebot_y, self.target_x- turtlebot_x)

        if angle_turtlebot < 0:
            angle_turtlebot = angle_turtlebot + 2*math.pi

        if angle_turtlebot_target < 0:
            angle_turtlebot_target = angle_turtlebot_target + 2*math.pi


        angle_diff = angle_turtlebot_target - angle_turtlebot
        if angle_diff < -math.pi:
            angle_diff = angle_diff + 2*math.pi
        if angle_diff > math.pi:
            angle_diff = angle_diff - 2*math.pi



        # prepare the normalized laser value and check if it is crash
        laser_msg = self.laser_ig.get_msg()
        laser_values = laser_msg.ranges

        # now, when lidar does not sense any object, it returns inf, which is a bad input for ddpg, this is the reason for inf reward
        # to solve this:
        normalized_laser = []
        for i in range(len(laser_values)):
            raw_val = laser_values[i]
            
            
            if raw_val == float('inf'):
                clean_val = 3.5
            elif math.isnan(raw_val):
                clean_val = 3.5         #check whether it should be 0 or 3.5
    
            # 3. Use actual value
            else:   
                clean_val = raw_val
                
            # Normalize: (0 to 3.5) becomes (0.0 to 1.0)
            normalized_laser.append(clean_val / 3.5)

        # print('turtlebot normalized laser range is %s', normalized_laser)


        # prepare state
        #state = np.append(normalized_laser, angle_diff)
        #state = np.append(normalized_laser,self.target_x- turtlebot_x)
        #state = np.append(state, self.target_y - turtlebot_y)
        current_distance_turtlebot_target = math.sqrt((self.target_x - turtlebot_x)**2 + (self.target_y - turtlebot_y)**2)

        state = np.append(normalized_laser, current_distance_turtlebot_target)
        state = np.append(state, angle_diff)
        state = np.append(state, linear_x*0.26)
        state = np.append(state, angular_z)
        # print("angle_turtlebot and angle_diff are %s %s", angle_turtlebot*180/math.pi, angle_diff*180/math.pi)
        # print("position x is %s position y is %s", turtlebot_x, turtlebot_y)
        # print("target position x is %s target position y is %s", self.target_x, self.target_y)
        # print("command angular is %s", angular_z*1.82)
        # print("command linear is %s", linear_x*0.26)
        #print("state is %s", state)

        state = state.reshape(1, self.state_num)


        # make distance reward
        (self.position, self.rotation) = self.get_odom()
        turtlebot_x = self.position.x
        turtlebot_y = self.position.y
        distance_turtlebot_target_previous = math.sqrt((self.target_x - turtlebot_x_previous)**2 + (self.target_y - turtlebot_y_previous)**2)
        distance_turtlebot_target = math.sqrt((self.target_x - turtlebot_x)**2 + (self.target_y - turtlebot_y)**2)
        
        
        
        '''
        MY REWARD FUNCTION:
        distance_reward = distance_turtlebot_target_previous - distance_turtlebot_target

        self.laser_crashed_reward = self.turtlebot_is_crashed(laser_values, range_limit=0.25)
        self.laser_reward = sum(normalized_laser)-24
        # my modification:
        # To make sure that the robot does not go out of bounds
        out_of_bounds=False
        if self.position.x > 9 or self.position.x < -9 or self.position.y > 9 or self.position.y < -9:
            out_of_bounds = True
        if out_of_bounds:
            self.laser_crashed_reward = -50  # Big penalty like a crash for -200
                                             # if penalty is lesser, around -50, then robot might explore more 
            self.reset()                      # Reset the game
            done = True
        self.collision_reward = self.laser_crashed_reward + self.laser_reward


        self.angular_punish_reward = 0
        self.linear_punish_reward = 0

        if angular_z > 0.8 or angular_z < 0.8:
            self.angular_punish_reward = -5.0 * abs(angular_z)

        if linear_x < 0.5:
            self.linear_punish_reward = -10


        self.arrive_reward = 0
        if distance_turtlebot_target<0.5:
            self.arrive_reward = 200            #if robot is within 0.5 units within the pink ball (target), it recieves a huge reward of 200 and respawns at the origin
            
            self.reset()
            time.sleep(1)
        
        # to reward the robot if its looking at the goal direction:
        facing_reward = 5.0 * (math.pi - abs(angle_diff)) / math.pi



 

        reward  = distance_reward*(5/time_step)*1.2*7 + self.arrive_reward + self.collision_reward + self.angular_punish_reward + self.linear_punish_reward+facing_reward
        # print("laser_reward is %s", self.laser_reward)
        # print("laser_crashed_reward is %s", self.laser_crashed_reward)
        # print("arrive_reward is %s", self.arrive_reward)
        # print("distance reward is : %s", distance_reward*(5/time_step)*1.2*7)


        return reward, state, self.laser_crashed_value

'''

        # GEMINI'S REWARD FUNCTION:
        # --- NEW REWARD FUNCTION START ---

        # 1. THE CARROT: Reward for getting closer (Dense Reward)
        # We multiply by 50 to make it significant (e.g., +0.5 to +2.0 per step)
        distance_reward = (distance_turtlebot_target_previous - distance_turtlebot_target) * 50

        # 2. THE GOAL: Big bonus for arriving
        self.arrive_reward = 0
        if distance_turtlebot_target < 0.5:
            self.arrive_reward = 200
            self.reset()
            done = True
            time.sleep(1)

        # 3. THE CRASH: Big penalty for hitting wall
        # (Your turtlebot_is_crashed function handles the reset logic)
        self.laser_crashed_reward = self.turtlebot_is_crashed(laser_values, range_limit=0.25)

        # 4. THE STICK: Simple spin tax
        # Punish absolute angular velocity. If it spins fast, it loses points.
        self.angular_punish_reward = -0.5 * abs(angular_z)

        # 5. THE WARNING: Small penalty for being too close to walls (Optional)
        # Instead of punishing existence, only punish if min_laser < 0.5 (Danger Zone)
        if min(normalized_laser) < (0.5 / 3.5): # 0.5 meters normalized
             self.laser_reward = -2.0
        else:
             self.laser_reward = 0.0

        # TOTAL REWARD
        # We add a small time penalty (-0.1) to encourage speed naturally, 
        # instead of forcing it with a hard threshold.
        reward = distance_reward + \
                 self.arrive_reward + \
                 self.laser_crashed_reward + \
                 self.angular_punish_reward + \
                 self.laser_reward - 0.1

        # --- NEW REWARD FUNCTION END ---
        return reward, state, self.laser_crashed_value


'''#to test whether the above block even works or not? code below is to drive the bot in circles:


if __name__ == '__main__':
    try:
        # Initialize the environment
        game_state = GameState()
        print("--- Manual Test Mode: Robot should drive in a circle ---")
        
        while True:
            # 1. Define a test action (Circle: Forward + Turn)
            test_linear_speed = 3   # 0 to 1
            test_angular_speed = 0  # -1 to 1
            
            # 2. Execute the action
            # This tests if your publishers and subscribers are working
            # Returns: reward, state, done
            reward, state, done = game_state.game_step(0.1, test_linear_speed, test_angular_speed)
            
            # 3. Print feedback
            print(f"Reward: {reward:.2f} | Crash Status: {done}")
            
            # 4. Check Odometry (Optional)
            # game_state.print_odom()
            
    except rospy.ROSInterruptException:
        pass
'''