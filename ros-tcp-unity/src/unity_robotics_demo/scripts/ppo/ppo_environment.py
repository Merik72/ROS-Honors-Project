#!/usr/bin/env python
# license removed for brevity
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

import random
import rospy
from std_msgs.msg import String

from geometry_msgs.msg import Twist  

#for general commands as reset cmd
from std_msgs.msg import String, Float32 , Bool
 

# environment.py:

# Encapsulates interaction with the robot (simulated or real).
# Processes RGB camera input, handles action outputs, and returns next states, rewards, and termination flags.

class DingoEnv:
    def __init__(self, image_topic="/image_raw0",resolution=(512, 512) , no_channels = 3 , action_space = 2 , tau = 0.02):
        self.image_topic = image_topic
        self.resolution = resolution 
        self.bridge = CvBridge()
        self.current_image = None

        #Size of image from observation
        self.observation_space =  resolution[0] * resolution[1] * no_channels
        self.action_space  = action_space

        # time between sending commands
        self.tau = tau # 0.02  # Time interval between state updates
 
        self.velocity_topic = "/cmd_vel"     # moving
        self.control_topic = "/control_cmd"  # resetting
        self.reward_topic ="/reward"
        self.done_topic ="/done_status"

        self.reward = None 
        self.done_status = None
        self.v_left = 0.0
        self.vright = 0.0
      
        rospy.init_node('dingo_environment', anonymous=True)
        rospy.Subscriber(self.image_topic, Image, self.image_callback)

        #for the done status        
        rospy.Subscriber(self.done_topic, Bool, self.donestatus_callback)
        
      
    #   for velocity publish
        self.pub = rospy.Publisher(self.velocity_topic, Twist, queue_size=10)
        rospy.loginfo("DingoEnv initialized and ready to publish velocities.")

    #  for command reset  = 1 is reset
        self.command_pub =  rospy.Publisher(self.control_topic, String, queue_size=10)
 

    # for reward  
        rospy.Subscriber(self.reward_topic, Float32, self.reward_callback)

    def reward_callback(self, rwd):
        """
        Callback to process incoming camera images from the ROS topic.

        Args:
            msg (sensor_msgs.msg.Image): ROS image message.
        """
        self.reward = rwd.data
        # rospy.logerr(f"Current reward: {self.reward}")
        return self.reward
    
    def donestatus_callback(self, _done):
        """
        Callback to process incoming camera images from the ROS topic.

        Args:
            msg (sensor_msgs.msg.Image): ROS image message.
        """
        self.done_status = _done.data
        # rospy.logerr(f"Current reward: {self.reward}")
        return self.done_status


    def image_callback(self, msg):
        """
        Callback to process incoming camera images from the ROS topic.

        Args:
            msg (sensor_msgs.msg.Image): ROS image message.
        """
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            cv_image = cv2.flip(cv_image, 0)  # Flip image vertically 

            # Resize image to the desired resolution
            self.current_image = cv2.resize(cv_image, self.resolution)
            # print(self.current_image)
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")
    def get_observation(self):
        """
        Returns the current observation from the environment.

        Returns:
            np.ndarray: Processed RGB image as a numpy array.
        """
        if self.current_image is not None:
            return self.current_image
        else:
            rospy.logwarn("No image received yet. Returning empty array.")
            return np.zeros((*self.resolution, 3), dtype=np.uint8)
        

    def reset(self):
        """
        Resets the environment for a new episode.

        Returns:
            np.ndarray: Initial observation.
        """
        # rospy.loginfo("Resetting environment...")
        for _ in range(3):
            self.command_pub.publish("reset")

            # rospy.sleep(1)  # Allow some time for the first image to be captured
        return self.get_observation()

    def step(self, action):
        """
        Takes an action and returns the next observation, reward, and done flag.

        Args:
            action (tuple): Desired velocities for the robot's wheels (left, right).

        Returns:
            tuple: (next_observation, reward, done)
        """
        self.send_velocity(action)

        # rospy.loginfo("Just send command")

        rospy.sleep(self.tau)  # Wait for the specified time interval

        next_observation = self.get_observation()
        reward = self.reward # self.calculate_reward(next_observation)
        done = self.check_goal_reached()  #next_observation)
        return next_observation, reward, done

    def send_velocity(self, action):
        """
        Sends velocity commands to the robot's wheels.

        Args:
            action (tuple): Desired velocities for the robot's wheels (left, right).
        """
        # Add code to publish wheel velocity commands via ROS topic
        # rospy.loginfo(f"Action taken: {action}")  
            
        vel = self.wheel_to_twist( v_left=action[0], v_right= action[1]) 
        
        # text_to_publish = "Hello from ROS!"
        # rospy.loginfo(vel)
        self.pub.publish(vel) 

    def wheel_to_twist(self, v_left=0.0, v_right = 0.0 ):
        # Compute linear and angular velocities 
        v =     (v_left  + v_right ) / 2.0
        omega = (v_right - v_left) / 0.48 #unit: metter  0.033  #

        # Create Twist message
        twist_msg = Twist()
        twist_msg.linear.x = v
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = omega

        # print(f"V: {v} | omega: {omega}")
        return twist_msg
    
    # def calculate_reward(self, observation):
    #     """
    #     Calculates the reward based on the current observation.
    #     Args:
    #         observation (np.ndarray): Current observation.
    #     Returns:
    #         float: Reward value.
    #     """
    #     # Example: Distance to goal using visual processing
    #     rospy.loginfo("Calculating reward...")
    #     return 0.0  # Placeholder logic

    def check_goal_reached(self):  #, observation):
        """
        Checks if the goal is reached.

        Args:
            observation (np.ndarray): Current observation.

        Returns:
            bool: True if the goal is reached, False otherwise.
        """ 
        if self.done_status is not None:
            return self.done_status
        else:
            rospy.logwarn("No Status received yet. Returning None object.")
            return None 
     
if __name__ == "__main__":
    env = DingoEnv()
    rospy.loginfo("Environment is running. Press Ctrl+C to exit.")

    env.reset()
    print("Just reset")
    time.sleep(1)

    try:
        while not rospy.is_shutdown():
            obs = env.get_observation()
            if obs is not None:
                cv2.imshow("DingoEnv Observation", obs)
                cv2.waitKey(1)
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down environment...")
    finally:
        cv2.destroyAllWindows()

 