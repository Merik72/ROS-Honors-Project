#!/usr/bin/env python
import rospy
import time
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
import random
from unity_robotics_demo_msgs.msg import UnityColor
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32, Bool
import numpy as np

WHEEL_TOPIC = 'TTTTT'
RESET_TOPIC = 'color'
LIDAR_TOPIC = '/mid/points'
NODE_NAME = 'multi_read_listen'
lidar_data = None
received_reward = None
received_termination = None

def post_reset():
    pub = rospy.Publisher(RESET_TOPIC, UnityColor, queue_size=10)

    color = UnityColor(0, 0, 0, 1)

    #wait_for_connections(pub, TOPIC_NAME)
    pub.publish(color)
    time.sleep(1)
    pub.publish(color)
    time.sleep(1)

def post_wheels(linear=None, angular=None):
    if linear==None:
        pub = rospy.Publisher(WHEEL_TOPIC, Twist, queue_size=10)
        x = random.uniform(-1,1)
        z = random.uniform(-1,1)
        linear = Vector3(x, 0, 0)
        angular = Vector3(0, 0, z)
        twist = Twist(linear, angular)


        pub.publish(twist)
        time.sleep(0.1)
    else:
        pub = rospy.Publisher(WHEEL_TOPIC, Twist, queue_size=10)
        pub.publish(Twist(Vector3(linear,0,0), Vector3(0,0,angular)))
        time.sleep(0.1)


def hear_lidar():
    rospy.Subscriber(LIDAR_TOPIC, PointCloud2, callback_lidar)
    
def callback_lidar(cloud_msg):
    global lidar_data
    # Helper function to convert PointCloud2 message to numpy array
    dtype_list = []
    for field in cloud_msg.fields:
        if field.datatype == PointField.FLOAT32:
            dtype_list.append((field.name, np.float32))
        elif field.datatype == PointField.UINT16:
            dtype_list.append((field.name, np.uint16))
        # Add other field data types if needed
    # Create numpy array from the point cloud data
    cloud_arr = np.frombuffer(cloud_msg.data, dtype=np.dtype(dtype_list))
    x = cloud_arr['x']
    y = cloud_arr['y']
    z = cloud_arr['z']
    intensity = cloud_arr['intensity']
    ring = cloud_arr['ring']
    time = cloud_arr['time']    

    # Stack columns to form an (n, 6) array
    points_array = np.column_stack((x, y, z, intensity, ring, time))    
    lidar_data = points_array[:,:3]

def hear_reward():
    rospy.Subscriber('/reward', Float32, callback_reward)

def callback_reward(msg):
    global received_reward
    received_reward = msg.data

def hear_termination():
    rospy.Subscriber('/terminated', Bool, callback_termination)

def callback_termination(msg):
    global received_termination
    received_termination = msg.data

def run():
    global lidar_data
    global received_reward
    global received_termination
    while 1:
        try: 
            #print("Lidar:\t" + lidar_data)
            #print("Reward:\t" + received_reward)
            #print("Termination:\t" + received_reward)
            
            hear_lidar()
            hear_reward()
            hear_termination()
        except rospy.ROSInterruptException:
            print("User break.")
            break

if __name__ == '__main__':
    rospy.init_node(NODE_NAME, anonymous=True, disable_signals=True)
    run()
    