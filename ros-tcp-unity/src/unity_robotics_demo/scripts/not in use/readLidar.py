#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2 , PointField
import sensor_msgs.point_cloud2 as pc2
import pandas as pd
import numpy as np

lidar_data = None
lidar_data_names = ["x", "y", "z", "intensity", "ring", "time"]

def callback(msg):
    global lidar_data
    lidar_data = msg

def listener():
    rospy.init_node('numpy_array_listener', anonymous=True)
    rospy.Subscriber("lidar_as_pc2", PointCloud2, callback)
    rospy.Rate(10).sleep(10)

def process_lidar():
    lidar_df = pd.DataFrame(lidar_data, columns = lidar_data_names)
    lidar_df.head()

if __name__ == '__main__':
    listener()
    if lidar_data is not None:
        process_lidar()
