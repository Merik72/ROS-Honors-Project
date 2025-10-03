#!/usr/bin/env python3

# Import libraries for getting the data
import rospy
from sensor_msgs.msg import PointCloud2 , PointField
from std_msgs.msg import Float32MultiArray
import numpy as np

import pandas as pd
from sklearn.preprocessing import Normalizer

lidar_data = None 
iter = 0
df_lidar = None

def pointcloud2_to_array(cloud_msg):
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
    return points_array 
 
def callback(msg):
    global lidar_data 
    lidar_data = pointcloud2_to_array(msg)[:,:3] 
    pub = rospy.Publisher("lidar_as_pc2", PointCloud2, queue_size=10)
    pub.publish(lidar_data)
    
def listener(): 
    global lidar_data 
    rospy.init_node('lidar_listener', anonymous=True)
    rospy.Subscriber("/mid/points", PointCloud2, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()