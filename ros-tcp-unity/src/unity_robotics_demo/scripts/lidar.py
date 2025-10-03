#!/usr/bin/env python3

# Import libraries for getting the data
import rospy
from sensor_msgs.msg import PointCloud2 , PointField
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayLayout
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import time
import socket
import open3d as o3d

import pandas as pd
from sklearn.preprocessing import Normalizer

LIDAR_TOPIC = "env_0/mid/points"

# Global variable to store the latest LiDAR data
#global lidar_data
lidar_data = None 
iter = 0
# create visualizer and window.
vis = o3d.visualization.Visualizer()
vis.create_window(height=720, width=1080)

# initialize pointcloud instance.
pcd = o3d.geometry.PointCloud()
# *optionally* add initial points
points = np.random.rand(10, 3)
pcd.points = o3d.utility.Vector3dVector(points) 
# include it in the visualizer before non-blocking visualization.
vis.add_geometry(pcd) 

def update_visualization(loaded_arr):
    pcd.points = o3d.utility.Vector3dVector(loaded_arr*0.2)    #*.2 as zoom out    
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()


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
    intensity = cloud_arr['intensity'] # how much of the emitted signal is returned
    ring = cloud_arr['ring'] # which laser it comes from (as it's spinning, each laser creates a "ring" of points)
    time = cloud_arr['time'] # travel time of the laser 

    # Stack columns to form an (n, 6) array
    points_array = np.column_stack((x, y, z, intensity, ring, time))    
    return points_array 

def send_data(lidar):
    # send on a rotating thing
    global iter
    filepath = "/home/admin2/ros-tcp-unity/src/unity_robotics_demo/scripts/data/test" + str(iter) +".csv"
    np.savetxt(filepath, lidar, delimiter=',')  
    iter += 1
    if iter > 10:
        iter = 0
 
def callback(msg):
    global lidar_data 
    lidar_data = pointcloud2_to_array(msg)  
    send_data(lidar_data)
    lidar_data = lidar_data[:,:3]
    update_visualization(lidar_data) 
    # learn(lidar_data)
    # pub = rospy.Publisher("lidar_as_pc2", Float32MultiArray, queue_size=10)
    # pub.publish(normalize(lidar_data))
    # lidar_message = []
    # pub.publish(lidar_data)
    
def listener(): 
    global lidar_data 
    rospy.init_node('lidar_listener', anonymous=True)
    # rospy.init_node("pc2_publisher")
    rospy.Subscriber(LIDAR_TOPIC, PointCloud2, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
    #parent_conn, child_conn = Pipe()
    #p = Process(target=f, args=(child_conn,))
    #print(parent_conn.recv())