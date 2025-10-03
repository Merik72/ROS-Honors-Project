#!/usr/bin/env python3

# Import libraries for getting the data
import rospy
from sensor_msgs.msg import PointCloud2 , PointField
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import time
import socket
import open3d as o3d

# Import libraries for doing machine learning
import pandas as pd # needed
from sklearn.preprocessing import Normalizer # preprocessing
# neural networks imports from tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
#from multiprocessing import Process, Queue, Pipe
#xfrom mp1 import f

# Global variable to store the latest LiDAR data
#global lidar_data
lidar_data = None 

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
    intensity = cloud_arr['intensity']
    ring = cloud_arr['ring']
    time = cloud_arr['time']    

    # Stack columns to form an (n, 6) array
    points_array = np.column_stack((x, y, z, intensity, ring, time))    
    return points_array 

def send_data(lidar):
    filepath = "/home/admin2/ros-tcp-unity/src/unity_robotics_demo/scripts/data/test.csv"
    np.savetxt(filepath, lidar, delimiter=',')  
 
def callback(msg):
    global lidar_data 
    lidar_data = pointcloud2_to_array(msg)[:,:3]  
    # send_data(lidar_data)
    # learn(lidar_data)
    # pub = rospy.Publisher("lidar_as_pc2", PointCloud2, queue_size=10)
    # pub.publish(pc2(lidar_data))
    
def listener(): 
    global lidar_data 
    rospy.init_node('lidar_listener', anonymous=True)
    #rospy.init_node("pc2_publisher")
    rospy.Subscriber("/mid/points", PointCloud2, callback)
    rospy.spin()

def learn(data):
    lidar_df = pd.Dataframe(data[:,:3], columns=['x','y','z'])


if __name__ == '__main__':
    np.load(filepath, lidar, delimiter=',')
    listener()
    #parent_conn, child_conn = Pipe()
    #p = Process(target=f, args=(child_conn,))
    #print(parent_conn.recv())