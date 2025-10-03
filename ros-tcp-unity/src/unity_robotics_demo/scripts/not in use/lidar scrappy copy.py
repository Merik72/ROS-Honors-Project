# Adding mapping software? 
# Perimeter = a queue of the last nodes compared against
# Point_map = a coordinate plane of equidistant 3D points
# array_to_map
#   for each point in lidar_data
#       if the lidar_data point is equidistant to a point in perimeter then 
#           add it to point_map 
#           add it to perimeter
#           pop the compared point from perimeter
# # Possible optimization: store the X and Y values of each point from point_map
# # if the lidar_data point doesn't match one, it cannot fit in the grid anyway, skipped
# # Has someone already done something like this? 

#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2 , PointField
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import time
import socket
import open3d as o3d
#from multiprocessing import Process, Queue, Pipe
#xfrom mp1 import f

# Global variable to store the latest LiDAR data
#global lidar_data
lidar_data = None 
point_map = None
perimeter = None

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

#def array_to_map(data):
#    global perimeter
 #   for entry in data:
  #      for point in perimeter:
   #         if distance(entry, point):    

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
    filepath = "/home/admin2/ros-tcp-unity/src/unity_robotics_demo/scripts/data/test.txt"
    np.savetxt(filepath, lidar, delimiter=',')  
 
def callback(msg):
    global lidar_data, point_map, perimeter
    lidar_data = pointcloud2_to_array(msg)[:,:3]
    if point_map == None:
        point_map = lidar_data(1)
        perimeter = point_map
    else:
        point_map = array_to_map(lidar_data)
    update_visualization(lidar_data) 
    # pub = rospy.Publisher("lidar_as_pc2", PointCloud2, queue_size=10)
    # pub.publish(pc2(lidar_data))
    
def loop(): 
    global lidar_data 
    rospy.init_node('lidar_listener', anonymous=True)
    #rospy.init_node("pc2_publisher")
    rospy.Subscriber("/mid/points", PointCloud2, callback)
    rospy.spin()

if __name__ == '__main__':
    loop()
    #parent_conn, child_conn = Pipe()
    #p = Process(target=f, args=(child_conn,))
    #print(parent_conn.recv())