import open3d as o3d
import numpy as np
import time
import socket

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
# to add new points each dt secs.
dt = 0.2 

previous_t = time.time()
filepath = "/home/admin2/ros-tcp-unity/src/unity_robotics_demo/scripts/data/test.txt" 

def read_data(path):
    try:
        loaded_array = np.loadtxt(path, delimiter=',')
        if loaded_array.shape[1] == 3:
            return loaded_array
        raise ValueError("The number of columns in the file is not consistent.")
        
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None
 
def update_visualization():
    global previous_t
    if time.time() - previous_t > dt: 
        loaded_arr  = read_data(filepath)
        if loaded_arr is not None:
            pcd.points = o3d.utility.Vector3dVector(loaded_arr*0.2)    #*.2 as zoom out    
            vis.update_geometry(pcd)
            print(loaded_arr.shape)       
        previous_t = time.time()   
    vis.poll_events()
    # keep_running = vis.poll_events()
    vis.update_renderer()
    

keep_running = True
while keep_running:    
    update_visualization()

vis.destroy_window() 