import rospy
import time
from unity_robotics_demo_msgs import UnityColor

def reset():
    pub = rospy.Publisher("color", UnityColor, queue_size=10)
    rospy.init_node('learning', anonymous=True)

    color = UnityColor(0, 0, 0, 1)

    #wait_for_connections(pub, TOPIC_NAME)
    pub.publish(color)

    time.sleep(1)

def hear_termination():
    rospy.Subscriber('/terminated', Bool, callback_termination)

def callback_termination(msg):
    global received_termination
    received_termination = msg

def callback_reward(msg):
    global received_reward
    received_reward = msg

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
    lidar_data = points_array 

def post_wheels(linear, angular):
    pub = rospy.Publisher('TTTTT', Twist, queue_size=10)
    pub.publish(Twist(Vector3(linear,0,0), Vector3(0,0,angular)))