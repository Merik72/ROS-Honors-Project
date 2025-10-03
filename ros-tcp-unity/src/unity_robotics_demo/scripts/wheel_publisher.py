#!/usr/bin/env python

import random
import rospy
import rosgraph
import time

from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Twist

from std_msgs.msg import String

TOPIC_NAME = '/vel/wheels'
NODE_NAME = 'test_command'


def post_command():
    pub = rospy.Publisher(TOPIC_NAME, Twist, queue_size=10)
    rospy.init_node(NODE_NAME, anonymous=True)
    
    x = random.uniform(-1,1)
    z = random.uniform(-1,1)
    linear = Vector3(x, 0, 0)
    angular = Vector3(0, 0, z)
    #linear = Vector3(1, 0, 0)
    #angular = Vector3(0, 0, 1)
    twist = Twist(linear, angular)


    pub.publish(twist)

    time.sleep(0.1)

if __name__ == '__main__':
    for i in range(1000):
        try:
            post_command()
            # post_postion("position","any")
        except rospy.ROSInterruptException:
            pass

#!/usr/bin/env python

"""
import rospy
from geometry_msgs.msg import Twist

def drive_in_circle():
    rospy.init_node('drive_in_circle', anonymous=True)
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(20)  # 10 Hz

    move_cmd = Twist()
    move_cmd.linear.x = 0.1  # Forward velocity
    move_cmd.angular.z = 0.1  # Angular velocity

    while not rospy.is_shutdown():
        pub.publish(move_cmd)
        rate.sleep()

if __name__ == '__main__':
    try:
        drive_in_circle()
    except rospy.ROSInterruptException:
        pass

"""