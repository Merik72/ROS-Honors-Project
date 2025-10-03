#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from unity_robotics_demo_msgs.msg import Velocity

def publisher():
    rospy.init_node('string_publisher', anonymous=True)
    pub = rospy.Publisher('unity_text', String, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    while not rospy.is_shutdown():
        text_to_publish = "Hello from ROS!"
        rospy.loginfo(text_to_publish)
        pub.publish(text_to_publish)
        rate.sleep()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
