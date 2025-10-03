#!/usr/bin/env python

import random
import rospy
import rosgraph
import time
from unity_robotics_demo_msgs.msg import UnityColor

from std_msgs.msg import String

TOPIC_NAME = 'color'
NODE_NAME = 'color_publisher'


def post_color():
    pub = rospy.Publisher(TOPIC_NAME, UnityColor, queue_size=10)
    rospy.init_node(NODE_NAME, anonymous=True)

    """
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    """
    color = UnityColor(0, 0, 0, 1)

    #wait_for_connections(pub, TOPIC_NAME)
    pub.publish(color)

    time.sleep(0.1)
"""
def post_postion(topicname, value2publish):
    pub = rospy.Publisher(topicname, String, queue_size=10)
    rospy.init_node(NODE_NAME, anonymous=True)

    tmp = "helo from ROS1"

    wait_for_connections(pub, topicname)
    pub.publish(tmp)

    time.sleep(0.1)



def wait_for_connections(pub, topic):
    ros_master = rosgraph.Master('/rostopic')
    topic = rosgraph.names.script_resolve_name('rostopic', topic)
    num_subs = 0
    for sub in ros_master.getSystemState()[1]:
        if sub[0] == topic:
            num_subs+=1

    for i in range(10):
        if pub.get_num_connections() == num_subs:
            return
        time.sleep(0.1)
    raise RuntimeError("failed to get publisher")
"""


if __name__ == '__main__':
    try:
        post_color()
        # post_postion("position","any")
    except rospy.ROSInterruptException:
        pass
