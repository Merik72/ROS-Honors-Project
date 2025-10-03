#!/usr/bin/env python
# license removed for brevity
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


## msg type:  <class 'sensor_msgs.msg._Image.Image'>
def image_callback(msg):
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb16")  # Convert to BGR format for OpenCV
        # cv_image = cv2.transpose(cv_image)
        cv_image = cv2.flip(cv_image,0)
        cropped_image = cv_image[0:405, 0:1280]
        cv2.imshow("Image", cropped_image)
        # Do something with the OpenCV image
        # cv2.imshow("Image", cropped_image)
        cv2.waitKey(1)  # Adjust the wait key as needed
    except Exception as e:
        print("Error converting image:", e)

def listener():
    rospy.init_node('image_subscriber', anonymous=True)
    rospy.Subscriber("/image_raw_0", Image, image_callback)


    rospy.spin()

if __name__ == '__main__':
    listener()



 