#!/usr/bin/env python

# add rosstuff
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float32, Bool
import time
import random
from unity_robotics_demo_msgs.msg import UnityColor

from torch.distributions import MultivariateNormal, Categorical
"""
Implementation of AlexNet, from paper
"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.

See: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""
import os
import torch
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils import data
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter
# import torchinfo

import ActorCritic
import PPO

# define pytorch device - useful for device-agnostic execution
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model parameters
NUM_EPOCHS = 30  #changed # original paper
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 2  # 1000 classes for imagenet 2012 dataset # cahnge to 2 
DEVICE_IDS = [0]  # GPUs to use, we have one not 1, 2, 3 ,4
# modify this to point to your data directory
# INPUT_ROOT_DIR = 'alexnet_data_in'
# TRAIN_IMG_DIR = 'alexnet_data_in/imagenet'
# OUTPUT_DIR = 'alexnet_data_out'
# LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
# CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints
NUM_STEPS = 90

# add ros stuff
CAMERA_TOPIC = 'env_0/image_raw_1'
WHEEL_TOPIC = 'env_0/vel_wheels'
NODE_NAME = 'ppo'
RESET_TOPIC = 'reset'

# make checkpoint path directory
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# add ros

# camera
def callback_image(msg, ppo_agent):
    global training
    if training:
        bridge = CvBridge()
        try:
            image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr16")
            image = cv2.flip(image,0)
            # cv2.imshow("image", image)
            # print(image.shape)
        except:
            print("Error getting image")
        try:
            training_callback(image, ppo_agent)
        except:
            print("Error with training callback")

steps = 0
###############################################################################################################
def training_callback(image, ppo_agent): 
    global training
    global steps
    steps += 1
    # print(steps)
    # convert image to be usable
    try: 
        image = np.transpose(image, (2,1,0))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
    except:
        print("THIS IS THE IMAGE")
        print(image.dtype)
        print(image.shape)
        print(image)
        
    try:
        end = time.time()
        print("Everything else time ", (start - end))
        start = time.time()
        action = ppo_agent.select_action(image)
        end = time.time()
        print("action time = ", (end-start))
        post_wheels(action[0],action[1])
        start = time.time()
        msg = rospy.wait_for_message('env_0/reward', Float32)
        print("msg time = ", (end-start))
        print("REWARD, ", msg.data)
        ppo_agent.buffer.rewards.append(msg.data)
    except:
        print("THIS IS THE ACTION")
        print(action)

    if steps >= NUM_STEPS:
        training=False

# control
def post_wheels(linear, angular):
    pub = rospy.Publisher(WHEEL_TOPIC, Twist, queue_size=10)
    x = Vector3(linear, 0, 0)
    z = Vector3(0, 0, angular)
    twist = Twist(x, z)
    pub.publish(twist)
    # time.sleep(0.1)

received_reward = 0
def callback_reward(msg):
    global received_reward
    received_reward = msg.data

def post_reset():
    pub = rospy.Publisher(RESET_TOPIC, Bool, queue_size=10)
    #wait_for_connections(pub, TOPIC_NAME)
    pub.publish(False)

training = True
def run():
    global training 
    global received_reward
    global steps
    # add ros
    rospy.init_node(NODE_NAME, anonymous=True, disable_signals=True)

    # why do i need this
    post_wheels(0,0)
    
    # print the seed value
    seed = torch.initial_seed()
    print('Used seed : {}'.format(seed))

    # tbwriter = SummaryWriter(log_dir=LOG_DIR)
    # print('TensorboardX summary writer created')

    # Create PPO
    state_dim = (1, 3, 227, 227)
    action_dim = (1,2)
    learning_rate_actor = 0.0003
    learning_rate_critic = 0.001
    gamma = 0.99
    epochs = NUM_EPOCHS
    epsilon_clip = 0.2
    continuous_action_space = True
    action_std = 0.6
    
    ppo_agent = PPO.PPO(state_dim, action_dim, learning_rate_actor, learning_rate_critic, gamma, epochs, epsilon_clip, continuous_action_space, action_std)

    # debugging and testing
    
    # start training!!
    print('Starting training...')
    epoch = 0
    # total_steps = 1
    while True:
    #for epoch in range(NUM_EPOCHS):
        # lr_scheduler.step()
        
        # in - 1 rosimage
        # out - 1 action
        # after XX seconds of play, update model based on 10 seconds of images
        
        # during this segment, do actions get rewarded
        # training segment
        
        post_reset()
        rospy.Subscriber(CAMERA_TOPIC, Image, callback_image, ppo_agent)
        #print("line after subscriber")
        training = True
        while training:
            pass
        rospy.Subscriber(CAMERA_TOPIC, Image, callback_image, ppo_agent).unregister()
        post_reset()


        #print("exiting")
        #rospy.Subscriber('/reward', Float32, callback_reward)
        #reward = received_reward
        #print(reward)
        #rospy.Subscriber('/reward', Float32, callback_reward).unregister()
        #print("spinning")
        #end if
        steps = 0
        epoch +=1
        # after, do ppo
        print("ppo update reached")
        ppo_agent.update()
        print(" ----- THE EPOCH IS:: ", epoch)

if __name__ == '__main__':
    run()
