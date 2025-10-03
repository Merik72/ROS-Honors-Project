#!/usr/bin/env python

# add rosstuff
from matplotlib.pyplot import step
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float32, Bool, Int16
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
import torchinfo

import ActorCritic
import PPO

# define pytorch device - useful for device-agnostic execution
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model parameters
NUM_EPOCHS = 1  #changed # original paper
BATCH_SIZE = 10
MOMENTUM = 0.9
LR_DECAY = 0.5
LR_ACTOR = 0.0003
LR_CRITIC = 0.001
LR_FLOOR = 1e-6
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 2  # 1000 classes for imagenet 2012 dataset # cahnge to 2 
DEVICE_IDS = [0]  # GPUs to use, we have one not 1, 2, 3 ,4
# modify this to point to your data directory
# INPUT_ROOT_DIR = 'alexnet_data_in'
# TRAIN_IMG_DIR = 'alexnet_data_in/imagenet'
# OUTPUT_DIR = 'alexnet_data_out'
# LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
# CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints

# For the purposes of batching: Needs to be able to make sure every batch is of uniform length: 
#   Therefore, you must tie each batch to its own independent step count 
NUM_STEPS = 20 # number of steps taken by every batch


# add ros stuff
CAMERA_TOPIC = '/image_raw_1'
WHEEL_TOPIC = "/vel_wheels"
NODE_NAME = 'ppo_batches'
RESET_TOPIC = 'reset'
STARTUP_TOPIC = 'start'
# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# make checkpoint path directory
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)
class Robot:
    def __init__(self, subscriber, id, action = None, training=True, steps=0, image=None):
        self.subscriber = subscriber
        self.id = id
        self.action = action
        self.training = training
        self.steps = steps
        self.image = image
    def reset(self):
        self.subscriber = None
        self.action = None
        self.training = True
        self.steps = 0
        image = None

# add ros
# camera
def callback_image(msg, list):
    global training
    if list[1].training:
        bridge = CvBridge()
        try:
            image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr16")
            image = cv2.flip(image,0)
        except:
            print("Error getting image")
        
        try:
            training_callback(image, list[0], list[1])
        except:
            print("Error with training callback")
           

steps = []
###############################################################################################################
# List [0] - PPO agent
# List [1] - env number
action = []
def training_callback(image, ppo_agent, robot): 
    global training
    global steps
    global someTraining
    global action
####print("Step count of", robot.id, " is ", robot.steps)
    if robot.steps >= NUM_STEPS+1:
        training[robot.id] = False # finished training
        robot.training = False
        result = False
        for val in training:
            result = val
            if(val == True):
                break

        someTraining = result
        if someTraining:
            print("env_", robot.id, " is finished but training is still happening")
            print(robot.steps)
            print(robot.action)
        else:
            print("No training! Ready to update")
            updateAndStuff(ppo_agent)
        robot.subscriber.unregister()
        return
    # convert image to be usable
    try: 
        image = np.transpose(image, (2,1,0))
        # print("image transposed")
        image = image.astype(np.float32)
        # print("astype")

        #normalized?
        image = image/65535

        image = torch.from_numpy(image)
        # states[robot.id] = image
        # print("image torched")
        image = image.unsqueeze(0) # makes a new line in tensor
        # image[0][0] = robot.id # fills it
        # print(image.shape)
        '''
        if robot.id == 3:
            print("Image:", image, "end")
            print("Image shape", image.shape, "end")
        '''
        
    
    except:
        print("IMAGE PROCESSING ERROR  env_", robot.id, " step ", robot.steps)
        print("shape", image.shape, "CLOSE")
        
    try:
        robot.action = ppo_agent.select_action(image)
    except:
        print("INITIAL ACTION ERROR env_", robot.id, " step ", robot.steps)
        print("Action: ", robot.action)

    try:
        if(not (robot.steps == 0)): 
            # print("after condition")
            msg = rospy.wait_for_message('env_' + str(robot.id) + '/reward', Float32)
            # print("msgreceived")
            ppo_agent.buffer.rewards.append(msg.data) ###################################################################
            if(robot.id == 0):
                print("\tReward: \t", msg.data)
    except:
        print("REWARD ERROR env_", robot.id, " step ", robot.steps)
        print("Action: ", robot.action)

    try:
        if(robot.id == 0):
                print("For env_", robot.id, " step ", robot.steps)
        if(robot.id == 0):
                print("\tTwist: \tLINEAR: \t", robot.action[0], " \tANGULAR", robot.action[1])
        post_wheels(robot.action[0], robot.action[1], robot.id)
        if(robot.steps < NUM_STEPS-1):
            # ppo_agent.select_robot.action(image) ###################################################################
            if(robot.id == 0):
                print("\nStarting")
            start_time = time.time()
            robot.action = ppo_agent.select_action(image) ###################################################################
            if(robot.id == 0):
                print("\nEnding")
            end_time = time.time()
            if(robot.id == 0):
                print("\nAction time was: ", (end_time-start_time))
            # print(action.size)
            
            #if torch.cuda.is_available():
            #    print("Device is ", device)
            #    print("Yay!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(action)
            # post_wheels(robot.action[0][0], robot.action[0][1], robot.id) #########################################################
        robot.steps += 1
#####   print("REWARD from batch ", str(robot.id), ": ", msg.data)
    except:
        print("ACTION ERROR env_", robot.id, " step ", robot.steps)
        print("Action: ", robot.action)
# control
def post_wheels(linear, angular, num):
    pub = rospy.Publisher("env_" + str(num) + WHEEL_TOPIC, Twist, queue_size=10)
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
    wait_for_connections(pub)
    pub.publish(False)

def startup():
    pub = rospy.Publisher(STARTUP_TOPIC, Int16, queue_size=10)
    wait_for_connections(pub)
    pub.publish(BATCH_SIZE)



def wait_for_connections(pub):
    while (pub.get_num_connections() < 1):
        pass

training = []
someTraining = True
robots = []
def defineThings():
    global training 
    global steps
    global someTraining
    global action
    global robots
    # Create PPO
    state_dim = (1, 3, 227, 227)
    action_dim = (1, 2)
    learning_rate_actor = LR_ACTOR # 0.03 initial learning rate
    learning_rate_critic = LR_CRITIC # 0.01 initial learning rate
    gamma = 0.99
    epochs = NUM_EPOCHS
    epsilon_clip = 0.2
    continuous_action_space = True
    action_std = 0.9
    
    ppo_agent = PPO.PPO(state_dim, action_dim, learning_rate_actor, learning_rate_critic, gamma, epochs, epsilon_clip, continuous_action_space, action_std, LR_DECAY, 0)
    #action_shape = []
    # ppo_agent.select_action(torch.zeros(state_dim))
    for x in range(BATCH_SIZE):
        training.append(True)
        #action_shape.append(ppo_agent.select_action(torch.zeros(state_dim)))
        action.append(np.array([0,0], dtype=np.float32))
        steps.append(0)
        robots.append(Robot(None, x))
    #print(action_shape)
    print(action)
    # global image
    global received_reward
    # add ros
    rospy.init_node(NODE_NAME, anonymous=True, disable_signals=True)
    # why do i need this
    post_reset()
    print("something")  
    
    # print the seed value
    seed = torch.initial_seed()
    print('Used seed : {}'.format(seed))

    # tbwriter = SummaryWriter(log_dir=LOG_DIR)
    # print('TensorboardX summary writer created')
    # debugging and testing
      
    #################### set up the number of environments ####################
    try:
        startup()
        post_reset()
    except:
        print("could not startup")
    
    trainStartup(ppo_agent)

def trainStartup(ppo_agent):
    global training 
    global steps
    global someTraining
    global action
    # start training!!
    print('Starting training...')
    # total_steps = 1
    #for epoch in range(NUM_EPOCHS):

    post_reset()

    #################### subscribe to each different camera ####################
    someTraining = True
    for x in range(BATCH_SIZE):
        training[x] = True
        steps[x] = 0
        action[x] = np.array([0,0], dtype=np.float32)

    for num in range(BATCH_SIZE):
        robots[num].subscriber = rospy.Subscriber("env_" + str(num) + CAMERA_TOPIC, Image, callback_image, (ppo_agent, robots[num]))
    # print("created number ", num)
    
g_epoch = 0
def updateAndStuff(ppo_agent):
    global training 
    global steps
    global someTraining
    global g_epoch
    post_reset()

    someTraining = True
    for x in range(BATCH_SIZE):
        training[x] = False
        steps[x] = 0
        robots[x].reset()
    print("ppo update reached")

    #for state in ppo_agent.buffer.states:
        #print(state)
    ppo_agent.update()
    min_action_std = 0.1
    action_std_decay_freq = 25
    action_std_decay_rate = 0.05
    # if continuous action space; then decay action std of ouput action distribution
    if g_epoch % action_std_decay_freq == 0: # every 25 epochs
        ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
    # print(ppo_agent.policy)
    # after, do ppo
    g_epoch +=1
    print(" ----- THE EPOCH IS:: ", g_epoch)
    trainStartup(ppo_agent)


if __name__ == '__main__':
    defineThings()
    rospy.spin()
