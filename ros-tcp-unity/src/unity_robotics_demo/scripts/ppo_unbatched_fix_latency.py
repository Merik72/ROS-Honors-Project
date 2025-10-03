#!/usr/bin/env python

# add rosstuff
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
BATCH_SIZE = 1
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
NUM_STEPS = 180 # number of steps taken by every batch


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

# add ros

# camera
def callback_image(msg, list):
    global training
    if training[list[1]]:
        bridge = CvBridge()
        try:
            image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr16")
            #cv2.imwrite("my_image.jpg", image)
            image = cv2.flip(image,0)
            cv2.imshow("Image", image)
            cv2.waitKey(1)

            #if list[1] == 0:
            #    print(image)
                # cv2.imshow("image", image)
        except:
            print("Error getting image")
        
        try:
            training_callback(image, (list[0], list[1]))
        except:
            print("Error with training callback")
           

steps = []
###############################################################################################################
# List [0] - PPO agent
# List [1] - env number
action = []
def training_callback(image, list): 
    global training
    global steps
    global someTraining
    global action
    global start_time
    global end_time
####print("Step count of", list[1], " is ", steps[list[1]])
    if steps[list[1]] >= NUM_STEPS+1:
        training[list[1]] = False # finished training
        
        result = False
        for val in training:
            result = val
            if(val == True):
                break

        someTraining = result
        # if someTraining:
            ## print("env_", list[1], " is finished but training is still happening")
        # else:
            ## print("No training! Ready to update")
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
        # states[list[1]] = image
        # print("image torched")
        image = image.unsqueeze(0) # makes a new line in tensor
        # image[0][0] = list[1] # fills it
        # print(image.shape)
        '''
        if list[1] == 3:
            ## print("Image:", image, "end")
            ## print("Image shape", image.shape, "end")
        '''
        
    
    except:
        print("IMAGE PROCESSING ERROR  env_", list[1], " step ", steps[list[1]])
        # print("This is the datatype", image.dtype, "CLOSE")
        ## print("shape", image.shape, "CLOSE")
        # print("This is the image", image, "CLOSE")
        
    try:
        # print("before first condition", not (steps[list[1]] == 0))
        # print("list1 ", list[1])
        # print("steplist1 ", steps[list[1]] )
        if (steps[list[1]] == 0):
            action = list[0].select_action(image)
        if(not (steps[list[1]] == 0)): 
            # print("after condition")
            msg = rospy.wait_for_message('env_' + str(list[1]) + '/reward', Float32)
            # print("msgreceived")
            list[0].buffer.rewards.append(msg.data) ###################################################################
            ## print("\tReward: \t", msg.data)
        ## print("For env_", list[1], " step ", steps[list[1]])
        ## print("\tTwist: \tLINEAR: \t", action[0], " \tANGULAR", action[1])
        post_wheels(action[0], action[1], list[1])
        if(steps[list[1]] < NUM_STEPS-1):
            # list[0].select_action(image) ###################################################################
            start_time = time.time()
            if(not (steps[list[1]] == 0)): 
                print("\nNon-action time was: ", (start_time-end_time))
            ## print("\nStarting")
            action = list[0].select_action(image) ###################################################################
            ## print("\nEnding")
            end_time = time.time()
            print("\nAction time was: ", (end_time-start_time))
            # print(action.size)
            
            #if torch.cuda.is_available():
            #    print("Device is ", device)
            #    print("Yay!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(action)
            # post_wheels(action[0][0], action[0][1], list[1]) #########################################################
        steps[list[1]] += 1
#####   print("REWARD from batch ", str(list[1]), ": ", msg.data)
    except:
        print("ACTION ERROR env_", list[1], " step ", steps[list[1]])
        ## print("Action: ", action)
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
currentID = 0
def run():
    global training 
    global steps
    global someTraining
    global currentID
    for x in range(BATCH_SIZE):
        training.append(True)
        steps.append(0)
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
    
    min_action_std = 0.1
    action_std_decay_freq = 25
    action_std_decay_rate = 0.05

    # debugging and testing
    
    # start training!!
    print('Starting training...')
    epoch = 0
    # total_steps = 1
      
    #################### set up the number of environments ####################
    try:
        startup()
        post_reset()
    except:
        print("could not startup")

    while True:
    #for epoch in range(NUM_EPOCHS):
        
        someTraining = False
        for x in range(BATCH_SIZE):
            training[x] = False
            steps[x] = 0
        currentID = 0
        post_reset()

        #################### subscribe to each different camera ####################
        someTraining = True
        for x in range(BATCH_SIZE):
            training[x] = True
            steps[x] = 0

        subscribers = []
        for num in range(BATCH_SIZE):
            if someTraining:
                subscribers.append(rospy.Subscriber("env_" + str(num) + CAMERA_TOPIC, Image, callback_image, (ppo_agent, num),queue_size=1))
                # print("created number ", num)
        
        #print("line after subscriber")
        ## print("now training")
        while someTraining:
            print(" \b", end='')
        
        for num in range(BATCH_SIZE):
            subscribers[num].unregister()  
        
        post_reset()

        someTraining = False
        for x in range(BATCH_SIZE):
            training[x] = False
            steps[x] = 0
        ## print("ppo update reached")

        #for state in ppo_agent.buffer.states:
            #print(state)
        ppo_agent.update()
        # if continuous action space; then decay action std of ouput action distribution
        if epoch % action_std_decay_freq == 0: # every 25 epochs
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
        # print(ppo_agent.policy)
        # after, do ppo
        epoch +=1
        ## print(" ----- THE EPOCH IS:: ", epoch)

if __name__ == '__main__':
    
    run()
