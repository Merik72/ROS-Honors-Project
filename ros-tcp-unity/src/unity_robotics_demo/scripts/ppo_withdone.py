#!/usr/bin/env python

# add rosstuff
import glob
import multiprocessing.pool
from threading import Thread
import threading

from imageio import save
from sympy import true
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float32, Bool, Int16
import time
import random
# from unity_robotics_demo_msgs.msg import UnityColor

from torch.distributions import MultivariateNormal, Categorical

import gc
import os
import torch
import PPO
import ActorCritic
from concurrent.futures import ThreadPoolExecutor

# import matplotlib.pyplot as plt
"""
Implementation of AlexNet, from paper
"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.

import torchinfo
See: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils import data
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter


# define pytorch device - useful for device-agnostic execution
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#parallyzed
# from time import sleep

# define model parameters
NUM_EPOCHS = 1  #changed # original paper
BATCH_SIZE = 5
MOMENTUM = 0.9
LR_DECAY = 0.5
LR_ACTOR = 0.0015
LR_CRITIC = 0.0005
LR_FLOOR = 1e-6
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 2  # 1000 classes for imagenet 2012 dataset # cahnge to 2 
DEVICE_IDS = [0]  # GPUs to use, we have one not 1, 2, 3 ,4
INCREASE_STEPS_MODE = False
# modify this to point to your data directory
# INPUT_ROOT_DIR = 'alexnet_data_in'
# TRAIN_IMG_DIR = 'alexnet_data_in/imagenet'
# OUTPUT_DIR = 'alexnet_data_out'
# LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
# CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints

# For the purposes of batching: Needs to be able to make sure every batch is of uniform length: 
#   Therefore, you must tie each batch to its own independent step count 


# add ros stuff
CAMERA_TOPIC = '/image_raw_1'
WHEEL_TOPIC = "/vel_wheels"
NODE_NAME = 'ppo_batches'
RESET_TOPIC = 'reset'
STARTUP_TOPIC = 'start'
DISTANCE_TOPIC = '/distance'
NUM_STEPS_MAX = 120
NUM_STEPS_EXPLORE = 700
NUM_STEPS_INIT = 120
step_increase_freq = 75
num_steps = NUM_STEPS_INIT # number of steps taken by every batch
REPORTING_RATE = 2 # epochs

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# make checkpoint path directory
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# add ros

class Robot:
    def __init__(self, id, steps=0, action = None, training=True, state=None):
        self.id = id
        self.steps = steps
        self.action = action
        self.training = training
        self.state = state
        
        self.action_buf = []
        self.action_logprobs_buf = []
        self.state_buf = []
        self.state_val_buf = []
        self.reward_buf = []
        self.done_buf = []

    def reset(self):
        self.steps = 0
        self.subscriber = None
        self.action = None
        self.training = True
        self.state = None
        
        del self.action_buf[:]
        del self.action_logprobs_buf[:]
        del self.state_buf[:]
        del self.state_val_buf[:]
        del self.reward_buf[:]
        del self.done_buf[:]

        self.action_buf = []        
        self.action_logprobs_buf = []
        self.state_buf = []
        self.state_val_buf = []
        self.reward_buf = []
        self.done_buf = []

# camera
def callback_image(msg, list):
    global training
    if training[list[1].id]:
        bridge = CvBridge()
        try:
            image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr16")
            image = cv2.flip(image,0)

            '''if list[1].steps == 60:
                print(list[1].steps)
                cv2.imwrite(filename="/home/admin2/ros-tcp-unity/src/unity_robotics_demo/scripts/pictures/"+ str(list[1].id) + "/at60.png", img=image)'''
            # if(list[1].id == 0)
            # cv2.imshow("Image", image)
            # cv2.waitKey(1)
        except:
            print("Error getting image")
        try: 
            image = np.transpose(image, (2,1,0))
            image = image.astype(np.float32)
            image = image/65535
            image = torch.from_numpy(image)
            image = image.unsqueeze(0)
        except:
            print("IMAGE PROCESSING ERROR  env_", list[1].id)
        try:
            training_callback(image, list[0], list[1])
        except:
            print("Error with training callback")

locked = False
def lock():
    global locked
    while locked:
        rospy.sleep(0.01)
    locked = True
    
def unlock():
    global locked
    locked = False

def training_callback(image, ppo_agent, robot): 
    global training
    global num_steps
    global someTraining
    try:
        if(robot.steps < num_steps):
            robot.action, action_logprobs, state_val, state, action = ppo_agent.parallel_happy_select_action(image)
            robot.action_buf.append(action)
            robot.action_logprobs_buf.append(action_logprobs)
            robot.state_val_buf.append(state_val)
            robot.state_buf.append(state)
        robot.steps += 1
    except:
        print("ACTION ERROR env_", str(robot.id), " step ", robot.steps)
    try:
        post_wheels(robot.action[1], robot.action[0], robot.id)
        msg = rospy.wait_for_message('env_' + str(robot.id) + '/reward', [Float32, Bool])
        # print(1)
        # ppo_agent.buffer.rewards.append(msg.data)
        robot.reward_buf.append(msg.data[0])
        if robot.steps >= num_steps:
            done = True
        else:  
            done = msg.data[1]
        '''if robot.steps == 60:
            print(robot.id, " : ", msg.data)'''
        # print(2)
        # done = msg.data == -10
        # print(3)
        allWall[robot.id] = done
        robot.done_buf.append(done)
        # print(4)
        # print(5)
    
    except:
        print("reward failure")
        print("robotaction after failures: ", robot.action)
        '''self.action_buf = []
        self.action_logprobs_buf = []
        self.state_buf = []
        self.state_val_buf = []
        self.reward_buf = []'''

# control
def post_wheels(linear, angular, num):
    pub = rospy.Publisher("env_" + str(num) + WHEEL_TOPIC, Twist, queue_size=1)
    x = Vector3(linear, 0, 0)
    z = Vector3(0, 0, angular)
    twist = Twist(x, z)
    pub.publish(twist)
    # time.sleep(0.1)

received_reward = 0
def callback_reward(msg):
    global received_reward
    received_reward = msg.data

distance = 0
def callback_distance(msg):
    global distance
    if(msg.data > 1):
        distance += msg.data

def post_reset():
    pub = rospy.Publisher(RESET_TOPIC, Bool, queue_size=0)
    wait_for_connections(pub)
    pub.publish(False)

def startup():
    pub = rospy.Publisher(STARTUP_TOPIC, Int16, queue_size=0)
    wait_for_connections(pub)
    pub.publish(BATCH_SIZE)



def wait_for_connections(pub):
    while (pub.get_num_connections() != 1):
        pass

training = []
someTraining = True
currentID = 0
def train(robot):
    global ppo_agent
    global num_steps
    # rospy.init_node(NODE_NAME + str(robot.id), anonymous=True, disable_signals=True)
    for x in range(num_steps+1):
        # print("It's working")
        # print("E", str(robot.id), "S" , x)#, " \t IMAGE GOTTEN")
        image = rospy.wait_for_message("env_" + str(robot.id) + CAMERA_TOPIC, Image)
        # print("E", str(robot.id), "S" , x)#, "\t SENT AWAY")
        callback_image(image, (ppo_agent, robot))
        # if allWall[robot.id] == true:
        #     break
        # alltrue = False
        # for bool in allWall:
        #     if bool == False:
        #         alltrue = False
        #         break
        #     else:
        #         alltrue = True
        # if alltrue:
        #     break


# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)
        print("applied")
    if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        print("conv applied")

robots = []
allWall = []
ppo_agent = None
def run():
    global training 
    #global steps
    global someTraining
    global currentID
    global robots
    global ppo_agent
    global allWall
    for x in range(BATCH_SIZE):
        training.append(True)
        allWall.append(False)
        #steps.append(0)
        robots.append(Robot(x))
    # global image
    global received_reward
    # add ros
    rospy.init_node(NODE_NAME, anonymous=True, disable_signals=True)
    # why do i need this
    post_reset()
    # print("something")  
    
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
    gamma = 0.9
    epochs = NUM_EPOCHS
    epsilon_clip = 0.2
    continuous_action_space = True
    action_std = .8
    
    ppo_agent = PPO.PPO(state_dim, action_dim, learning_rate_actor, learning_rate_critic, gamma, epochs, epsilon_clip, continuous_action_space, action_std, LR_DECAY, 0)
    ppo_agent.policy.apply(weights_init_uniform_rule)
    ppo_agent.policy_old.apply(weights_init_uniform_rule)
    # ppo_agent.policy_old.load_state_dict(torch.load('old.pt',map_location=lambda storage, loc: storage))
    # ppo_agent.policy.load_state_dict(torch.load('old.pt',map_location=lambda storage, loc: storage))
    # debugging and testing
    # takes in a module and applies the specified weight initialization

    # start training!!
    print('Starting training...')
    # total_steps = 1
      
    #################### set up the number of environments ####################
    
    try:
        startup()
        post_reset()
    except:
        print("could not startup")

def training_loop():
    global someTraining
    global num_steps
    global step_increase_freq
    global allWall
    global distance
    post_reset()
    epoch = 1
    trainingTime = time.time()
    while True:
        someTraining = True
        for x in range(BATCH_SIZE):
            training[x] = True
            allWall[x] = False
            #steps[x] = 0
        ############ This is the part where he trains you ###############
        if(INCREASE_STEPS_MODE):
            if epoch == 0:
                num_steps = int(NUM_STEPS_EXPLORE/BATCH_SIZE)
            if epoch == 1:
                num_steps = NUM_STEPS_INIT
        
        with ThreadPoolExecutor(BATCH_SIZE) as exe:
            exe.map(train, robots)
        
        # Update visualization
        post_reset()
        rospy.Subscriber(DISTANCE_TOPIC, Float32, callback_distance)
        if(epoch % REPORTING_RATE == 0):
            print('---- Metrics ----')
            # Wall clock time taken across attempts
            trainingTime = time.time() - trainingTime
            print(trainingTime)
            # AVGPath lengths
            # Path calculation downtime
            # Storage space (Doesn't need to do)
            # Time taken to reach each waypoint
        else:
            distance = 0
        
        print(distance/BATCH_SIZE)

        someTraining = False

        # print("updating")
        if(robots[0].steps == 0):
            # print('why')
            rospy.sleep(1)
            continue
        for robot in robots:
            # print(robot.steps)
            for step in range(robot.steps-1):
                ppo_agent.push_values(
                    robot.state_buf[step],
                    robot.action_buf[step],
                    robot.action_logprobs_buf[step],
                    robot.state_val_buf[step],
                    robot.reward_buf[step],
                    robot.done_buf[step]
                )
            # print("Robot", robot.id , "reward is")
            # print(np.mean(robot.reward_buf))

        ppo_agent.update()
        post_reset()

        min_action_std = 0.25
        action_std_decay_freq = 100
        action_std_decay_rate = 0.05
        # if continuous action space; then decay action std of ouput action distribution
        if epoch > 1 and epoch % action_std_decay_freq == 0: # every 25 epochs
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
        if(INCREASE_STEPS_MODE):
            if num_steps<NUM_STEPS_MAX and epoch > 1 and epoch % step_increase_freq == 0: # every 25 epochs
                step_increase_freq *= 1.5
                num_steps +=10
                print(num_steps)
        # print(num_steps, "steps per epoch")
        # print(ppo_agent.policy)

        # after, do ppo
        epoch +=1
        for x in range(BATCH_SIZE):
            training[x] = False
            robots[x].reset()
            #steps[x] = 0
        ## print("ppo update reached")
        print(" ----- THE EPOCH IS:: ", epoch)
        ###if this is the beest save it

        #'''py
        ### if GPU memory still crashing

        #if (Best_Average_Reward < Current_Average_Reward)
        torch.save(ppo_agent.policy_old.state_dict(), 'old.pt')

        # del ppo_agent.policy_old
        # del ppo_agent.policy

        gc.collect()
        torch.cuda.empty_cache()

        ppo_agent.policy_old.load_state_dict(torch.load('old.pt',map_location=lambda storage, loc: storage))
        ppo_agent.policy.load_state_dict(torch.load('old.pt',map_location=lambda storage, loc: storage))
        #'''
        
if __name__ == '__main__':
    run()
    training_loop()
    
