#!/usr/bin/env python

# add rosstuff
import glob
import multiprocessing.pool
from threading import Thread
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

#parallyzed
from concurrent.futures import ThreadPoolExecutor
# from time import sleep

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
NUM_STEPS = 60 # number of steps taken by every batch


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
class Robot:
    def __init__(self, id, steps=0, action = None, training=True, state=None):
        self.id = id
        self.steps = steps
        self.action = action
        self.training = training
        self.state = state
        
        self.action_buf = []
        self.state_buf = []
        self.reward_buf = []
    def reset(self):
        self.steps = 0
        self.subscriber = None
        self.action = None
        self.training = True
        self.state = None
        
        del self.action_buf 
        del self.state_buf 
        del self.reward_buf 
        self.action_buf = []
        self.state_buf = []
        self.reward_buf = []

# camera
def callback_image(msg, list):
    global training
    if training[list[1].id]:
        bridge = CvBridge()
        try:
            image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr16")
            image = cv2.flip(image,0)
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
            list[1].state_buf.append(image)
        except:
            print("IMAGE PROCESSING ERROR  env_", list[1].id)
        try:
            training_callback(image, list[0], list[1])
        except:
            print("Error with training callback")

def training_callback(image, ppo_agent, robot): 
    global training
    #global steps
    global someTraining
    global start_time
    global end_time
    try:
        if (robot.steps == 0):
            robot.action = ppo_agent.parallel_happy_select_action(image)
            robot.action_buf.append(robot.action)
    except:
        print("init action failure")
    
    try:
        print("\taction:", robot.action, "\n")
        post_wheels(robot.action[0], robot.action[1], robot.id)
        robot.action_buf.append(robot.action)
        if(not (robot.steps == 0)): 
            msg = rospy.wait_for_message('env_' + str(robot.id) + '/reward', Float32)
            ppo_agent.buffer.rewards.append(msg.data)
            robot.reward_buf.append(robot.action)
    
    except:
        print("reward failure")

    try:
        if(robot.steps < NUM_STEPS-1):
            robot.action = ppo_agent.select_action(image)
            '''start_time = time.time()
            if(not (robot.steps == 0)): 
                print("\nNon-action time was: ", (start_time-end_time))
            robot.action = ppo_agent.parallel_happy_select_action(image)
            end_time = time.time()
            print("\nAction time was: ", (end_time-start_time))'''
        robot.steps += 1
    except:
        print("ACTION ERROR env_", str(robot.id), " step ", robot.steps)
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
    # rospy.init_node(NODE_NAME + str(robot.id), anonymous=True, disable_signals=True)
    for x in range(NUM_STEPS+1):
        print("E", str(robot.id), "S" , x, " \t IMAGE GOTTEN")
        image = rospy.wait_for_message("env_" + str(robot.id) + CAMERA_TOPIC, Image)
        print("E", str(robot.id), "S" , x, "\t SENT AWAY")
        callback_image(image, (ppo_agent, robot))

robots = []
ppo_agent = None
def run():
    global training 
    #global steps
    global someTraining
    global currentID
    global robots
    global ppo_agent
    for x in range(BATCH_SIZE):
        training.append(True)
        #steps.append(0)
        robots.append(Robot(x))
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

    # debugging and testing
    
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

    post_reset()
    epoch = 0
    
    while True:
        someTraining = True
        for x in range(BATCH_SIZE):
            training[x] = True
            #steps[x] = 0
        ############ This is the part where he trains you ###############
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as exe:
            exe.map(train, robots)

        #processes = []
        '''for x in range(BATCH_SIZE):
            p = multiprocessing.Process(target=train, args=[robots[x]])
            #processes.append(p)
            p.start()
            p.join()'''
        '''# pool = multiprocessing.Pool(BATCH_SIZE)
        p = multiprocessing.Pool(processes=BATCH_SIZE)
        p.map(train, robots)
        p.close()
        p.join()'''
        '''
        for x in range(BATCH_SIZE):
            p = multiprocessing.Process(target=train, args=[robots[x]])
            p.start()'''
        '''# Create a second NodeHandle 
        ros::NodeHandle secondNh;
        ros::CallbackQueue secondQueue;
        secondNh.setCallbackQueue(&secondQueue);
        secondNh.subscribe("/high_priority_topic", 1,
                        highPriorityCallback);
        // Spawn a new thread for high-priority callbacks.
        std::thread prioritySpinThread([&secondQueue]() {
            ros::SingleThreadedSpinner spinner;
            spinner.spin(&secondQueue);
        });
        prioritySpinThread.join();'''

        post_reset()

        someTraining = False
        
        ## Before reseting the robots add everything to ppo ################################################################################################
        
        for x in range(BATCH_SIZE):
            training[x] = False
            robots[x].reset()
            #steps[x] = 0
        ## print("ppo update reached")

        '''for state in ppo_agent.buffer.states:
            print(state)'''
        print("updating")
        ppo_agent.update()
        min_action_std = 0.1
        action_std_decay_freq = 25
        action_std_decay_rate = 0.05
        # if continuous action space; then decay action std of ouput action distribution
        if epoch % action_std_decay_freq == 0: # every 25 epochs
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
        # print(ppo_agent.policy)
        # after, do ppo
        epoch +=1
        ## print(" ----- THE EPOCH IS:: ", epoch)

if __name__ == '__main__':
    run()
    training_loop()
    
