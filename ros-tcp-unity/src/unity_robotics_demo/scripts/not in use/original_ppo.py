#!/usr/bin/env python

# add rosstuff
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float32
import time
import random
from unity_robotics_demo_msgs.msg import UnityColor
"""
Implementation of AlexNet, from paper
"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.

See: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model parameters
NUM_EPOCHS = 3  #changed # original paper
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 2  # 1000 classes for imagenet 2012 dataset # cahnge to 2 
DEVICE_IDS = [0]  # GPUs to use, we have one not 1, 2, 3 ,4
# modify this to point to your data directory
INPUT_ROOT_DIR = 'alexnet_data_in'
TRAIN_IMG_DIR = 'alexnet_data_in/imagenet'
OUTPUT_DIR = 'alexnet_data_out'
LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints

# add ros stuff
CAMERA_TOPIC = '/image_raw_1'
WHEEL_TOPIC = 'TTTTT'
NODE_NAME = 'ppo'
RESET_TOPIC = 'color'

# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, num_classes=2):
        """
        Define and allocate layers for this neural net.

        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.Sigmoid(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.Sigmoid(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.Sigmoid(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.Sigmoid(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.Sigmoid(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Sigmoid(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        """
        Pass the input through the net.

        Args:
            x (Tensor): input tensor

        Returns:
            output (Tensor): output tensor
        """
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input
        return self.classifier(x)

# add ros

# camera
image = None
def callback_image(msg, alexnet):
    global image
    global training
    bridge = CvBridge()
    try:
        image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr16")
        image = cv2.flip(image,0)
        # print(image.shape)
        training_callback(image, alexnet)
    except:
        print("Error getting image")

# control
def post_wheels(linear, angular):
    pub = rospy.Publisher(WHEEL_TOPIC, Twist, queue_size=10)
    x = Vector3(linear, 0, 0)
    z = Vector3(0, 0, angular)
    twist = Twist(x, z)
    pub.publish(twist)
    # time.sleep(0.1)

steps = 0
def training_callback(image, alexnet): 
    global training
    global steps
    steps += 1
    print(steps)
    # convert image to be usable
    try: 
        image = np.transpose(image, (2,1,0))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        imageout = alexnet(image)
    except:
        print(image.dtype)
        print(image.shape)
        print(imageout)
    try:
        post_wheels(imageout[0][0], imageout[0][1])
    except:
        print("something went wrong")
        print(imageout[0][0])
        print(imageout[0][1])
    if steps >= 600:
        training=False

received_reward = 0
def callback_reward(msg):
    global received_reward
    received_reward = msg.data

def post_reset():
    pub = rospy.Publisher(RESET_TOPIC, UnityColor, queue_size=10)

    color = UnityColor(0, 0, 0, 1)

    #wait_for_connections(pub, TOPIC_NAME)
    pub.publish(color)

training = True
def run():
    global training 
    global image
    global received_reward
    global steps
    # add ros
    rospy.init_node(NODE_NAME, anonymous=True, disable_signals=True)

    # why do i need this
    post_wheels(0,0)
    
    # print the seed value
    seed = torch.initial_seed()
    print('Used seed : {}'.format(seed))

    tbwriter = SummaryWriter(log_dir=LOG_DIR)
    print('TensorboardX summary writer created')

    # create model
    alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)
    # train on multiple GPUs
    alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=DEVICE_IDS)
    
    state_dim = (3,227,227)
    action_dim = (1,2)
    learning_rate_actor = 0.0003
    learning_rate_critic = 0.001
    gamma = 0.99
    epochs = NUM_EPOCHS
    epsilon_clip = 0.2
    continuous_action_space = True
    action_std = 0.6

    ppo_agent = PPO(state_dim, action_dim, learning_rate_actor, learning_rate_critic, gamma, epochs, epsilon_clip, continuous_action_space, action_std)

    # debugging and testing
    
    tensor = torch.randn(size=(3, 227, 227))
    output = alexnet(tensor)
    print(output)
    print("output: ") 
    print(output.shape)
    try:
        post_wheels(output[0][0], output[0][1])
    except:
        print("something went wrong")
        print(output[0][0])
        print(output[0][1])
    print('AlexNet created')
    print("Alexnet: ")
    print(alexnet)
    
    '''
    # create dataset and data loader
    dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose([
        # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    print('Dataset created')
    dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)
    print('Dataloader created')
    '''

    # create optimizer
    # the one that WORKS
    optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
    ### BELOW is the setting proposed by the original paper - which doesn't train....
    # optimizer = optim.SGD(
    #     params=alexnet.parameters(),
    #     lr=LR_INIT,
    #     momentum=MOMENTUM,
    #     weight_decay=LR_DECAY)
    print('Optimizer created')

    # multiply LR by 1 / 10 after every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('LR Scheduler created')

    # start training!!
    print('Starting training...')
    total_steps = 1
    for epoch in range(NUM_EPOCHS):
        lr_scheduler.step()
        
        # in - 1 rosimage
        # out - 1 action
        # after XX seconds of play, update model based on 10 seconds of images
        
        # during this segment, do actions get rewarded
        reward = 0
        rospy.Subscriber(CAMERA_TOPIC, Image, callback_image, alexnet)
        print("line after subscriber")
        training = True
        while training:
            print(" ", end="")
        if steps >= 600:
            print("exiting")
            rospy.Subscriber(CAMERA_TOPIC, Image, callback_image, alexnet).unregister()
            rospy.Subscriber('/reward', Float32, callback_reward)
            #rospy.wait_for_message('/reward', Float32)
            reward = received_reward
            print(reward)
            rospy.Subscriber('/reward', Float32, callback_reward).unregister()
            print("spinning")
        post_reset()
        steps = 0

        # after, do ppo

        '''
        lr_scheduler.step()
        for imgs, classes in dataloader:
            imgs, classes = imgs.to(device), classes.to(device)

            # calculate the loss
            output = alexnet(imgs)
            loss = F.cross_entropy(output, classes)

            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log the information and add to tensorboard
            if total_steps % 10 == 0:
                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    accuracy = torch.sum(preds == classes)

                    print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                        .format(epoch + 1, total_steps, loss.item(), accuracy.item()))
                    tbwriter.add_scalar('loss', loss.item(), total_steps)
                    tbwriter.add_scalar('accuracy', accuracy.item(), total_steps)

            # print out gradient values and parameter average values
            if total_steps % 100 == 0:
                with torch.no_grad():
                    # print and save the grad of the parameters
                    # also print and save parameter values
                    print('*' * 10)
                    for name, parameter in alexnet.named_parameters():
                        if parameter.grad is not None:
                            avg_grad = torch.mean(parameter.grad)
                            print('\t{} - grad_avg: {}'.format(name, avg_grad))
                            tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
                            tbwriter.add_histogram('grad/{}'.format(name),
                                    parameter.grad.cpu().numpy(), total_steps)
                        if parameter.data is not None:
                            avg_weight = torch.mean(parameter.data)
                            print('\t{} - param_avg: {}'.format(name, avg_weight))
                            tbwriter.add_histogram('weight/{}'.format(name),
                                    parameter.data.cpu().numpy(), total_steps)
                            tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)

            total_steps += 1
        '''

        # save checkpoints
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epoch + 1))
        state = {
            'epoch': epoch,
            'total_steps': total_steps,
            'optimizer': optimizer.state_dict(),
            'model': alexnet.state_dict(),
            'seed': seed,
        }
        torch.save(state, checkpoint_path)

if __name__ == '__main__':
    run()

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
    