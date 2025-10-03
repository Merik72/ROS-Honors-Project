import torch
import ActorCritic
import PPO_GAE as PPO

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#parallyzed
# from time import sleep

# define model parameters
NUM_EPOCHS = 1  #changed # original paper
BATCH_SIZE = 5
MOMENTUM = 0.9
LR_DECAY = 0.5
LR_ACTOR = 0.0001
LR_CRITIC = 0.0005
LR_FLOOR = 1e-6
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 2  # 1000 classes for imagenet 2012 dataset # cahnge to 2 
DEVICE_IDS = [0]  # GPUs to use, we have one not 1, 2, 3 ,4
INCREASE_STEPS_MODE = False
LOADING = True
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

if __name__ == "__main__": 
    state_dim = (1, 3, 227, 227)
    action_dim = (1, 2)
    learning_rate_actor = LR_ACTOR # 0.03 initial learning rate
    learning_rate_critic = LR_CRITIC # 0.01 initial learning rate
    gamma = 0.9
    epochs = NUM_EPOCHS
    epsilon_clip = 0.2
    continuous_action_space = True
    action_std = .8
    
    model = PPO.PPO(state_dim, action_dim, learning_rate_actor, learning_rate_critic, gamma, epochs, epsilon_clip, continuous_action_space, action_std, LR_DECAY, 0)
    
    total_params = sum(p.numel() for p in model.policy_old.parameters())
    print(f"Total parameters: {total_params}")