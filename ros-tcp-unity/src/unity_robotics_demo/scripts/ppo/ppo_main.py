#!/usr/bin/env python
from sympy import true
import rospy
import cv2
from ppo_environment import DingoEnv
import numpy as np
import torch
import os.path

from tqdm import trange

# network ResNet-like
from ppo_network import ActorCriticResNet as ActorCriticNetwork
from ppo_train import PPOTrainer

import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
CHECKPOINT_DIR =  '/home/admin2/ros-tcp-unity/src/unity_robotics_demo/scripts/ppo/checkpoint'

from utils import Untils 

CAMERA_TOPIC = '/image_raw_1'
backup_freq = 100
LOADING = True

def main():
    utils = Untils(device=DEVICE ) #, max_steps_rollout= max_steps_rollout )
    # Initialize the environment, Action space here is the number of Node in last layer NN
    env = DingoEnv(image_topic=CAMERA_TOPIC, resolution=(64,64) , action_space =len( utils.action_space_real))
    
    # print(utils.action_space_real)
    # exit()


    # Initialize the model
    model = ActorCriticNetwork(input_channels=3, action_dim=env.action_space)
    start_epoch = 0
    path  = f"{CHECKPOINT_DIR}/best_model.pt"
    # print("Model: ", model)


    # to load
    if(LOADING):
        if os.path.isfile(path):
            print("Loading the pretrained ......")
            model, start_epoch = model.load_ckp(path, model) 
            print(f"Starting epoch: {start_epoch}")
        else:
            print("File not found")

    # print("Model: ", model)
    

    # to load    
    # CHECKPOINT_DIR = "checkpoint"
    # path  = f"{CHECKPOINT_DIR}/best_model.pt"
    # if os.path.isfile(path):        
    #     print("Loading the pretrained ......")
    #     model = model.load_ckp_model(model=model )
          
    model = model.to(DEVICE)

    # train_data, reward = utils.rollout(model, env) # Test rollout function
    # rospy.loginfo("Just tested rollout function with DingoEnv...")
    # exit()
    
    ppo = PPOTrainer(   model,
                        policy_lr = 3e-4*1,
                        value_lr = 1e-3*1,
                        target_kl_div = 0.02,
                        max_policy_train_iters = 100,
                        value_train_iters = 100,
                        batch_size=1024)
    
    # Ensure the model is in CUDA
    # assert all(param.is_cuda for param in model.parameters()), "Model is not fully on CUDA!"
    # print("Model is on CUDA.")
    # exit()

    # Example PPO Hyperparameters (to be replaced with actual PPO integration)

    rospy.loginfo("Starting PPO training with DingoEnv...")
    #reset the robot to the origin
    env.reset()
          
    # policy_logits, value = model(processed_img.unsqueeze(0).to(DEVICE))
    # # Print the outputs 
    #  4.1699 2.748 6.874 position of waypoints

    max_episodes = 1500
    ep_rewards = []
    max_reward = -99999
     # Define training params 
    print_freq = 20
    reset_freq = 1

    # for episode in range(max_episodes):
    # for episode in trange(max_episodes, desc="Progress:", unit="%"):
    progress_bar = trange(max_episodes, desc="Progress:", unit="ep")
    start_time = time.time()
    for episode in progress_bar:
 
         
        # rospy.loginfo(f"Episode {episode + 1}/{max_episodes}") 
     
        #     # Update the observation
        # try 1000 steps, then stop for the training progress...
        train_data, reward = utils.rollout(model, env, max_steps= 700 , imshow_while_training= True)  

        if (episode +1) % reset_freq ==0:
            env.reset() 
 
        
        action = [0.0, 0.0]
        #after rolling out, just stop, and waits
        env.step(action)

        ep_rewards.append(reward)

        # Shuffle
        # permute_idxs = np.random.permutation(len(train_data[0]))
        permute_idxs = np.random.permutation(len(train_data[0])-1)

        # Policy data
        obs = torch.tensor(train_data[0][permute_idxs],
                            dtype=torch.float32, device=DEVICE)
        acts = torch.tensor(train_data[1][permute_idxs],
                            dtype=torch.int32, device=DEVICE)
        
        
        gaes = torch.tensor(train_data[3][permute_idxs],
                            dtype=torch.float32, device=DEVICE)
        
        act_log_probs = torch.tensor(train_data[4][permute_idxs],
                                    dtype=torch.float32, device=DEVICE)

        # Value data
        returns = utils.discount_rewards(train_data[2])[permute_idxs]
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        # Train model
        ppo.train_policy(obs, acts, act_log_probs, gaes)
        ppo.train_value(obs, returns)

        #if (episode + 1) % print_freq == 0:
        #     print('Episode {} | Avg Reward {:.1f}'.format(
        #         episode + 1, np.mean(ep_rewards[-print_freq:])))
        
        is_best = False 
        if reward > max_reward:
            max_reward  = reward
            is_best = True
        if is_best:
            checkpoint = {
                'epoch': episode + start_epoch + 1,
                'state_dict': model.state_dict()
            }
            name = f"checkpoint_EP__{episode}__AT__{time.time()}.pt" 
            model.save_ckp(checkpoint, is_best, CHECKPOINT_DIR, name)
        if episode % backup_freq == 0:
            state = model.state_dict()
            
            
            # file_name = f"checkpoint_{episode}.pt"
            # model.save_ckp_model(  state= state,
            #                         CHECKPOINT_DIR=CHECKPOINT_DIR,
            #                         file_name=file_name,
            #                         is_best=is_best)
             
            checkpoint = {
                'epoch': episode + start_epoch + 1,
                'state_dict': state
            }
            name = f"checkpointEP{episode}AT{time.time()}.pt" 
            model.save_ckp(checkpoint, is_best, CHECKPOINT_DIR, name)

        # rospy.loginfo(f" Reward : { round(reward, 1)} | Time: {round( time.time() - start_time , 1)}(s)")
        # print("numbebr ò  steps: ", len( train_data[0]) )
        progress_bar.set_postfix({"reward": f"{reward:.1f} | steps: {len( train_data[0])}"})
        
 
    #STOP  and reset
    action = [0.0, 0.0]
    env.step(action) 

    rospy.loginfo("Training completed.")
    rospy.loginfo(f" Time: {time.time() - start_time}")
     
if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted.")
    except KeyboardInterrupt:
        rospy.loginfo("Program terminated by user.")
