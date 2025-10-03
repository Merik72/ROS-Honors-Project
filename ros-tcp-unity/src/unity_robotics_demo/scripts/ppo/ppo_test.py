#!/usr/bin/env python
import rospy
import cv2
from ppo_environment import DingoEnv
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torchvision import transforms

# network ResNet-like
from ppo_network import ActorCriticResNet as ActorCriticNetwork
from ppo_train import PPOTrainer

import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

from utils import Untils 

def main():
    utils = Untils(device=DEVICE ) #, max_steps_rollout= max_steps_rollout )
    # Initialize the environment, Action space here is the number of Node in last layer NN
    env = DingoEnv(image_topic="/image_raw0", resolution=(64,64) , action_space =len( utils.action_space_real))
 
    # Initialize the model
    model = ActorCriticNetwork(input_channels=3, action_dim=env.action_space)
    # model.load_state_dict(torch.load("checkpoint/trained_ppo_dingo450ep.pth", weights_only=True))
    model.load_state_dict(torch.load("checkpoint/best_model.pt", weights_only=True))
 
    # print("Model: ", model)
    model = model.to(DEVICE)  

    rospy.loginfo("Starting PPO testing with DingoEnv...")
    #reset the robot to the origin

    for i in range(5):

        env.reset()
        time.sleep(0.5)

        obs = env.get_observation()

        # Expand dimensions (to add batch dimension) and convert to PyTorch tensor
        # obs_tensor = torch.tensor([obs], dtype=torch.float32, device=DEVICE)
        ep_reward = 0
        idx = 0
        # for _ in range(max_steps): 
        done = False
        max_step = 200
        while idx < max_step and not done:
            
            # logits, val = model(torch.tensor([obs], dtype=torch.float32,
            #                                 device= self.device))
            tmp_image =  utils.array_to_tensor(obs)
            logits, val = model(tmp_image.unsqueeze(0).to(DEVICE))
            

            act_distribution = Categorical(logits=logits)
            act = act_distribution.sample()
            act_log_prob = act_distribution.log_prob(act).item()

            act, val = act.item(), val.item()

            # next_obs, reward, done, _ , _ = env.step(act)

            # print(f"Action like:  {act} |  {self.action_space_real[act]}")
            act_real =  utils.action_space_real[act]

            # next_obs, reward, done = env.step(act)
            next_obs, reward, done = env.step(act_real)

            # for i, item in enumerate((obs, act, reward, val, act_log_prob)):
            #     train_data[i].append(item)

            obs = next_obs  # x , x' , omega, omega'
            ep_reward += reward

            if done:
                print("Just get to the point..................") 
                env.reset()
                #after rolling out, just stop, and waits
                action = [0.0, 0.0]
                env.step(action) 
                time.sleep(0.5)
                # break
            idx+=1

    
        rospy.loginfo(f"Episode {i + 1} | Reward : {ep_reward} ")

    action = [0.0, 0.0]
    env.step(action)    
    time.sleep(0.5)
    env.reset()
    rospy.loginfo("Testing completed.")
    
  

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted.")
    except KeyboardInterrupt:
        rospy.loginfo("Program terminated by user.")

    
 



