 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical
from torchvision import transforms
import cv2

import time
import shutil
class Untils():
    def __init__(self, device):
        self.device = device 
        self.action_space_real = self.action_space_sample()


        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert NumPy array to PIL Image
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def array_to_tensor(self, numpy_array_cv2):       
        """ 
        Input: image from ROS
        Return :[0, 1] tensor
        
        """ 
        # Transform the NumPy array into a tensor
        img = cv2.cvtColor(numpy_array_cv2, cv2.COLOR_BGR2RGB)

          # Convert uint16 to uint8 if necessary
        # if img.dtype == np.uint16:
        #     img = (img / 256).astype(np.uint8)  # Scale down to 8-bit range [0, 255]
        #     print( "16-bittttttttttttt")

        return self.transform(img)

    def action_space_sample(self,min_velocity = 0.0 , max_velocity = 1, step_size = .1):
        
        # Define the velocity range and step size
     

        # Create the list of possible velocities
        velocities = np.arange(min_velocity, max_velocity + step_size, step_size , dtype=float)

        # Create the action space as a list of tuples (left_velocity, right_velocity)
        return [(round(left, 1), round(right,1)) for left in velocities for right in velocities]
        

    def discount_rewards(self, rewards, gamma=0.99):
        """
        Return discounted rewards based on the given rewards and gamma param.
        """
        new_rewards = [float(rewards[-1])]
        for i in reversed(range(len(rewards)-1)):
            new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
        return np.array(new_rewards[::-1])

    def calculate_gaes(self, rewards, values, gamma=0.99, decay=0.97):
        """
        Return the General Advantage Estimates from the given rewards and values.
        Paper: https://arxiv.org/pdf/1506.02438.pdf
        """
        next_values = np.concatenate([values[1:], [0]])
        deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

        # print(f"Deltas len: { len(deltas)}")

        # print( f"GAES size of deltas {len(deltas)} | reward: {rewards} | reward len: {len(rewards)} |  Value: {len(values)} | next vl: {next_values}")
        gaes = [deltas[-1]]
        for i in reversed(range(len(deltas)-1)):
            gaes.append(deltas[i] + decay * gamma * gaes[-1])

        return np.array(gaes[::-1])

    def rollout(self, model, env, max_steps = 1000, imshow_while_training = False ):
        """
        Performs a single rollout.
        Returns training data in the shape (n_steps, observation_shape)
        and the cumulative reward.
        """ 
        ### Create data storage
        train_data = [[], [], [], [], []] # obs, act, reward, values, act_log_probs
        
        # obs, info = env.reset()
        obs = env.get_observation()

        # Expand dimensions (to add batch dimension) and convert to PyTorch tensor
        # obs_tensor = torch.tensor([obs], dtype=torch.float32, device=DEVICE)

        ep_reward = 0
        idx = 0
        # for _ in range(max_steps): 
        done = False
        while idx <= max_steps and not done:
            
            # logits, val = model(torch.tensor([obs], dtype=torch.float32,
            #                                 device= self.device))
            tmp_image =  self.array_to_tensor(obs)
            logits, val = model(tmp_image.unsqueeze(0).to(self.device))
            

            act_distribution = Categorical(logits=logits)
            act = act_distribution.sample()
            act_log_prob = act_distribution.log_prob(act).item()

            act, val = act.item(), val.item()

            # next_obs, reward, done, _ , _ = env.step(act)

            # print(f"Action like:  {act} |  {self.action_space_real[act]}")
            act_real =  self.action_space_real[act]

            # next_obs, reward, done = env.step(act)
            next_obs, reward, done = env.step(act_real)

            for i, item in enumerate((obs, act, reward, val, act_log_prob)):
                train_data[i].append(item)

            # print(f"Reward: {reward} -----------------")
            obs = next_obs  # x , x' , omega, omega'
            ep_reward += reward

            if done:
                print("Just get to the point..................")
                env.reset()
                #after rolling out, just stop, and waits
                action = [0.0, 0.0]
                env.step(action) 
                time.sleep(0.01)
                # break
            idx+=1

            if imshow_while_training == True:
                if obs is not None:
                    # print( len(obs))
                    cv2.imshow("Dingo Observation", obs)
                    cv2.waitKey(1)
        # print("---------------------" , train_data)    

        train_data = [np.asarray(x) for x in train_data]

        # print(f" Train data [2] :{len( train_data[2])} | train_date[3]: {len(train_data[3])}")

        ### Do train data filtering
        train_data[3] = self.calculate_gaes(train_data[2], train_data[3])

        return train_data, ep_reward
    



