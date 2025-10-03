#!/usr/bin/env python

import ActorCritic
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
LR_INIT = 0.01
LR_FLOOR = 1e-6
# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6, lr_decay=0.5, epoch=0):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lr_decay = lr_decay
        self.epoch = epoch
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic.ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        self.policy_old = ActorCritic.ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def parallel_happy_select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)
        return action.detach().cpu().numpy().flatten(), action_logprob, state_val, state, action
    
    
    def set_values(self, state, action, action_logprob, state_val, reward):
        self.buffer.states = state
        self.buffer.actions = action
        self.buffer.logprobs = action_logprob
        self.buffer.state_values = state_val
        self.buffer.rewards = reward
        
    def push_values(self, state, action, action_logprob, state_val, reward, done):
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)
    
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()

    # this is where training happens
    def update(self):
        # learning rate decay
        #self.lr_actor = LR_INIT/(1+self.lr_decay*self.epoch)
        #self.lr_critic = LR_INIT/(1+self.lr_decay*self.epoch)
        #print("Current learning rate is: ", self.lr_actor)

        # Monte Carlo estimate of returns
        # print(len(self.buffer.rewards))

        # rewards = []

        # if self.buffer.is_terminals[-1]:
        #     discounted_reward = 0
        # else:
        #     discounted_reward = self.policy_old.critic(self.buffer.states[-1]).item()

        # for reward in reversed(self.buffer.rewards):
        #     discounted_reward = reward + (self.gamma * discounted_reward)
        #     rewards.insert(0, discounted_reward)

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        
        # rewards having a negative mean
        #print('\t |rewards shape ', rewards.shape, ". CLOSE")
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        print("Average reward for this step: ", rewards.mean().detach().to('cpu').item())

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        # print(len(self.buffer.rewards))
        # print(len(self.buffer.states))
        # print(len(self.buffer.actions))
        # print(len(self.buffer.logprobs))
        # print(len(self.buffer.state_values))
        

        #print('rewardlength: ', rewards.size(), ". CLOSE")
        # calculate advantages
        # print('\t |Actions: ', self.buffer.actions, ". CLOSE")
        
        #print('\t |rewards size ', rewards.size(), ". CLOSE")
        #print('\t |old_states vales ', old_state_values.size(), ". CLOSE")
        while rewards.size() != old_state_values.size():
            #print('AAAAAAAAAAAAAAAAAA')
            while (rewards.size() > old_state_values.size()):
                #print("Rewards > Old Rewards")
                rewards = rewards[:-1]
            while (old_state_values.size() > rewards.size()):
                #print("Old Rewards > Rewards")
                old_state_values = old_state_values[:-1]
                #print("old_state_values.size(): ", old_state_values.size())
        # print(len(self.buffer.rewards))
        # print(len(self.buffer.states))
        # print(len(self.buffer.actions))
        # print(len(self.buffer.logprobs))
        # print(len(self.buffer.state_values))
        # how do i fix the size of these tensors??
        # Doesn't work
        # while (old_states[0].size() > rewards.size()):
        #     print("old_states > Rewards")
        #     old_states[0] = old_states[0][:-1]
        # print("old_states.size(): ", old_states.size())
        # while (old_actions[0].size() > rewards.size()):
        #     print("old_actions > Rewards")
        #     old_actions[0] = old_actions[0][:-1]
        # print("old_actions.size(): ", old_actions.size())

        advantages = rewards.detach() - old_state_values.detach()

        '''print("advantages:\n\t|", advantages)
        print("old_state_values:\n\t|", old_state_values)
        print("rewards:\n\t|", rewards)
        print("actions:\n\t|", old_actions)'''

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            print(logprobs)
            print(old_logprobs)
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs.mean() - old_logprobs.detach().mean())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            print(loss)

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Work this thingy out
        if self.epoch % 150 == 0:
            self.save("/home/admin2/ros-tcp-unity/src/unity_robotics_demo/scripts/savedModels/name.pt")

        # clear buffer
        self.buffer.clear()
        self.epoch += 1
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))