#!/usr/bin/env python
# using tensordict

from math import gamma
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
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
import torchrl

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
    
    # this is where training happens
    def update(self):

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
    
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        while rewards.size() != old_state_values.size():
            while (rewards.size() > old_state_values.size()):
                rewards = rewards[:-1]
            while (old_state_values.size() > rewards.size()):
                old_state_values = old_state_values[:-1]

        # advantages = rewards.detach() - old_state_values.detach()
        advantages = torch.tensor(self.calculate_gaes(rewards.cpu(), old_state_values.cpu())).to(device)

        permute_idxs = np.random.permutation(len(self.buffer.states))
        old_states = torch.tensor(old_states[permute_idxs], dtype=torch.float32, device=device)
        old_actions = torch.tensor(old_actions[permute_idxs], dtype=torch.float32, device=device)
        old_logprobs = torch.tensor(old_logprobs[permute_idxs], dtype=torch.float32, device=device)
        old_state_values = torch.tensor(old_state_values[permute_idxs], dtype=torch.float32, device=device)
        advantages = torch.tensor(advantages[permute_idxs], dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards[permute_idxs], dtype=torch.float32, device=device)


        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            self.optimizer.zero_grad()

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            #print(logprobs.detach().mean())
            #print(old_logprobs.detach().mean())
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            surr1 = ratios * advantages

            # final loss of clipped objective PPO
            # loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            loss = -torch.min(surr1, surr2)
            #print(loss.detach().mean())

            # take gradient stepw
            loss.mean().backward()#(torch.Tensor([1,1]))
            self.optimizer.step()
            if (old_logprobs.detach() - logprobs).mean() >= 0.01: 
                break
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Work this thingy out
        if self.epoch % 150 == 0:
            self.save("/home/admin2/ros-tcp-unity/src/unity_robotics_demo/scripts/savedModels/12-3/old.pt")

        # clear buffer
        self.buffer.clear()
        self.epoch += 1
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage)) 