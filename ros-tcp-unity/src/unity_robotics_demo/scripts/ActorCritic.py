#!/usr/bin/env python

import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal, Categorical
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
NUM_CLASSES = 2
# BATCH_SIZE = 1 # make sure this matches with ppo_batches...

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()
        print("Alexnet is using this device: ", device)
        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full(self.action_dim, action_std_init * action_std_init).to(device)
        # actor
        # https://d2l.ai/chapter_convolutional-modern/alexnet.html
        #if has_continuous_action_space :
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.LeakyReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.LeakyReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.LeakyReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.LeakyReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
            nn.Flatten(),  # Flatten the output
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Linear(in_features=4096, out_features=1000),
            nn.Sigmoid(),
            nn.Linear(in_features=1000, out_features=NUM_CLASSES),
            nn.Tanh(),
        )
       
        # critic
        self.critic = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
                nn.LeakyReLU(),
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
                nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
                nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
                nn.LeakyReLU(),
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
                nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
                nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
                nn.LeakyReLU(),
                nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
                nn.LeakyReLU(),
                nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
                nn.Flatten(),  # Flatten the output
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(in_features=4096, out_features=4096),
                nn.Linear(in_features=4096, out_features=1000),
                nn.Sigmoid(),
                nn.Linear(in_features=1000, out_features=1),
                nn.Tanh(),
                )
    

        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full(self.action_dim, new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        # print("in if")
        # state = state[None, :, :, :]
        # print(state.shape)
        action_mean = self.actor(state)
        # print("\naction mean: ", action_mean)
        
        # print("self.action_var: ", self.action_var.unsqueeze())
        cov_mat = torch.diag(self.action_var.squeeze()).to(device) # torch.diag(self.action_var).unsqueeze(dim=1)
        # print("cov_mat: ", cov_mat)
        # print(action_mean, cov_mat)
        distLinear = MultivariateNormal(action_mean, cov_mat)
        # print("distribution: ", distLinear)
        '''# print("escaped ", dist.mean.shape[0])
        if dist.mean.shape[0] == 1:
            #print("in 1")
            action = action_mean
            #print("action sampled: ", action)
            action_logprob = action
            #print("logbprodb : ", action)
            state_val = self.critic(state)
            #print("statevale: ", state_val)
            return action, action_logprob, state_val
        else:'''
        #print("action sampled: ", action)
        action = distLinear.sample()
        # print(action)
        action_logprob = distLinear.log_prob(action)
        #print("probs logged")
        state_val = self.critic(state)
        #print("returning next")
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        # action_logprobs = action
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy