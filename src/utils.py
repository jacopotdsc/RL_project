import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch

import random
from collections import namedtuple, deque



class Buffer:

    def __init__(self, agent, memory_size=500, batch_size=32):

        self.agent = agent
        self.batch_size = batch_size
        self.memory_size = memory_size
        
        self.buffer_env1  = deque(maxlen=memory_size)
        self.buffer_env2  = deque(maxlen=memory_size)
        self.buffer_env3  = deque(maxlen=memory_size)
        
        
    def add(self, state, env_id):

        #self.replay_memory.append( self.Buffer(state) ) 
        
        if env_id == self.agent.env1_id: self.buffer_env1.append(state)
        elif env_id == self.agent.env2_id: self.buffer_env2.append(state)
        elif env_id == self.agent.env3_id: self.buffer_env3.append(state)


    def sample(self, env_id):
        
        print(f"sampling from: {env_id}")
        if env_id == self.agent.env1_id: 
             print(f" entered {env_id}")
             samples = random.sample(self.buffer_env1, self.batch_size)

        elif env_id == self.agent.env2_id: 
             print(f" entered {env_id}")
             samples = random.sample(self.buffer_env2, self.batch_size)
             
        elif env_id == self.agent.env3_id: 
             print(f" entered {env_id}")
             samples = random.sample(self.buffer_env3, self.batch_size)

        return samples
    
    def buffer_size(self, env_id):

        if env_id == self.agent.env1_id: 
             return len(self.buffer_env1)

        elif env_id == self.agent.env2_id: 
             return len(self.buffer_env2)
        
        elif env_id == self.agent.env3_id: 
             return len(self.buffer_env3)

def preprocess_image(state):
    state = torch.tensor(state)

    return state

def plot_training_rewards(agent):
        cumulative_mean = np.cumsum(agent.training_reward ) / len(agent.training_reward )
        plt.plot(cumulative_mean)
        plt.title('Mean training rewards')
        plt.ylabel('Reward')
        plt.xlabel('Episods')
        #plt.show()
        plt.savefig('plot/mean_training_rewards.png')
        plt.clf()

def plot_training_loss(agent):
        cumulative_mean = np.cumsum(agent.training_loss) / len(agent.training_loss)
        plt.plot(cumulative_mean)
        plt.title('plot/Mean training loss')
        plt.ylabel('loss')
        plt.xlabel('timestep')
        #plt.show()
        plt.savefig('mean_training_loss.png')
        plt.clf()

def plot_episode_reward(agent):
    plt.plot(agent.reward_episode)
    plt.title('Rewards')
    plt.ylabel('rewards')
    plt.xlabel('timestep')
    #plt.show()
    plt.savefig('plot/episode_rewards.png')
    plt.clf()