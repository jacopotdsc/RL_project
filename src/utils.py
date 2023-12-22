import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch

import random
from collections import namedtuple, deque

import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler


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
        
        if env_id == self.agent.env1_id: 
             samples = random.sample(self.buffer_env1, self.batch_size)

        elif env_id == self.agent.env2_id: 
             samples = random.sample(self.buffer_env2, self.batch_size)
             
        elif env_id == self.agent.env3_id: 
             samples = random.sample(self.buffer_env3, self.batch_size)

        return samples
    
    def buffer_size(self, env_id):

        if env_id == self.agent.env1_id: 
             return len(self.buffer_env1)

        elif env_id == self.agent.env2_id: 
             return len(self.buffer_env2)
        
        elif env_id == self.agent.env3_id: 
             return len(self.buffer_env3)

class RBFFeatureEncoder:
    def __init__(self, env1, env2, env3, n_component=100): 

        self.env1 = env1
        self.env2 = env2
        self.env3 = env3

        data1 = np.array([env1.observation_space.sample() for x in range(10000)]); data1 = np.clip(data1, -1e38, 1e38)
        data2 = np.array([env2.observation_space.sample() for x in range(10000)]); data2 = np.clip(data2, -1e38, 1e38)
        data3 = np.array([env3.observation_space.sample() for x in range(10000)]); data3 = np.clip(data3, -1e38, 1e38)

        self.rbf_sampler1 = RBFSampler(gamma=0.999, n_components=n_component)
        self.rbf_sampler2 = RBFSampler(gamma=0.999, n_components=n_component)
        self.rbf_sampler3 = RBFSampler(gamma=0.999, n_components=n_component)
        
        self.standard_scaler1 = sklearn.preprocessing.StandardScaler()
        self.standard_scaler2 = sklearn.preprocessing.StandardScaler()
        self.standard_scaler3 = sklearn.preprocessing.StandardScaler()

        self.standard_scaler1.fit(data1) 
        self.standard_scaler2.fit(data2) 
        self.standard_scaler3.fit(data3) 

        transformed_data1 = self.standard_scaler1.transform(data1)
        transformed_data2 = self.standard_scaler2.transform(data2)
        transformed_data3 = self.standard_scaler3.transform(data3)
        
        self.rbf_sampler1.fit( transformed_data1 )
        self.rbf_sampler2.fit( transformed_data2 )
        self.rbf_sampler3.fit( transformed_data3 )
        

    def encode(self, model, model_input): # modify
        
        state = model_input['state']
        env_id = model_input['env_id']

        if model.env1_id == env_id:
            transformed_state = self.standard_scaler1.transform([state])
            encoded_state = self.rbf_sampler1.transform(transformed_state).flatten()

        elif model.env2_id == env_id:
            transformed_state = self.standard_scaler2.transform([state])
            encoded_state = self.rbf_sampler2.transform(transformed_state).flatten()

        elif model.env3_id == env_id:
            transformed_state = self.standard_scaler3.transform([state])
            encoded_state = self.rbf_sampler3.transform(transformed_state).flatten()

        return torch.tensor(encoded_state, dtype=torch.float32) 

    @property
    def size(self): # modify
        # TODO return the number of features
        return self.rbf_sampler1.n_components 

def preprocess(model, model_input):

    #return torch.tensor(model_input['state'])
    state = model.encoder.encode(model, model_input)

    return state.clone().detach()

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