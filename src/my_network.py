import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self, n_inputs, env1_outputs, env2_outputs, env3_outputs, learning_rate, bias=False):
        super(Net, self).__init__()

        self.n_inputs= n_inputs

        self.env1_outputs = env1_outputs
        self.env2_outputs = env2_outputs
        self.env3_outputs = env3_outputs

        # activation function
        self.relu    = nn.ReLU()
        self.pool    = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.softm   = nn.Softmax(dim=-1)

        # layers
        self.cn1 = nn.Conv2d(in_channels=self.n_inputs, out_channels=6, kernel_size=7, stride=1, bias=bias)
        self.cn2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4, stride=1, bias=bias)                     
        self.fl1 = nn.Linear(5292, 216, bias=bias) 

        # output layers: one for each enviroment
        self.fl_half_cheetah = nn.Linear(216, self.env1_outputs, bias=bias)
        self.fl_human_walk   = nn.Linear(216, self.env2_outputs, bias=bias)
        self.fl_multi_agent  = nn.Linear(216, self.env3_outputs, bias=bias)


        # optimizer -> check how it work values
        self.optimizer = torch.optim.Adam(self.parameters(),lr=learning_rate)


        self.half_cheetah_id = 'half cheetah'
        self.human_walk_id   = 'human walk'
        self.multi_agent_id  = 'multi agent'

    # return the action's probabilities
    def forward(self, x):
        
        x = self.q_val(x)
        x = self.softm(x)

        return x

    # return the q-value -> input = ( x, env_id )
    def q_val(self, input):

        x = input[0]
        env_id = input[1]
        
        x = self.cn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.cn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)

        x = self.fl1(x)
        x = self.relu(x)
         
        if env_id == self.half_cheetah_id:
            x = self.fl_half_cheetah(x)

        if env_id == self.human_walk_id:
            x = self.fl_human_walk(x)

        if env_id == self.multi_agent_id :
            x = self.fl_multi_agent(x)
        
        return x
    
# Plotting function
def plot_training_rewards(agent):
        cumulative_mean = np.cumsum(agent.training_reward ) / len(agent.training_reward )
        plt.plot(cumulative_mean)
        plt.title('Mean training rewards')
        plt.ylabel('Reward')
        plt.xlabel('Episods')
        #plt.show()
        plt.savefig('mean_training_rewards.png')
        plt.clf()

def plot_training_loss(agent):
        cumulative_mean = np.cumsum(agent.training_loss) / len(agent.training_loss)
        plt.plot(cumulative_mean)
        plt.title('Mean training loss')
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
        plt.savefig('episode_rewards.png')
        plt.clf()