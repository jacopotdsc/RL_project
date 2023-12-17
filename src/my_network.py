import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self,  env1_id, env1_input, env1_outputs, 
                        env2_id, env2_input, env2_outputs, 
                        env3_id, env3_input, env3_outputs, 
                        learning_rate, bias=False):


        super(Net, self).__init__()

        self.env1_id = env1_id
        self.env2_id = env2_id
        self.env3_id = env3_id

        self.env1_input = env1_input
        self.env2_input = env2_input
        self.env3_input = env3_input

        self.env1_outputs = env1_outputs
        self.env2_outputs = env2_outputs
        self.env3_outputs = env3_outputs

        # activation function
        self.relu    = nn.ReLU()
        self.tan     = nn.Tanh()
        self.softm   = nn.Softmax(dim=-1)

        # input layers
        self.input_env1 = nn.Linear(in_features=self.env1_input, out_features=32, bias=bias)
        self.input_env2 = nn.Linear(in_features=self.env1_input, out_features=32, bias=bias)
        self.input_env3 = nn.Linear(in_features=self.env1_input, out_features=32, bias=bias)

        # hidden layers                  
        self.hl1 = nn.Linear(32, 216 , bias=bias) 
        self.hl2 = nn.Linear(216, 216, bias=bias) 
        self.hl3 = nn.Linear(216, 32, bias=bias) 

        # output layers: one for each enviroment
        self.fl_out_env1 = nn.Linear(in_features=32, out_features=self.env1_outputs, bias=bias)
        self.fl_out_env2 = nn.Linear(in_features=32, out_features=self.env2_outputs, bias=bias)
        self.fl_out_env3 = nn.Linear(in_features=32, out_features=self.env3_outputs, bias=bias)


        # optimizer -> check how it work values
        self.optimizer = torch.optim.Adam(self.parameters(),lr=learning_rate)


    # return the action's probabilities
    def forward(self, x):

        q_val = self.q_val(x)
        probs = self.softm(q_val)

        return probs

    # return the q-value -> input = ( x, env_id )
    def q_val(self, my_input):

        x = my_input['state']
        env_id = my_input['env_id']

        if env_id == self.env1_id:
            x = self.input_env1(x)

        elif env_id == self.env2_id:
            x = self.input_env2(x)

        elif env_id == self.env3_id :
            x = self.input_env3(x)
        
        x = self.hl1(x)
        x = self.relu(x)
        x = self.hl2(x)
        x = self.relu(x)
        x = self.hl3(x)
        x = self.relu(x)
         
        if env_id == self.env1_id:
            x = self.fl_out_env1(x)

        elif env_id == self.env2_id:
            x = self.fl_out_env2(x)
            x = self.tan(x)

        elif env_id == self.env3_id :
            x = self.fl_out_env3(x)
        
        return x
    
# Plotting function
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