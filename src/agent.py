import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import random
import time

from my_network import *


class Agent(nn.Module):

    def __init__(   self, env = None, gamma = 0.95, 
                    epsilon = 1.0, epsilon_min = 0.2, epsilon_decay  = 0.99,
                    learning_rate   = 0.001 ):
        
        super(Agent, self).__init__()

        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # hyperparameters
        self.gamma           = gamma
        self.epsilon         = epsilon
        self.epsilon_min     = epsilon_min
        self.epsilon_decay   = epsilon_decay
        self.learning_rate   = learning_rate
        self.loss_function   = nn.MSELoss()


        # list of enviroment needed for training
        self.evaluation_env = None 

        self.env1.id_ = 'CarRacing-v2'
        self.env1 = gym.make('CarRacing-v2', continuous=False) # gym.make('HalfCheetah-v2')
        self.env1_n_action = self.env1.action_space.n

        self.env2.id_ = 'CarRacing-v2'
        self.env2 = gym.make('CarRacing-v2', continuous=False) # gym.make('HumanoidSmallLeg-v0')   
        self.env2_n_action = self.env1.action_space.n

        self.env3.id_ = 'CarRacing-v2'
        self.env3 = gym.make('CarRacing-v2', continuous=False)# gym.make('RoboschoolHumanoid-v1')   
        self.env3_n_action = self.env1.action_space.n

        self.env_array = { self.env1_id: self.env1, 
                           self.env2_id: self.env2, 
                           self.env3_id: self.env3 }


        # Network
        self.model           = Net(self.frame_stack_num, 
                                   self.evaluation_env,
                                   self.env1_n_action, 
                                   self.env2_n_action, 
                                   self.env3_n_action, 
                                   self.learning_rate).to(self.device) 

        # for plotting
        self.training_reward = []
        self.training_loss   = []
        self.reward_episode  = []

        print(self.model)
        print(f"Device: {self.device}")


    # return action index -> used in evaluation
    def forward(self, x):
        n_actions = self.evaluation_env.action_space.n

        if n_actions == 6:
            half_cheetah_id = self.model.half_cheetah_id
            action = self.model.forward(x, half_cheetah_id)

        elif n_actions == 17:
            human_walk_id = self.model.human_walk_id
            action = self.model.forward(x, human_walk_id)

        else:
            multi_agent_id = self.model.multi_agent_id 
            action = self.model.forward(x, multi_agent_id)

        return action

    # TODO
    def act(self, state, env_id):
        
        if np.random.rand() > self.epsilon:

            model_input = ( state, env_id)
            act_values = self.forward( model_input )
            action = torch.argmax(act_values[0])

        else:

            # it vary based on enviroment id
            action = random.randrange(self.action_space)
            # action = np.random.uniform(low=-1, high=1, size=3)

        return action

    # training loop
    def train(self):
       
       
        max_epochs = 400
        switch_env_frequency = 50

        old_mean_reward = -999
        start_time = time.perf_counter()

        for e in range(max_epochs):

            # create dictionary for switching easier between states of enviroments
            env_states = {}
            for env_id in self.env_array.keys():
                init_state, _ = self.env_array[env_id].reset()
                env_states[env_id] = init_state
                

            steps  = 0
            window = 50
            total_reward = 0
            done = False                

            while True:

                action = self.act(state)

                reward = 0
                
                state, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated  # truncated if fail to stand up -> penalize!

                reward += r
                
                total_reward += reward
                steps += 1

                # plot statstics
                if done:
                
                    self.training_reward.append(total_reward)
                    self.reward_episode.append(total_reward)

                    mean_rewards = np.mean(self.training_reward[-window:])
                    mean_loss    = np.mean(self.training_loss[-window:])
                    
                    print("\nEpisode {:d}/{:d} Step: {:d} Mean Rewards {:.2f}  Episode reward = {:.2f}  lr: {:.5f} e: {:.3f} mean loss = {:.3f}\t\t".format(
                                e, max_epochs,steps, mean_rewards, total_reward,  self.learning_rate, self.epsilon, mean_loss))


                    self.epsilon *= self.epsilon_decay

                    if self.epsilon < self.epsilon_min:
                        self.epsilon = self.epsilon_min


                    plot_training_rewards(self)
                    plot_training_loss(self)
                    plot_episode_reward(self)

                    actual_mean_reward = np.mean(self.training_reward[-window:]) 

                    if actual_mean_reward > old_mean_reward:
                        print("Better model! old_mean_reward: {:.2f}, new_mean_reward: {:.2f}".format(old_mean_reward, actual_mean_reward))
                        old_mean_reward = actual_mean_reward
                        self.save('end_episode.pt')

                    break

           
            plot_training_rewards(self)
            plot_training_loss(self)
            plot_episode_reward(self)

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            print("Elapsed time: {:.2f} seconds".format(elapsed_time))
            print("Elapsed time: {:.2f} minutes".format(elapsed_time/60))
            print()

            self.save('model.pt')

    # how calculate loss
    def calculate_loss(self, x):
        return x


    # Utility functions
    def save(self, name = 'model.pt' ):
        torch.save(self.state_dict(), name )

    def load(self, name = 'model.pt'):
        self.load_state_dict(torch.load(name, map_location = self.device))
         
    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret