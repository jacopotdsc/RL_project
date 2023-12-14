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

        # it contain  actual state for each enviroment
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

    def swtich_enviroment(self, actual_id_env, env_ids, env_done, env_index):

        # support variable: env_index, env_ids, env_done
        # changed: actual_id_env, actual_env

        # actual_id_env: actual id of the enviroment
        # env_ids: array with all keys ( names ) of enviroment
        # env_states: actual state of the actual enviroment
        # env_done: dictionary which contain done-boolean
        # env_index: general counter

        switched = False

        while not switched:

            env_index += 1
            if env_done[ actual_id_env ] == False:

                switched = True
                actual_id_env = env_ids[ env_index % len(env_ids) ]    # env_ids has key for dictionary
                actual_env = self.env_array[ actual_id_env ]   # picking the state of the enviroment

        return actual_env, actual_id_env, env_index

    # training loop
    def train(self):
       
       
        max_epochs = 200
        switch_env_frequency = 50
        window = 50 

        old_mean_reward = -999
        start_time = time.perf_counter()

        for e in range(max_epochs):

            # create dictionary for switching easier between states of enviroments
            env_states = {}     # contain state to pass when callicng act
            env_step   = {}     # counter for switch
            env_done   = {}     # contain "done" enviroment
            env_reward = {}     # counter for reward
            for env_id in self.env_array.keys():

                init_state, _ = self.env_array[env_id].reset()

                env_states[env_id] = init_state
                env_done[env_id]   = False
                env_step[env_id]   = 0
                env_reward[env_id] = 0
    

            env_ids       = self.env_array.keys()  # contain id of all enviroment
            actual_id_env = env_ids[env_ids]       # contain id of the actual enviroment
            actual_env    = self.env_array[actual_id_env]  # contain the state of the selected enviroment accordin to id

            env_index  = 0      # used for switch
            done_counter = 0    # use to terminate episode

            while True:

                if env_step[actual_id_env] % switch_env_frequency == 0:
                    actual_env, actual_id_env, env_index = self.swtich_enviroment( actual_id_env, env_ids, env_done, env_index  )

                state = env_states[ actual_id_env ]
                action = self.act(state, actual_env)
                
                next_state, reward, terminated, truncated, _ = actual_env.step(action)
                env_states[ actual_id_env ] = next_state
                done = terminated or truncated  # truncated if fail to stand up -> penalize!
                

                
                env_reward[actual_id_env] += reward
                env_step[actual_id_env] += 1

                # plot statstics
                if done:

                    done_counter += 1
                    env_done[actual_id_env] = True
                
                    total_reward = np.array(env_reward.values).sum()
                    total_steps = np.array(env_step.values).sum()

                    self.training_reward.append(total_reward)
                    self.reward_episode.append(total_reward)

                    mean_rewards = np.mean(self.training_reward[-window:])
                    mean_loss    = np.mean(self.training_loss[-window:])
                    
                    print("\nEpisode {:d}/{:d} Step: {:d} Mean Rewards {:.2f}  Episode reward = {:.2f}  mean loss = {:.3f}\t\t".format(
                                e, max_epochs, total_steps, mean_rewards, total_reward, mean_loss))
                    
                    print("lr: {:.5f} e: {:.3f} \t\t".format( self.learning_rate, self.epsilon))



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

                    if done_counter >= len(self.env_array):
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