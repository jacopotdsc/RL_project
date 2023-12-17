import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import random
import time
#import mujoco_py

from my_network import *
from calculate_loss_function import *


###
# from baselines.common.cmd_util import mujoco_arg_parser, make_mujoco_env
# "RoboSumo-Ant-vs-Ant-v0"
#from gym_extensions.continuous import mujoco
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
        self.evaluation_env = env # used during evaluation: it contian the id of the enviroment ( maybe do a gym.make(env) )

        self.render = None #'human' # None
        self.action_type_discrete   = 'Discrete'
        self.action_type_continuous = 'Continuous'

        self.env1_id     = 'LunarLander-v2'                                 # https://www.gymlibrary.dev/environments/box2d/lunar_lander/
        self.env1        = gym.make(self.env1_id, render_mode=self.render)  # gym.make('HalfCheetah-v2')
        obs1_shape       = self.env1.observation_space.shape
        self.env1_input  = obs1_shape[0] if len(obs1_shape) == 1 else obs1_shape[0]*obs1_shape[1]
        self.env1_action = self.env1.action_space.n if type(self.env1.action_space)== gym.spaces.discrete.Discrete else self.env1.action_space.shape[0]
        self.env1_type_a = self.action_type_discrete if type(self.env1.action_space)== gym.spaces.discrete.Discrete else self.action_type_continuous
        
        
        self.env2_id     = 'BipedalWalker-v3'                               # https://www.gymlibrary.dev/environments/box2d/bipedal_walker/
        self.env2        = gym.make(self.env2_id, render_mode=self.render)  # gym.make('HumanoidSmallLeg-v0')  
        obs2_shape       = self.env2.observation_space.shape
        self.env2_input  = obs2_shape[0] if len(obs2_shape) == 1 else obs2_shape[0]*obs2_shape[1]
        self.env2_action = self.env2.action_space.n if type(self.env2.action_space)== gym.spaces.discrete.Discrete else self.env2.action_space.shape[0]
        self.env2_type_a = self.action_type_discrete if type(self.env2.action_space)== gym.spaces.discrete.Discrete else self.action_type_continuous
        
        
        self.env3_id     = 'Acrobot-v1'                                     # https://www.gymlibrary.dev/environments/classic_control/acrobot/
        self.env3        = gym.make(self.env3_id, render_mode=self.render)  # gym.make('RoboschoolHumanoid-v1')   
        obs3_shape       = self.env3.observation_space.shape
        self.env3_input  = obs3_shape[0] if len(obs3_shape) == 1 else obs3_shape[0]*obs3_shape[1]
        self.env3_action = self.env3.action_space.n if type(self.env3.action_space)== gym.spaces.discrete.Discrete else self.env3.action_space.shape[0]
        self.env3_type_a = self.action_type_discrete if type(self.env3.action_space)== gym.spaces.discrete.Discrete else self.action_type_continuous

        # it contain  actual state for each enviroment
        self.env_array = { self.env1_id: self.env1, 
                           self.env2_id: self.env2, 
                           self.env3_id: self.env3 }



        # Network
        self.input = 256 # find a standar dimension. probably each enviroment has it's input's shape
        
        self.model           = Net( env1_id=self.env1_id, env1_input=8, env1_outputs=4, 
                                    env2_id=self.env1_id, env2_input=24, env2_outputs=2, 
                                    env3_id=self.env1_id, env3_input=6, env3_outputs=3, 
                                    learning_rate=self.learning_rate).to(self.device) 
        

        # for plotting
        self.training_reward = []
        self.training_loss   = []
        self.reward_episode  = []

        print(self.model)
        print(f"{self.env1_id}:\t input:{self.env1_input}, output:{self.env1_action}")
        print(f"{self.env2_id}:\t input:{self.env2_input}, output:{self.env2_action}")
        print(f"{self.env3_id}:\t input:{self.env3_input}, output:{self.env3_action}")
        print(f"Device: {self.device}")


    # TODO: return action index -> used in evaluation
    def forward(self, x, env_id):

        model_input = {'state':x, 'env_id':env_id}

        if self.env1_id == env_id:
            action = self.model.forward(model_input)

        elif self.env1_id == env_id:
            action = self.model.forward(model_input)

        else:
            action = self.model.forward(model_input)

        return action

    # TODO: implement epsilon greedy
    def act(self, state, env_id):

        #print(f"id: {env_id}")
        
        if env_id == self.env1_id:
            if self.env1_type_a == self.action_type_discrete:
                action = random.randint(0, self.env1_action -1 )
            else:
                action = np.random.uniform(-1, 1, self.env1_action)

        elif env_id == self.env2_id:
            if self.env2_type_a == self.action_type_discrete:
                action = random.randint(0, self.env2_action -1 )
            else:
                action = np.random.uniform(-1, 1, self.env2_action)

        elif env_id == self.env3_id:
            if self.env3_type_a == self.action_type_discrete:
                action = random.randint(0, self.env3_action -1 )
            else:
                action = np.random.uniform(-1, 1, self.env3_action)

        '''
        if np.random.rand() > self.epsilon:

            model_input = ( state, env_id)
            act_values = self.forward( model_input )
            action = torch.argmax(act_values[0])

        else:
            # it vary based on enviroment id
            action = random.randint(0,4)
            # action = np.random.uniform(low=-1, high=1, size=3) # for continuous case
        '''
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
            
            #print(f"env_index: {env_index}, id:{actual_id_env}, done: {env_done[actual_id_env]}")
            env_index += 1
            actual_id_env = env_ids[ env_index % len(env_ids) ]    # env_ids has key for dictionary
            if env_done[ actual_id_env ] == False:

                switched = True
                actual_env = self.env_array[ actual_id_env ]   # picking the state of the enviroment
                
                return actual_env, actual_id_env, env_index


    # training loop
    def train(self):
       
       
        max_epochs = 10
        switch_env_frequency = 20

        window = 50 
        old_mean_reward = -999
        start_time = time.perf_counter()

        # for CASC loss i need distribution at iteration 'k', 'k-1', 'k+1'
        policy_distribution_env1 = {'actual':None, 'old-1':None, 'old-2':None}
        policy_distribution_env2 = {'actual':None, 'old-1':None, 'old-2':None}
        policy_distribution_env3 = {'actual':None, 'old-1':None, 'old-2':None}

        self.beta = 0.3

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
    

            env_index  = random.randint(0,len(self.env_array.keys()) - 1)      # used for switch
            done_counter = 0    # use to terminate episode
            switch_counter = 0  # plotting purporse

            env_ids       = list(self.env_array.keys())  # contain id of all enviroment
            actual_id_env = env_ids[env_index]       # contain id of the actual enviroment
            actual_env    = self.env_array[actual_id_env]  # contain the state of the selected enviroment accordin to id

            done = False            

            while True:

                if env_step[actual_id_env] % switch_env_frequency == 0 or done == True:
                    #print(f"old: {actual_id_env}")
                    actual_env, actual_id_env, env_index = self.swtich_enviroment( actual_id_env, env_ids, env_done, env_index  )
                    switch_counter += 1
                    #print(f"new: {actual_id_env}\n")

                state = env_states[ actual_id_env ]
                action = self.act(state, actual_id_env) # TODO: implement epsilon-greedy policy
                
                
                next_state, reward, terminated, truncated, _ = actual_env.step(action)
                env_states[ actual_id_env ] = next_state
                done = terminated or truncated  # truncated if fail to stand up -> penalize!
                
                
                env_reward[actual_id_env] += reward
                env_step[actual_id_env] += 1

                self.calculate_loss(state, next_state, reward, done, actual_id_env)

                # terminate condition
                if env_step[actual_id_env] >= 200: done = True

                # plot statstics
                if done:

                    done_counter += 1
                    env_done[actual_id_env] = True
                
                    total_reward = np.array(list(env_reward.values())).sum()
                    total_steps = np.array(list(env_step.values())).sum()

                    self.training_reward.append(total_reward)
                    self.reward_episode.append(total_reward)

                    mean_rewards = np.mean(self.training_reward[-window:])
                    mean_loss    = np.mean(self.training_loss[-window:])
                    
                    #print("\nEpisode {:d}/{:d}, id: {}, Step: {:d}".format(e, max_epochs, actual_id_env, env_step[actual_id_env]))
                    #print("Mean Rewards {:.2f},  Episode reward = {:.2f},  mean loss = {:.3f}".format(mean_rewards, env_reward[actual_id_env], mean_loss ) )
                    #print("lr: {:.5f}, e: {:.3f} \t\t".format( self.learning_rate, self.epsilon))
                    print(f"Enviroment done: {actual_id_env}")

                    self.epsilon *= self.epsilon_decay

                    if self.epsilon < self.epsilon_min:
                        self.epsilon = self.epsilon_min


                    #plot_training_rewards(self)
                    #plot_training_loss(self)
                    #plot_episode_reward(self)

                    actual_mean_reward = np.mean(list(self.training_reward[-window:])) 

                    if actual_mean_reward > old_mean_reward:
                        #print("Better model! old_mean_reward: {:.2f}, new_mean_reward: {:.2f}".format(old_mean_reward, actual_mean_reward))
                        old_mean_reward = actual_mean_reward
                        #self.save('end_episode.pt')

                    if done_counter >= len(self.env_array):
                        break

           
            print("\nEpisode {:d}/{:d}, \nid: [{}], \nStep: [{}], Switch: {}".format( e,
                                                                    max_epochs,
                                                                    ', '.join( map(str,list(self.env_array.keys()) )), 
                                                                    ', '.join( map(str,list(env_step.values()) )),
                                                                    switch_counter
                                                                    ))
            print("Mean Rewards {:.2f},  Episode reward = [{}],  mean loss = {:.3f}".format(mean_rewards, 
                                                                                              ', '.join( map(lambda x: '{:.2f}'.format(x),list(env_reward.values()) )), 
                                                                                              mean_loss ) 
                                                                                              )
            print("lr: {:.5f}, e: {:.3f} \t\t".format( self.learning_rate, self.epsilon))

            #plot_training_rewards(self)
            #plot_training_loss(self)
            #plot_episode_reward(self)

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            print("Elapsed time: {:.2f} seconds".format(elapsed_time))
            print("Elapsed time: {:.2f} minutes".format(elapsed_time/60))
            print()

            #self.save('model.pt')

    # how calculate loss
    def calculate_loss(self, state, next_state, reward, done, actual_id_env):
        self.model.optimizer.zero_grad()

        loss = loss_ppo(self, actual_id_env, state, next_state, reward, done)


        #for param in self.layer_without_gradient.parameters():
        #    param.requires_grad = False

        loss.backward()
        self.model.optimizer.step()
        self.model.optimizer.zero_grad()

        #for param in self.layer_without_gradient.parameters():
        #    param.requires_grad = True

        return 


    # Utility functions
    def save(self, name = 'model.pt' ):
        torch.save(self.state_dict(), name )

    def load(self, name = 'model.pt'):
        self.load_state_dict(torch.load(name, map_location = self.device))
         
    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret