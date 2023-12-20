import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import random
import time
from copy import deepcopy
from torchviz import make_dot
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

        self.beta            = 0.1  ## added by giordano
        self.omega           = 0.25

        # loss functions
        self.learning_rate   = learning_rate
        self.mse_loss        = nn.MSELoss()

        # list of enviroment needed for training
        self.evaluation_env = env # used during evaluation: it contian the id of the enviroment ( maybe do a gym.make(env) )

        self.render = None #'human' # None
        self.action_type_discrete   = 'Discrete'
        self.action_type_continuous = 'Continuous'

        self.env1_id     = 'MountainCar-v0' #'LunarLander-v2'                                 # https://www.gymlibrary.dev/environments/box2d/lunar_lander/
        self.env1        = gym.make(self.env1_id, render_mode=self.render)  # gym.make('HalfCheetah-v2')
        obs1_shape       = self.env1.observation_space.shape
        self.env1_input  = obs1_shape[0] if len(obs1_shape) == 1 else obs1_shape[0]*obs1_shape[1]
        self.env1_action = self.env1.action_space.n if type(self.env1.action_space)== gym.spaces.discrete.Discrete else self.env1.action_space.shape[0]
        self.env1_type_a = self.action_type_discrete if type(self.env1.action_space)== gym.spaces.discrete.Discrete else self.action_type_continuous
        
        
        self.env2_id     = 'LunarLander-v2' #'Pendulum-v1'#'BipedalWalker-v3'                               # https://www.gymlibrary.dev/environments/box2d/bipedal_walker/
        self.env2        = gym.make(self.env2_id, render_mode=self.render)  # gym.make('HumanoidSmallLeg-v0')  
        obs2_shape       = self.env2.observation_space.shape
        self.env2_input  = obs2_shape[0] if len(obs2_shape) == 1 else obs2_shape[0]*obs2_shape[1]
        self.env2_action = self.env2.action_space.n if type(self.env2.action_space)== gym.spaces.discrete.Discrete else self.env2.action_space.shape[0]
        self.env2_type_a = self.action_type_discrete if type(self.env2.action_space)== gym.spaces.discrete.Discrete else self.action_type_continuous
        
        '''
        # discretizzazione dell'Action Space se continuous:
        if not type(self.env2.action_space)== gym.spaces.discrete.Discrete:
            n_actions = self.env2_action
            low_value = self.env2.action_space.low
            high_value = self.env2.action_space.high
            discrete_actions_env2 = []
            for i in range(0, n_actions):
                action = torch.zeros(n_actions)
                action[i] = torch.tensor(low_value[0])
                discrete_actions_env2.append(action)
                action[i] = torch.tensor(high_value[0])
                discrete_actions_env2.append(action)
            self.discrete_actions_env2 = discrete_actions_env2
        '''
        
        
        self.env3_id     = 'CartPole-v0' #'Acrobot-v1'                                     # https://www.gymlibrary.dev/environments/classic_control/acrobot/
        self.env3        = gym.make(self.env3_id, render_mode=self.render)  # gym.make('RoboschoolHumanoid-v1')   
        obs3_shape       = self.env3.observation_space.shape
        self.env3_input  = obs3_shape[0] if len(obs3_shape) == 1 else obs3_shape[0]*obs3_shape[1]
        self.env3_action = self.env3.action_space.n if type(self.env3.action_space)== gym.spaces.discrete.Discrete else self.env3.action_space.shape[0]
        self.env3_type_a = self.action_type_discrete if type(self.env3.action_space)== gym.spaces.discrete.Discrete else self.action_type_continuous

        # it contain  actual state for each enviroment
        self.env_array = { self.env1_id: self.env1, 
                           self.env2_id: self.env2, 
                           self.env3_id: self.env3 }


        self.buffer = None
        # Network
        self.input = 256 # find a standar dimension. probably each enviroment has it's input's shape
        
        self.model           = Net( env1_id=self.env1_id, env1_input=2, env1_outputs=3, 
                                    env2_id=self.env2_id, env2_input=8, env2_outputs=4,#len(self.discrete_actions_env2), 
                                    env3_id=self.env3_id, env3_input=4, env3_outputs=2, 
                                    learning_rate=self.learning_rate, device=self.device).to(self.device) 
        

        self.old_policy = None
        

        # for plotting
        self.training_reward = {    self.env1_id: [], 
                                    self.env2_id: [], 
                                    self.env3_id: [] }
        self.training_loss   = {    self.env1_id: [], 
                                    self.env2_id: [], 
                                    self.env3_id: [] }
        self.reward_episode  = []

        print(self.model)
        print(f"id -> env1: {self.model.env1_id}, env2: {self.model.env2_id}, env3: {self.model.env3_id}")
        print(f"{self.env1_id}:\t input:{self.env1_input}, output:{self.env1_action}")
        print(f"{self.env2_id}:\t input:{self.env2_input}, output:{self.env2_action}")
        print(f"{self.env3_id}:\t input:{self.env3_input}, output:{self.env3_action}")
        print(f"Device: {self.device}")


    # TODO: return action index -> used in evaluation
    def forward(self, x):

        model_input = self.model.create_model_input(x, self.evaluation_env)
        action = self.model.forward(model_input)

        return action

    # TODO: implement epsilon greedy
    def act(self, state, env_id, env_type = None):  
        
        '''
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
    
        if np.random.uniform(0,1) < self.epsilon:
            if env_id == self.env1_id:
                action = random.randint(0, self.env1_action -1 )

            elif env_id == self.env2_id:
                #action = np.random.uniform(-1, 1, self.env2_action)
                action = random.randint(0, self.env2_action -1 )
                
            elif env_id == self.env3_id:
                action = random.randint(0, self.env3_action -1 )

        else:
            model_input = self.model.create_model_input(state, env_id)
            #action = self.model.forward(model_input)
            action = self.model.q_val(model_input)

            if env_id == self.env2_id:
                #action = action.detach().cpu()
                action = torch.argmax(action).item()
            else:
                action = torch.argmax(action).item()

        # when changing to dinamic choice, use env_type to do it

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

    def train(self):
       
        self.buffer = Buffer(self, memory_size=500, batch_size = 64)
        max_epochs = 300
        switch_env_frequency = 20

        
        step_train = 0
        calculate_loss_frequency = 5

        window = 50 
        old_mean_reward = -999
        start_time = time.perf_counter()

        # for CASC loss i need distribution at iteration 'k', 'k-1', 'k+1'
        policy_distribution_env1 = {'actual':None, 'old-1':None, 'old-2':None}
        policy_distribution_env2 = {'actual':None, 'old-1':None, 'old-2':None}
        policy_distribution_env3 = {'actual':None, 'old-1':None, 'old-2':None}

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

                self.buffer.add(init_state, env_id)
    

            env_index  = random.randint(0,len(self.env_array.keys()) - 1)      # used for switch
            done_counter = 0    # use to terminate episode
            switch_counter = 0  # plotting purporse

            env_ids       = list(self.env_array.keys())  # contain id of all enviroment
            actual_id_env = env_ids[env_index]       # contain id of the actual enviroment
            actual_env    = self.env_array[actual_id_env]  # contain the state of the selected enviroment accordin to id

            done = False            

            while True:
                step_train += 1

                if env_step[actual_id_env] % switch_env_frequency == 0 or done == True:
                    #print(f"old: {actual_id_env}")
                    actual_env, actual_id_env, env_index = self.swtich_enviroment( actual_id_env, env_ids, env_done, env_index  )
                    switch_counter += 1
                    #print(f"new: {actual_id_env}\n")

                state = env_states[ actual_id_env ]
                action = self.act(state, actual_id_env) 
                
                
                next_state, reward, terminated, truncated, _ = actual_env.step(action)
                env_states[ actual_id_env ] = next_state
                done = terminated or truncated  # truncated if fail to stand up -> penalize!
                
                if torch.is_tensor(reward): reward = reward.item()
                reward = float(reward)
                
                env_reward[actual_id_env] += reward
                env_step[actual_id_env] += 1

                
                self.buffer.add(next_state, actual_id_env)

                if self.buffer.buffer_size(actual_id_env) >= self.buffer.batch_size and (step_train % calculate_loss_frequency) == 0:
                    #print(f"enter {actual_id_env}, buffer size: {self.buffer.buffer_size(actual_id_env)}")
                    self.calculate_loss(state, action, next_state, reward, done, actual_id_env, self.beta, self.omega)

                # terminate condition
                if env_step[actual_id_env] >= 500: done = True

                # plot statstics
                if done:

                    done_counter += 1
                    env_done[actual_id_env] = True
                
                    total_reward = np.array(list(env_reward.values())).sum()
                    total_steps = np.array(list(env_step.values())).sum()

                    self.training_reward[actual_id_env].append(total_reward)
                    self.reward_episode.append(total_reward)

                    #mean_rewards = np.mean(self.training_reward[-window:])
                    #mean_loss    = np.mean(self.training_loss[-window:])
                    
                    #print("\nEpisode {:d}/{:d}, id: {}, Step: {:d}".format(e, max_epochs, actual_id_env, env_step[actual_id_env]))
                    #print("Mean Rewards {:.2f},  Episode reward = {:.2f},  mean loss = {:.3f}".format(mean_rewards, env_reward[actual_id_env], mean_loss ) )
                    #print("lr: {:.5f}, e: {:.3f} \t\t".format( self.learning_rate, self.epsilon))
                    #print(f"Enviroment done: {actual_id_env}")

                    


                    #plot_training_rewards(self)
                    #plot_training_loss(self)
                    #plot_episode_reward(self)      

                    '''
                    actual_mean_reward = np.mean(list(self.training_reward[-window:])) 

                    if actual_mean_reward > old_mean_reward:
                        #print("Better model! old_mean_reward: {:.2f}, new_mean_reward: {:.2f}".format(old_mean_reward, actual_mean_reward))
                        old_mean_reward = actual_mean_reward
                        #self.save('end_episode.pt')
                    '''

                    if done_counter >= len(self.env_array):
                        self.epsilon *= self.epsilon_decay

                        if self.epsilon < self.epsilon_min:
                            self.epsilon = self.epsilon_min
                        break
                    
           
            print("\nEpisode {:d}/{:d}, \nid: [{}], \nStep: [{}], Switch: {}".format( e,
                                                                    max_epochs,
                                                                    ', '.join( map(str,list(self.env_array.keys()) )), 
                                                                    ', '.join( map(str,list(env_step.values()) )),
                                                                    switch_counter
                                                                    ))
            
            mean_loss = []
            for key, value in self.training_loss.items():
                if len(value) > 0:
                    mean_loss.append( torch.mean(torch.stack(value[-window:]) ).item() )

            '''
            mean_reward = []
            for key, value in self.training_reward.items():
                if len(value) > 0:
                    mean_reward.append( torch.mean(torch.stack(value[-window:]) ).item() )
            '''
            
            print("Mean Rewards [{}],  Episode reward = [{}],  mean loss = [{}]".format(    0, 
                                                                                              ', '.join( map(lambda x: '{:.2f}'.format(x),list(env_reward.values()) )), 
                                                                                              ', '.join( map(lambda x: '{:.2f}'.format(x),list(mean_loss) ))
                                                                                              )
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

            self.model.save('model.pt')

    # how calculate loss
    def calculate_loss(self, state, action, next_state, reward, done, actual_id_env, beta, omega):
        self.model.optimizer.zero_grad()
        
        if actual_id_env == self.env1_id:
            self.model.set_gradient_layer( self.model.input_env2, False )
            self.model.set_gradient_layer( self.model.output_env2, False )
            self.model.set_gradient_layer( self.model.input_env3, False )
            self.model.set_gradient_layer( self.model.output_env3, False )

        elif actual_id_env == self.env2_id:
            self.model.set_gradient_layer( self.model.input_env1, False )
            self.model.set_gradient_layer( self.model.output_env1, False )
            self.model.set_gradient_layer( self.model.input_env3, False )
            self.model.set_gradient_layer( self.model.output_env3, False )

        elif actual_id_env == self.env3_id:
            self.model.set_gradient_layer( self.model.input_env1, False )
            self.model.set_gradient_layer( self.model.output_env1, False )
            self.model.set_gradient_layer( self.model.input_env2, False )
            self.model.set_gradient_layer( self.model.output_env2, False )

        #################### computation of actual policy
        
        model_input = self.model.create_model_input(state, actual_id_env)
        q_vals = self.model.q_val(model_input)

        next_model_input = self.model.create_model_input(next_state, actual_id_env)
        next_qvals = self.model.q_val(next_model_input)

        if self.old_policy is None:
            # calculating td-error
            if actual_id_env == self.env2_id:
                qvals = q_vals#.detach().cpu()
                next_qvals = next_qvals#.detach().cpu()
                target_qvals = reward*torch.ones( len(next_qvals) ) + (1 - done)*self.gamma*next_qvals
                loss_value= self.mse_loss(qvals, target_qvals)
            else:
                qval = q_vals[action].to(self.device)
                target_qval = reward + (1 - done)*self.gamma*torch.max(next_qvals, dim=-1)[0].reshape(-1, 1).item()
                loss_value = self.mse_loss(qval, torch.tensor(target_qval).to(self.device))

        else:
            old_pi = self.old_policy.forward(model_input)  # la media delle DKLs è relativa al tempo t ---> s(t)
            new_pi = self.model.forward(model_input)
            
            prediction_old = []
            prediction_actual = []
            batch = self.buffer.sample(actual_id_env)

            #print(f"batch size: {len(batch)}, sample: {batch[0].shape}")

            for sample in batch:

                new_model_input = self.model.create_model_input(sample, actual_id_env)
                #print(f"model input: {new_model_input}")
                prediction_old.append( self.old_policy.forward(new_model_input) )
                prediction_actual.append( self.model.forward(new_model_input) )

            old_pi_stacked_tensor = torch.stack(prediction_old, dim=0)
            actual_pi_stacked_tensor = torch.stack(prediction_actual, dim=0)

            #loss_value = loss_ppo(self, new_pi, old_pi, beta, omega)
            loss_value = loss_ppo(actual_id_env, actual_pi_stacked_tensor, old_pi_stacked_tensor)

            #graph = make_dot(new_pi, params=dict(self.model.named_parameters()))
            #graph.render("computational_graph", format="png")  
            
            #print(f"old policy: {old_pi}")
            #print(f"new policy: {new_pi}\n")
            #print(f"id: {actual_id_env}, loss value: {loss_value}, type: {type(loss_value)}\n")


        # keeping the record of my policy
        self.old_policy = deepcopy(self.model)
        for param in self.old_policy.parameters():
            param.requires_grad = False

        self.old_policy.optimizer = None
        
        #loss_value = torch.tensor(loss_value.item(), requires_grad = True)

        self.training_loss[actual_id_env].append(loss_value)

        loss_value.backward()
        self.model.optimizer.step()
        self.model.optimizer.zero_grad()
       


        if actual_id_env == self.env1_id:
            self.model.set_gradient_layer( self.model.input_env2, True )
            self.model.set_gradient_layer( self.model.output_env2, True )
            self.model.set_gradient_layer( self.model.input_env3, True )
            self.model.set_gradient_layer( self.model.output_env3, True )

        elif actual_id_env == self.env2_id:
            self.model.set_gradient_layer( self.model.input_env1, True )
            self.model.set_gradient_layer( self.model.output_env1, True )
            self.model.set_gradient_layer( self.model.input_env3, True )
            self.model.set_gradient_layer( self.model.output_env3, True )

        elif actual_id_env == self.env3_id:
            self.model.set_gradient_layer( self.model.input_env1, True )
            self.model.set_gradient_layer( self.model.output_env1, True )
            self.model.set_gradient_layer( self.model.input_env2, True )
            self.model.set_gradient_layer( self.model.output_env2, True )


        return 


    # Utility functions
    def save(self, name = 'model.pt' ):
        torch.save(self.state_dict(), name )

    def load(self, name = 'model.pt'):
        self.load_state_dict(torch.load(name,  map_location=self.device) )
         
    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret