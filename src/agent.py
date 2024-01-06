import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import random
import time
from copy import deepcopy
#from torchviz import make_dot
#import mujoco_py

from my_network import *
from calculate_loss_function import *


###
# from baselines.common.cmd_util import mujoco_arg_parser, make_mujoco_env
# "RoboSumo-Ant-vs-Ant-v0"
#from gym_extensions.continuous import mujoco
class Agent(nn.Module):

    def __init__(   self, env = None, gamma = 0.95, 
                    epsilon = 1, epsilon_min = 0.2, epsilon_decay  = 0.99,
                    learning_rate   = 0.001 ):
        
        super(Agent, self).__init__()

        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # hyperparameters
        self.gamma           = gamma
        self.epsilon         = epsilon
        self.epsilon_min     = epsilon_min
        self.epsilon_decay   = epsilon_decay

        # loss functions
        self.learning_rate   = learning_rate
        self.mse_loss        = nn.MSELoss()

        # list of enviroment needed for training
        self.evaluation_env = env # used during evaluation: it contian the id of the enviroment ( maybe do a gym.make(env) )

        self.render = None #'human' # None
        self.action_type_discrete   = 'Discrete'
        self.action_type_continuous = 'Continuous'

        self.env1_id     = 'Acrobot-v1' #'Walker2d-v4'  #'LunarLander-v2'                    # https://www.gymlibrary.dev/environments/box2d/lunar_lander/
        self.env1        = gym.make(self.env1_id, render_mode=self.render)  # gym.make('HalfCheetah-v2')
        obs1_shape       = self.env1.observation_space.shape
        self.env1_input  = obs1_shape[0] if len(obs1_shape) == 1 else obs1_shape[0]*obs1_shape[1]
        self.env1_action = self.env1.action_space.n if type(self.env1.action_space)== gym.spaces.discrete.Discrete else self.env1.action_space.shape[0]
        self.env1_type_a = self.action_type_discrete if type(self.env1.action_space)== gym.spaces.discrete.Discrete else self.action_type_continuous
        
        
        self.env2_id     = 'CartPole-v1' #'Walker2dBigLeg-v0'        # https://www.gymlibrary.dev/environments/box2d/bipedal_walker/
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
        
        
        self.env3_id     = 'LunarLander-v2' #'Acrobot-v1'                                     # https://www.gymlibrary.dev/environments/classic_control/acrobot/
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
        self.encoder = RBFFeatureEncoder(self.env1, self.env2, self.env3)
        print(f"self.env2_action: {self.env2_action}")
        self.model           = Net( env1_id=self.env1_id, env1_outputs=self.env1_action, 
                                    env2_id=self.env2_id, env2_outputs=self.env2_action,
                                    env3_id=self.env3_id, env3_outputs=self.env3_action, 
                                    learning_rate=self.learning_rate, encoder = self.encoder).to(self.device) 
        self.model2          = Net( env1_id=self.env1_id, env1_outputs=self.env1_action, 
                                    env2_id=self.env2_id, env2_outputs=self.env2_action,
                                    env3_id=self.env3_id, env3_outputs=self.env3_action, 
                                    learning_rate=self.learning_rate, encoder = self.encoder).to(self.device)
        self.model3          = Net( env1_id=self.env1_id, env1_outputs=self.env1_action, 
                                    env2_id=self.env2_id, env2_outputs=self.env2_action,
                                    env3_id=self.env3_id, env3_outputs=self.env3_action, 
                                    learning_rate=self.learning_rate, encoder = self.encoder).to(self.device)
        

        self.old_policy = deepcopy(self.model)
        self.old_policy2 = deepcopy(self.model2)
        self.old_policy3 = deepcopy(self.model3)
        

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

    def forward(self, x):

        model_input = self.model.create_model_input(x, self.evaluation_env)
        action = self.model.forward(model_input)

        return action

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
            model_input = self.encoder.encode(self, model_input)
            action = self.model.q_val(model_input)
            action = torch.argmax(action).item()

            #if env_id == self.env2_id:
            #    #action = action.detach().cpu()
            #    action = torch.argmax(action).item()
            #else:
            #    action = torch.argmax(action).item()

        # when changing to dinamic choice, use env_type to do it

        return action

    def swtich_enviroment(self, actual_id_env, env_ids, env_index):

        # support variable: env_index, env_ids, env_done
        # changed: actual_id_env, actual_env

        # actual_id_env: actual id of the enviroment
        # env_ids: array with all keys ( names ) of enviroment
        # env_states: actual state of the actual enviroment
        # env_done: dictionary which contain done-boolean
        # env_index: general counter
        '''
        switched = False

        while not switched:
            
            #print(f"env_index: {env_index}, id:{actual_id_env}, done: {env_done[actual_id_env]}")
            env_index += 1
            actual_id_env = env_ids[ env_index % len(env_ids) ]    # env_ids has key for dictionary
            if env_done[ actual_id_env ] == False:

                switched = True
                actual_env = self.env_array[ actual_id_env ]   # picking the state of the enviroment
                
                return actual_env, actual_id_env, env_index
        '''
        env_index += 1
        actual_id_env = env_ids[ env_index % 2 ]   # -------- the switch is only for the first two environments    
        actual_env = self.env_array[ actual_id_env ]
        return actual_env, actual_id_env, env_index
    

    def custom_reset_enviroment(self, id_enviroment, step_ahead = None):
        # it will reset in a certain state.
        # it will continue to reset enviroment until it doesn't reach a state at least
        # steap_ahed steps ahead to the initial state

        resetted = False
        state_returned = None

        if self.env1_id == id_enviroment:
            step_ahead = 40
        elif self.env2_id == id_enviroment:
            step_ahead = 10
        elif self.env3_id == id_enviroment:
            step_ahead = 3

        while not resetted:
            self.env_array[id_enviroment].reset()
            action = random.randint(0, self.env_array[id_enviroment].action_space.n -1)

            step_counter = 0
            while step_counter < step_ahead:
                step_counter += 1
                state_returned, reward, terminated, truncated, _ = self.env_array[id_enviroment].step(action)
                done = terminated or truncated 

                #print(f"id_enviroment: {id_enviroment}, done: {done}, step_counter: {step_counter}")
                if done:
                    break
                if step_counter >= step_ahead:
                    return state_returned, None

    def train(self):
       
        max_epochs = 400 
        self.buffer = Buffer(self, memory_size=5000)

        window = 50 
        start_time = time.perf_counter()

        env_index  = 0 
        # create dictionary for switching easier between states of enviroments
        env_states = {}     # contain state to pass when callicng act
        env_step   = {}     # counter for switch
        env_reward = {}     # counter for reward
        env_ids       = list(self.env_array.keys())  # contain id of all enviroment
        '''--------------------------- DA RIVEDERE LA FREQUENZA ------------------------------ '''
        switch_env_frequency = 10
        for e in range(max_epochs):
            # RESET of OLD Environment
            actual_id_env = env_ids[env_index % 2]       # contain id of the actual enviroment
            actual_env    = self.env_array[actual_id_env]  # contain the state of the selected enviroment according to id
            init_state, _ = self.env_array[actual_id_env].reset() # Reset of environment that has just finished
            env_states[actual_id_env] = init_state
            env_step[actual_id_env]   = 0
            env_reward[actual_id_env] = 0

            switch_counter = 0  # plotting purporse

            done = False
            # PRESET of NEW Environment
            if e % switch_env_frequency == 0 and e > 0: # ------- check for the switch
                print(f"\nold enviroment: {actual_id_env}")
                actual_env, actual_id_env, env_index = self.swtich_enviroment( actual_id_env, env_ids, env_index )
                switch_counter += 1
                print(f"new enviroment: {actual_id_env}")
                actual_env    = self.env_array[actual_id_env]  # contain the state of the selected enviroment according to id
                init_state, _ = self.env_array[actual_id_env].reset() # Reset of environment that has just finished
                env_states[actual_id_env] = init_state
                env_step[actual_id_env]   = 0
                env_reward[actual_id_env] = 0

            while not done:

                state = env_states[ actual_id_env ]
                action = self.act(state, actual_id_env) 
                next_state, reward, terminated, truncated, _ = actual_env.step(action)
                env_states[ actual_id_env ] = next_state # STATE <--- NEXT STATE
                done = terminated or truncated  # truncated if fail to stand up -> penalize!
                
                if torch.is_tensor(reward): reward = reward.item()
                reward = float(reward)

                '''
                if reward < 0:
                    counter_negative_reward[actual_id_env] += 1
                if actual_id_env == self.env1_id and counter_negative_reward[self.env1_id] > 100:
                    reward = -5.0
                elif actual_id_env == self.env2_id and counter_negative_reward[self.env2_id] > 40:
                    reward = -5.0
                elif actual_id_env == self.env3_id and env_step[self.env3_id] < 15:
                    reward = 0
                '''

                env_reward[actual_id_env] += reward
                env_step[actual_id_env] += 1

                #self.buffer.add( actual_id_env, state, action, reward, next_state, done  )
                #if self.buffer.buffer_size(actual_id_env) >= self.buffer.batch_size and ( env_step[actual_id_env]  % update_frequency ) == 0:  # implemented update frequency
                #    self.calculate_loss(state, action, next_state, reward, done, actual_id_env)

                self.calculate_loss(state, action, next_state, reward, done, actual_id_env)

                # plot statstics
                if done:
                    print(" Episode done ")
                    #env_done[actual_id_env] = True
                
                    total_reward = np.array(list(env_reward.values())).sum()
                    total_steps = np.array(list(env_step.values())).sum()

                    self.training_reward[actual_id_env].append(env_reward[actual_id_env])
                    self.reward_episode.append(env_reward[actual_id_env])

                    self.epsilon *= self.epsilon_decay
                    if self.epsilon < self.epsilon_min:
                        self.epsilon = self.epsilon_min

                    if done:
                        break
                    '''
                    if done_counter >= len(self.env_array):
                        self.epsilon *= self.epsilon_decay

                        if self.epsilon < self.epsilon_min:
                            self.epsilon = self.epsilon_min
                        break
                    '''
                    
           
            print("\nEpisode {:d}/{:d}, id: {}\nids: [{}], \nStep: [{}], Switch: {}".format( e,
                                                                    max_epochs, actual_id_env,
                                                                    ', '.join( map(str,list(self.env_array.keys()) )), 
                                                                    ', '.join( map(str,list(env_step.values()) )),
                                                                    switch_counter
                                                                    ))
            
            mean_loss = []
            for key, value in self.training_loss.items():
                if len(value) > 0:
                    mean_loss.append( torch.mean(torch.stack(value[-window:])).item() )

            
            mean_reward = []
            for key, value in self.training_reward.items():
                if len(value) > 0:
                    mean_reward.append( np.mean(np.stack(value[-window:])).item() )
            
            
            print("Mean Rewards [{}],  Episode reward = [{}],  mean loss = [{}]".format( ', '.join( map(lambda x: '{:.2f}'.format(x),list(mean_reward) )), 
                                                                        ', '.join( map(lambda x: '{:.2f}'.format(x),list(env_reward.values()) )), 
                                                                        ', '.join( map(lambda x: '{:.2f}'.format(x),list(mean_loss) ))
                                                                                              )
                                                                                              )
            print("lr: {:.5f}, e: {:.3f} \t\t".format( self.learning_rate, self.epsilon))

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            print("Elapsed time: {:.2f} seconds".format(elapsed_time))
            print("Elapsed time: {:.2f} minutes".format(elapsed_time/60))
            print()

            #self.model.save('training_progress_total_loss.pt')

        self.model.save('model.pt')

    '''
    def calculate_loss_input(self, my_model, state, next_state, reward, done, action, actual_id_env, pg_flag = False):
        
        state = self.encoder.encode(self, state, actual_id_env) # Encoding state with encoder 1
        model_input = my_model.create_model_input(state, actual_id_env)
        q_vals = my_model.q_val(model_input)  # Q values estimated by the network, "PREDICTION"
        
        next_state = self.encoder.encode(self, next_state, type=1) # Encoding next_state with encoder 1
        next_model_input = my_model.create_model_input(next_state, actual_id_env)
        next_qvals = self.model.q_val(next_model_input)  # Q values estimated by the network of next state
        
        # Q values computed by the network with experience, "OBSERVATED EXPERIENCE"
        target_qval = reward + (1 - done)*self.gamma*torch.max(next_qvals, dim=-1)[0].reshape(-1, 1).item()
        target_qvals = q_vals
        target_qvals[action] = target_qval

        old_pi = my_model.log_pi(q_vals)# the output predicted by the model must be computed with logSoftmax
        new_pi = my_model.pi(target_qvals)

        pi_1 = self.model.pi(q_vals)# this is the policy which we use to compute the PG Loss, must be computed with Softmax
        ADV_err = reward + (1 - done)*self.gamma*torch.max(next_qvals, dim=-1)[0].reshape(-1, 1).item() - q_vals[action]

        if pg_flag == True:
            return old_pi, new_pi, pi_1, ADV_err
        else:
            return old_pi, new_pi, None, None
    '''
    # how calculate loss
    def calculate_loss(self, state, action, next_state, reward, done, actual_id_env):
        self.model.optimizer.zero_grad()
        
        if actual_id_env == self.env1_id:
            self.model.set_gradient_layer( self.model.output_env2, False )
            self.model.set_gradient_layer( self.model.output_env3, False )

        elif actual_id_env == self.env2_id:
            self.model.set_gradient_layer( self.model.output_env1, False )
            self.model.set_gradient_layer( self.model.output_env3, False )

        elif actual_id_env == self.env3_id:
            self.model.set_gradient_layer( self.model.output_env1, False )
            self.model.set_gradient_layer( self.model.output_env2, False )
        '''
        batch = self.buffer.sample(actual_id_env)
        for sample in batch:
            state, action, reward, next_state, done = sample
            old_pi, new_pi, pi_1, ADV_err  = self.calculate_loss_input(self.model,  state, next_state, reward, done, action, actual_id_env, pg_flag = True)
            old_pi_2, new_pi_2, _, _       = self.calculate_loss_input(self.model2, state, next_state, reward, done, action, actual_id_env, pg_flag = False)
            old_pi_3, new_pi_3, _, _       = self.calculate_loss_input(self.model3, state, next_state, reward, done, action, actual_id_env, pg_flag = False)
            vec_old_pi.append(old_pi);    vec_old_pi.append(old_pi_2);  vec_old_pi.append(old_pi_3)
            vec_new_pi.append(new_pi);    vec_new_pi.append(new_pi_2);  vec_new_pi.append(new_pi_3)
            loss_value = total_loss(self, vec_new_pi, vec_old_pi, ADV_err, action, pi_1) if loss_value is None else loss_value + total_loss(self, vec_new_pi, vec_old_pi, ADV_err, action, pi_1)
        loss_value = loss_value / self.buffer.batch_size
        '''
        ######################################## computation of actual policy
        vec_old_pi = []
        vec_new_pi = []

        # Q values estimated by the network, "PREDICTION"
        model_input = self.model.create_model_input(state, actual_id_env)
        model_input = self.encoder.encode(self, model_input)                     # Encoding state
        q_vals = self.model.q_val(model_input)
        # Q values estimated by the network of next state
        next_model_input = self.model.create_model_input(next_state, actual_id_env)
        next_model_input = self.encoder.encode(self, next_model_input)           # Encoding next_state
        next_qvals = self.model.q_val(next_model_input)
        # Q values computed by the network with experience, "OBSERVATED EXPERIENCE"
        target_qval = reward + (1 - done)*self.gamma*torch.max(next_qvals, dim=-1)[0].reshape(-1, 1).item()
        target_qvals = q_vals.clone().detach()
        target_qvals[action] = target_qval
        old_pi = self.model.log_pi(q_vals)# the output predicted by the model must be computed with logSoftmax
        pi_1 = self.model.pi(q_vals)# this is the policy which we use to compute the PG Loss, must be computed with Softmax
        new_pi = self.model.pi(target_qvals)
        ADV_err = reward + (1 - done)*self.gamma*torch.max(next_qvals, dim=-1)[0].reshape(-1, 1).item() - q_vals[action]
        vec_old_pi.append(old_pi)
        vec_new_pi.append(new_pi)

        ########## repeat for the network 2
        
        # Q values estimated by the network 2, "PREDICTION"
        model_input2 = self.model2.create_model_input(state, actual_id_env)
        model_input2 = self.encoder.encode(self, model_input2)                     # Encoding state
        q_vals2 = self.model2.q_val(model_input2)
        # Q values estimated by the network 2 of next state
        next_model_input2 = self.model2.create_model_input(next_state, actual_id_env)
        next_model_input2 = self.encoder.encode(self, next_model_input2)           # Encoding next_state
        next_qvals2 = self.model2.q_val(next_model_input2)
        # Q values computed by the network 2 with experience, "OBSERVATED EXPERIENCE"
        target_qval2 = reward + (1 - done)*self.gamma*torch.max(next_qvals2, dim=-1)[0].reshape(-1, 1).item()
        target_qvals2 = q_vals2.clone().detach()
        target_qvals2[action] = target_qval2
        old_pi2 = self.model2.log_pi(q_vals2)# the output predicted by the model 2 must be computed with logSoftmax, not Softmax
        new_pi2 = self.model2.pi(target_qvals2)
        vec_old_pi.append(old_pi2)
        vec_new_pi.append(new_pi2)

        ########## repeat for the network 3
        
        # Q values estimated by the network 3, "PREDICTION"
        model_input3 = self.model3.create_model_input(state, actual_id_env)
        model_input3 = self.encoder.encode(self, model_input3)                     # Encoding state
        q_vals3 = self.model3.q_val(model_input3)
        # Q values estimated by the network 3 of next state
        next_model_input3 = self.model3.create_model_input(next_state, actual_id_env)
        next_model_input3 = self.encoder.encode(self, next_model_input3)           # Encoding next_state
        next_qvals3 = self.model3.q_val(next_model_input3)
        # Q values computed by the network 3 with experience, "OBSERVATED EXPERIENCE"
        target_qval3 = reward + (1 - done)*self.gamma*torch.max(next_qvals3, dim=-1)[0].reshape(-1, 1).item()
        target_qvals3 = q_vals3.clone().detach()
        target_qvals3[action] = target_qval3
        old_pi3 = self.model3.log_pi(q_vals3)# the output predicted by the model 3 must be computed with logSoftmax, not Softmax
        new_pi3 = self.model3.pi(target_qvals3)
        vec_old_pi.append(old_pi3)
        vec_new_pi.append(new_pi3)

        loss_value = total_loss(self, vec_new_pi, vec_old_pi, ADV_err, action, pi_1)

        # keeping the record of my policies
        self.old_policy  = deepcopy(self.model)
        self.old_policy2 = deepcopy(self.model2)
        self.old_policy3 = deepcopy(self.model3)

        for param in self.old_policy.parameters():
            param.requires_grad = False
        for param in self.old_policy2.parameters():
            param.requires_grad = False
        for param in self.old_policy3.parameters():
            param.requires_grad = False

        self.old_policy.optimizer = None
        self.old_policy2.optimizer = None
        self.old_policy3.optimizer = None
        

        self.training_loss[actual_id_env].append(loss_value)
        
        loss_value = torch.tensor(loss_value.item(), requires_grad = True)
        #print(f"loss_value: {loss_value}\n")

        loss_value.backward()

        self.model.optimizer.step()
        self.model2.optimizer.step()
        self.model3.optimizer.step()

        self.model.optimizer.zero_grad()
        self.model2.optimizer.zero_grad()
        self.model3.optimizer.zero_grad()
       


        if actual_id_env == self.env1_id:
            self.model.set_gradient_layer( self.model.output_env2, True )
            self.model.set_gradient_layer( self.model.output_env3, True )

        elif actual_id_env == self.env2_id:
            self.model.set_gradient_layer( self.model.output_env1, True )
            self.model.set_gradient_layer( self.model.output_env3, True )

        elif actual_id_env == self.env3_id:
            self.model.set_gradient_layer( self.model.output_env1, True )
            self.model.set_gradient_layer( self.model.output_env2, True )


        return 


    # Utility functions
    def save(self, name = 'model.pt' ):
        #torch.save(self.state_dict(), "../model_folder/"+name )
        self.model.save(name)

    def load(self, name = 'model.pt'):
        #self.load_state_dict(torch.load("../model_folder/"+name,  map_location=self.device) )
        self.model.load(name)

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret