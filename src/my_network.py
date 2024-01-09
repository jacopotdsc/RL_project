from importlib.metadata import requires
import torch
import torch.nn as nn
import numpy as np
from utils import *
from scipy.stats import norm


class Net(nn.Module):
    def __init__(self,  env1_id, env1_input, env1_outputs, 
                        env2_id, env2_input, env2_outputs, 
                        env3_id, env3_input, env3_outputs, 
                        learning_rate, encoder = None, bias=True):


        super(Net, self).__init__()

        self.encoder = encoder

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
        self.log_soft = nn.LogSoftmax(dim=-1)

        self.hidden_size = 64

        # input layers
        self.common_input_layer = nn.Linear(in_features=self.encoder.size, out_features=self.hidden_size, bias=bias)
        self.input_env1 = nn.Linear(in_features=self.env1_input, out_features=self.hidden_size, bias=bias)
        self.input_env2 = nn.Linear(in_features=self.env2_input, out_features=self.hidden_size, bias=bias)
        self.input_env3 = nn.Linear(in_features=self.env3_input, out_features=self.hidden_size, bias=bias)

        # hidden layers   
        self.hl1    = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=bias) 
        self.hl_fin = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=bias) 


        # output layers: one for each enviroment
        self.output_env1 = nn.Linear(in_features=self.hidden_size, out_features=self.env1_outputs, bias=bias)
        self.output_env2 = nn.Linear(in_features=self.hidden_size, out_features=self.env2_outputs, bias=bias)
        self.output_env3 = nn.Linear(in_features=self.hidden_size, out_features=self.env3_outputs, bias=bias)


        # optimizer -> check how it work values
        self.optimizer = torch.optim.Adam(self.parameters(),lr=learning_rate)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_model_input(self, state, env_id):
         return {'state':state, 'env_id': env_id}
    
    # return the action's probabilities
    def forward(self, x):

        q_val = self.q_val(x)

        # implement dinamic choice with env_type in agent
        #if x['env_id'] == self.env2_id:probs = self.softm(q_val)
        #else: probs = self.softm(q_val)
        probs = self.softm(q_val)
        
        return probs

    def log_pi(self, q_val):
        probs = self.log_soft(q_val)
        return probs
    
    def pi(self, q_val):
        probs = self.softm(q_val)
        return probs

    def regression_pi(self, q_val):
        sigma = 0.5
        range = np.linspace(-2, 2, 100)
        q_val = q_val.detach().cpu().numpy()
        probs = norm.pdf(range, q_val, sigma)

        return torch.tensor(probs, device=self.device, requires_grad = True)

    # return the q-value -> input = ( x, env_id )
    def q_val(self, my_input):

        x = my_input['state']
        env_id = my_input['env_id']

        x = preprocess(self, my_input).to(self.device)

        x = self.common_input_layer(x)
        '''
        if env_id == self.env1_id:
            x = self.input_env1(x)

        elif env_id == self.env2_id:
            x = self.input_env2(x)

        elif env_id == self.env3_id :
            x = self.input_env3(x)
        '''

        x = self.hl1(x)
        x = self.relu(x)
        x = self.hl_fin(x)
        x = self.relu(x)
        
        if env_id == self.env1_id:
            x = self.output_env1(x)

        elif env_id == self.env2_id:
            x = self.output_env2(x)

        elif env_id == self.env3_id :
            x = self.output_env3(x)
        
        return x

    def set_gradient_layer(self, layer, gradient_value):

        # layer.paramters is a tensor. It set the gradient to all parameter in one iteration of the loop
        for param in layer.parameters():
            param.requires_grad = gradient_value

    # Utility functions
    def save(self, name = 'model.pt' ):
        print("saving weight model")
        torch.save(self.state_dict(), "../model_folder/" + name )

    def load(self, name = 'model.pt'):
        self.load_state_dict(torch.load("../model_folder/" + name,  map_location=self.device) )
        print(f"loaded: {name}")
         
    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret


        