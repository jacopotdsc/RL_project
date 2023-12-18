import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch
#from PIL import Image

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