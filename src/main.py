import argparse
import random
import numpy as np
from agent import *
import gymnasium as gym

def evaluate(env=None, n_episodes=1, render=False):
    
    agent = Agent(env)
    print(f"\nplaying env: {env}\n")
    agent.model.load('training_progress.pt')

    #env = gym.make('CarRacing-v2', continuous=agent.continuous)
    #if render:
        #env = gym.make('CarRacing-v2', continuous=agent.continuous, render_mode='human')
    
    render = 'human'
    env = gym.make(env, render_mode=render)

    rewards = []
    for episode in range(n_episodes):
        total_reward = 0
        done = False
        s, _ = env.reset()
        while not done:
            action = torch.argmax(agent.forward(s)).item()
            
            s, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        rewards.append(total_reward)
        
    print('Mean Reward:', np.mean(rewards))


def train():
    agent = Agent()
    agent.train()
    agent.model.save()


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e1', '--evaluate1', action='store_true')
    parser.add_argument('-e2', '--evaluate2', action='store_true')
    parser.add_argument('-e3', '--evaluate3', action='store_true')

    args = parser.parse_args()

    if args.train:
        train()

    # write the tree enviroment 
    if args.evaluate1:
        env1 = 'MountainCar-v0'  # insert the string id, enviroment will be created into the agent
        evaluate(render=args.render, env=env1)

    if args.evaluate2:
        env2 = 'LunarLander-v2'
        evaluate(render=args.render, env=env2)

    if args.evaluate3:
        env3 = 'CartPole-v0'
        evaluate(render=args.render, env=env3)

    
if __name__ == '__main__':
    main()
