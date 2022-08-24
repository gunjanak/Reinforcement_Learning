import gym
import torch

from collections import deque
import random
from nn_estimator import DQN
import pickle
import time

#need to import class of PolicyNetwork before importing its pickled object 
#otherwise it will generate error
from PolicyNetwork_AC import PolicyNetwork
from PolicyNetwork_AC import ActorCriticModel


#Episode ends when free end reaches target height
env = gym.envs.make("Acrobot-v1")
#env.seed(416)

#this one is good
env.seed(777)

env.reset()
time.sleep(5)


file_to_read = open("actor_critic_acrobat_2.pkl", "rb")

policy_net = pickle.load(file_to_read)
print(type(policy_net))
state = env.reset()
is_done = False
total_reward_episode = 0

#the agent selecting action randomly
while not is_done:
    env.render()
    time.sleep(0.05)
    #action = random.choice([0, 1, 2])
    action, log_prob, state_value  = policy_net.get_action(state)
    next_state, reward, is_done, _ = env.step(action)
   
    state = next_state
    total_reward_episode += reward


print(total_reward_episode)
