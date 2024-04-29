# import depedencies
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# load environment
environment_name = 'CartPole-v1'
env = gym.make(environment_name, render_mode='rgb_array')

# action space:         env.action_space
# obersevation space:   env.observation_space
episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample() # take random action
        n_state, reward, done, trunc, info = env.step(action) # apply action to environment
        score += reward
    print(f'Episode: {episode}, Score: {score}')
env.close()

# understand the environment
