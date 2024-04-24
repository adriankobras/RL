import gymnasium as gym
from stable_baselines3 import A2C
import os

models_dir = "models/A2C"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = gym.make('LunarLander-v2', render_mode='rgb_array')
env.reset()

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000
for i in range(1,30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{models_dir}/{i*TIMESTEPS}")

# episodes = 10
# for ep in range(episodes):
#     obs = env.reset()
#     done = False
#     while not done:
#         obs, rewards, done, trunc, info = env.step(env.action_space.sample())

env.close()