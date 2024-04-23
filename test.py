import gymnasium as gym

env = gym.make('LunarLander-v2', render_mode="human")

env.reset()

for step in range(200):
    env.render()
    env.step(env.action_space.sample())

env.close()