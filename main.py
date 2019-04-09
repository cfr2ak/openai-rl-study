import gym
import numpy as np
import time

env = gym.make("FrozenLake-v0")

state = env.reset()
print(state)

env.render()

env = env.unwrapped

print(env.action_space)
print(env.observation_space)


def epsilon_greedy(Q, s, n_action):
    # epsilon-greedy approach
    # for exploration and exploitation
    # of the state-action spaces
    epsilon = 0.3
    p = np.random.uniform(low = 0, high = 1)

    if p > epsilon:
        # return Action that has highest Q value given s
        return np.argmax(Q[s, :])
    else:
        return env.action_space.sample()


Q = np.zeros([env.observation_space.n, env.action_space.n])
learning_rate = 0.5
discount_factor = 0.9
total_episodes = 10000

for i in range(total_episodes):
    state = env.reset()
    terminate = False

    while True:
        action = epsilon_greedy(Q, state, env.action_space.n)
        next_state, reward, terminate, _ = env.step(action)

        if reward == 0:
            if terminate:
                # negative reward for hole
                reward = -5
                Q[next_state] = np.ones(env.action_space.n) * reward
            else:
                # reward negative to avoid long routes
                reward = -1
        elif reward == 1:
            reward = 100
            Q[next_state] = np.ones(env.action_space.n) * reward

        Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state, action]) - Q[state, action])
        state = next_state

        if terminate:
            break


print(Q)
state = env.reset()
env.render()

while True:
    action = np.argmax(Q[state])
    next_state, reward, terminate, _ = env.step(action)
    print()
    env.render()

    state = next_state

    if terminate:
        break
