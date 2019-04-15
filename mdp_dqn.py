import gym
import numpy as np

env = gym.make("FrozenLake-v0")

state = env.reset()

print(state)
print()
env.render()
print()

print(env.action_space)
print(env.observation_space)
print("Number of action_space: " + str(env.action_space.n))
print("Number of observation_space: " + str(env.observation_space.n))

# init all Utilities of all states with zero
U = np.zeros([env.observation_space.n])
U[15] = 1  # goal state, only for frozenlake-v0
U[[5, 7, 11, 12]] = -1  # hole state, only for frozenlake-v0
terminal_states = [5, 7, 11, 12, 15]  # terminal states, only for frozenlake-v0

discount_factor = 0.8
epsilon = 1e-3  # threshold if the learning difference, previous_U - U goes below this value will break the learning

i = 0
while True:
    i += 1
    previous_U = np.copy(U)

    for state_index in range(env.observation_space.n):
        U_value_table = [] # = Q_{sa}
        for action_index in range(env.action_space.n):
            U_value_for_action = 0
            for p, next_state, reward, _ in env.env.P[state_index][action_index]:
                U_value_for_action += p * (reward + discount_factor * previous_U[next_state])
            U_value_table.append(U_value_for_action)

        if state_index not in terminal_states:
            U[state_index] = max(U_value_table)

    if np.sum(np.fabs(previous_U - U)) <= epsilon:
        print("Value iteration converges at iteration #{0}".format(i+1))
        break

print("After learning completion printing the utilities for each states below from state id 0-15")
print()
print(U[:4])
print(U[4:8])
print(U[8:12])
print(U[12:16])
