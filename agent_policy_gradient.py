import numpy as np
import _pickle as pickle
import gym


n_hidden_nodes = 200
batch_size = 10
learning_rate = 1e-4
discount_factor = 0.99  # gamma
decay_rate = 0.99  # RMSProp Optimizer for Gradient descent
resume = False

# Policy neural network model initialize
input_dimension = 80 * 80
if resume:
    model = pickle.load(open('model.v', 'rb'))
else:
    model = {}
    # xavier initialization of weights
    model['W1'] = np.random.randn(n_hidden_nodes, input_dimension) * np.sqrt(2.0 / input_dimension)
    model['W2'] = np.random.randn(n_hidden_nodes) * np.sqrt(2.0 / n_hidden_nodes)
    gradient_buffer = {k: np.zeros_like(v) for k, v in model.items()}
    rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    x[x < 0] = 0
    return x


def preprocess(image):
    """
    :param image: 210 x 160 x 3 frame
    :return: 80 x 80 1D float vector
    """
    image = image[35:195]  # crop image
    image = image[::2, ::2, 0]  # downsample and reduce color form RGB to R only
    image[image == 144] = 0  # erase background type 1
    image[image == 109] = 0  # erase background type 2
    image[image != 0] = 1  # everything else (other than paddles and ball) to 1

    return image.astype('float').ravel()  # flattening to 1D


def discount_reward(reward):
    """
    :param reward: 1D flat array of rewards
    :return: discounted rewards
    """
    discount_reward = np.zeros_like(reward)
    sum_rewards = 0
    for t in reversed(range(0, reward.size)):
        if reward[t] != 0: # episode ends
            sum_rewards = 0
        sum_rewards = discount_factor * sum_rewards + reward[t]
        discount_reward[t] = sum_rewards
    return discount_reward


def policy_forward(x):
    """
    :param x: preprocessed image vector
    :return: probability of action
    """
    hidden = np.dot(model['W1'], x)
    hidden = relu(hidden)
    logit = np.dot(model['W2'], hidden)
    probability = sigmoid(logit)

    return probability, hidden

def policy_back_propagation(arr_hidden_state, gradient_logp, observation_values):
    """

    :param arr_hidden_state: array of intermediate hidden state values, [200 x 1]
    :param gradient_logp: the error, loss value[1 x 1]
    :param observation_values: observations to compute the derivatives with respect to different weight parameters
    :return: dictionary of delta Weights
    """
    dW2 = np.dot(arr_hidden_state.T, gradient_logp).ravel()
    dh = np.outer(gradient_logp, model['W2'])
    dh = relu(dh)
    dW1 = np.dot(dh.T, observation_values)

    return {'W1': dW1, 'W2': dW2}


env = gym.make("Pong-v0")
observation = env.reset()
previous_x = None
episode_hidden_layer_values = []
episode_observations = []
episode_gradient_log_ps = []
episode_rewards = []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
    env.render()
    current_x = preprocess(observation)

    if previous_x is None:
        previous_x = np.zeros(input_dimension)

    x = current_x - previous_x
    previous_x = current_x

    action_probability, hidden = policy_forward(x)

    if np.random.uniform() < action_probability:
        action = 2
    else:
        action = 3

    episode_observations.append(x)
    episode_hidden_layer_values.append(hidden)

    if action == 2:
        y = 1
    else:
        y = 0

    episode_gradient_log_ps.append(y - action_probability)

    observation, reward, done, info = env.step(action)
    reward_sum += reward
    episode_rewards.append(reward)

    if done:
        episode_number += 1

        arr_hidden_state = np.vstack(episode_hidden_layer_values)
        gradient_logp = np.vstack(episode_gradient_log_ps)
        observation_values = np.vstack(episode_observations)
        reward_values = np.vstack(episode_rewards)

        episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []

        discount_episoderewards = discount_reward(reward_values)
        discount_episoderewards = (discount_episoderewards - np.mean(discount_episoderewards)) / np.std(discount_episoderewards) # advantage

        gradient_logp *= discount_episoderewards

        grad = policy_back_propagation(arr_hidden_state, gradient_logp, observation_values)

        for layer in model:
            gradient_buffer[layer] += grad[layer]

        if episode_number % batch_size == 0:
            epsilon = 1e-5

            for weight in model.keys():
                g = gradient_buffer[weight]
                rmsprop_cache[weight] = decay_rate * rmsprop_cache[weight] + (1 - decay_rate) * g ** 2
                model[weight] += learning_rate * g / (np.sqrt(rmsprop_cache[weight]) + epsilon)
                gradient_buffer[weight] = np.zeros_like(model[weight])

        if running_reward is None:
            running_reward = reward_sum
        else:
            running_reward = running_reward * learning_rate + reward_sum * (1 - learning_rate)

        if episode_number % 100 == 0:
            pickle.dump(model, open('model.v', 'wb'))

        reward_sum = 0
        previous_x = None
        observation = env.reset()

    if reward != 0:
        print("Episodes {} ended with reward {}".format(episode_number, reward))

