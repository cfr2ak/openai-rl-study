import gym
import numpy as np
import tensorflow as tf

env = gym.make("FrozenLake-v0")

# defining network
tf.reset_default_graph()
inputs = tf.placeholder(shape=[None, env.observation_space.n], dtype=tf.float32)
weights = tf.get_variable(
    name="weights",
    dtype=tf.float32,
    shape=[env.observation_space.n, env.action_space.n],
    initializer=tf.contrib.layers.xavier_initializer()
    )
bias = tf.Variable(tf.zeros(shape=[env.action_space.n]), dtype=tf.float32)

# contains q_value of the actions at current state 'state'
predicted_q_tf = tf.add(tf.matmul(inputs, weights), bias)
predicted_action_tf = tf.argmax(predicted_q_tf, 1)

target_q_tf = tf.placeholder(shape=[1, env.action_space.n], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(target_q_tf - predicted_q_tf))

trainer = tf.train.AdamOptimizer(learning_rate=0.001)
minimizer = trainer.minimize(loss)

# training network
init = tf.global_variables_initializer()
discount_factor = 0.5
epsilon = 0.3
episodes = 10000

with tf.Session() as sess:
    sess.run(init)
    for i in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            predicted_action, predicted_q = sess.run(
                [predicted_action_tf, predicted_q_tf],
                feed_dict={inputs: np.identity(env.observation_space.n)[state:state + 1]}
            )

            if np.random.uniform(low=0, high=1) < epsilon:
                predicted_action = env.action_space.sample()

            next_state, reward, terminate, _ = env.step(predicted_action[0])

            if reward == 0:
                if terminate:
                    reward = -5
                else:
                    reward = -1
            elif reward == 1:
                reward = 5

            new_predicted_q = sess.run(
                predicted_q_tf,
                feed_dict={inputs: np.identity(env.observation_space.n)[next_state:next_state + 1]}
                )

            # update q_target to predicted_q
            target_q = predicted_q
            max_predicted_q_number = np.max(new_predicted_q)
            target_q[0, predicted_action[0]] = reward + discount_factor * max_predicted_q_number

            _ = sess.run(minimizer, feed_dict={
                    inputs: np.identity(env.observation_space.n)[state:state + 1],
                    target_q_tf: target_q
                    })
            state = next_state

            if terminate:
                break

    print("output after learning")
    print()
    state = env.reset()
    env.render()
    while True:
        action = sess.run(
            predicted_action_tf,
            feed_dict={inputs: np.identity(env.observation_space.n)[state:state + 1]}
        )
        next_state, reward, terminate, _ = env.step(action[0])

        print("=" * 10)
        env.render()
        state = next_state
        if terminate:
            break
