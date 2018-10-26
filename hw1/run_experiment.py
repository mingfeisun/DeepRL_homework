#!/usr/bin/env python
import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def policy_fn(obs):
    with open('bc_policy/bc_weights.pkl', 'rb') as f:
        bc_weights = pickle.loads(f.read())
    with open('bc_policy/bc_biases.pkl', 'rb') as f:
        bc_biases = pickle.loads(f.read())

    obs_bo = tf.placeholder(tf.float32, [None, None])
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(obs_bo, bc_weights['h1']), bc_biases['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, bc_weights['h2']), bc_biases['b2']))
    a_ba = tf.matmul(layer_2, bc_weights['out']) + bc_biases['out']

    policy_fn = tf_util.function([obs_bo], a_ba)
    return policy_fn(obs)

def main():
    print('loading and building expert policy')
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make("Hopper-v2")
        max_steps = env.spec.timestep_limit

        returns = []
        observations = []
        actions = []

        obs = env.reset()
        done = False
        steps = 0
        # while not done:
        while done:
            action = policy_fn(obs[None,:])
            obs, r, done, _ = env.step(action)
            steps += 1
            env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break

if __name__ == '__main__':
    main()
