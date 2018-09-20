import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
import cv2

import argparse
import struct
import sys
import copy

import rospy
import rospkg

from replay_buffer import ReplayBuffer

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

RANDOM_SEED = 12451

# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, 84, 84, 1])
        network = tflearn.conv_2d(inputs, 16, 8, activation='relu', strides=4)
        #network = tflearn.max_pool_2d(network, 2, strides=2)
        network = tflearn.conv_2d(network, 32, 4, activation='relu', strides=2)
        #network = tflearn.conv_2d(network, 128, 3, activation='relu')
        #network = tflearn.conv_2d(network, 128, 3, activation='relu')
        #network = tflearn.conv_2d(network, 256, 3, activation='relu', strides=2)
        #network = tflearn.max_pool_2d(network, 2, strides=2)
        network = tflearn.fully_connected(network, 100, activation='relu')
        #network = tflearn.dropout(network,0.8)
        #network = tflearn.fully_connected(network, 300, activation='relu')
        #network = tflearn.fully_connected(network, 100, activation='relu')
        #network = tflearn.dropout(network,0.8)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        # out = tflearn.fully_connected(network, self.a_dim, activation='sigmoid', weights_init=w_init)
        out = tflearn.fully_connected(network, self.a_dim, activation='tanh')
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, 84, 84, 1])
        action = tflearn.input_data(shape=[None, self.a_dim])

        network = tflearn.conv_2d(inputs, 16, 8, activation='relu', strides=4)
        network = tflearn.conv_2d(network, 32, 4, activation='relu', strides=2)

        network = tflearn.fully_connected(network, 100, activation='relu')

        network = tflearn.merge([network, action], 'concat')
        # Add the action tensor in the 2nd hidden layer

        #network = tflearn.fully_connected(network, 300, activation='relu')
        #network = tflearn.fully_connected(network, 100, activation='relu')
        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        # out = tflearn.fully_connected(network, 1, weights_init=w_init)
        out = tflearn.fully_connected(network, 1)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# ===========================
#   Agent Training
# ===========================
class RLBrain:

    def __init__(self, bax_env, starting_angles, block_pose):
        self.env = bax_env
        self.starting_angles = starting_angles
        self.block_pose = block_pose


    def train(self, sess, actor, critic, actor_noise, delete_model_func, load_model_func):

        # # Set up summary Ops
        # summary_ops, summary_vars = build_summaries()
        #
        sess.run(tf.global_variables_initializer())
        # writer = tf.summary.FileWriter('./results/tf_ddpg',  sess.graph)

        # Initialize target network weights
        actor.update_target_network()
        critic.update_target_network()

        # Initialize replay memory
        replay_buffer = ReplayBuffer(1000000, RANDOM_SEED)
        SUCCESSFUL_GRASPS = 0

        for i in range(5000):
            #delete_model_func()
            #load_model_func()
            print 'Episode ', i
            delete_model_func()
            load_model_func()
            s_ = self.env.reset(self.starting_angles)
            if s_ is None:
                print('No images supplied')
            #s = self.env.reset()
            ep_reward = 0
            ep_ave_max_q = 0
            # Added exploration noise
            #a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            a = actor.predict(np.reshape(s_, (1, 84, 84, 1))) + actor_noise()
            print '-----------Predicted Joint Angles--------------------'
            print a
            print '-----------------------------------------------------'
            self.block_pose.orientation = Quaternion(x=a[0][0], y=a[0][1], z=a[0][2], w=a[0][3])
            #IDEAL values
            #self.block_pose.orientation = Quaternion(x=-0.0249590815779,y=0.999649402929, z=0.00737916180073, w=0.00486450832011)
            r, terminal, info = self.env.step(self.block_pose)
            if r == 1:
                SUCCESSFUL_GRASPS += 1
            print '------------------SUCCESSFUL_GRASPS-----------'
            print SUCCESSFUL_GRASPS
            print '----------------------------------------------'
            #replay_buffer.add(np.reshape(s_, (100,100,1,)), np.reshape(a, (4,)), r, terminal, np.reshape(s_, (100,100,1,)))
            replay_buffer.add(np.reshape(s_, (84, 84, 1,)), np.reshape(a, (4,)), r, terminal)
            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > 64:
                s_batch, a_batch, r_batch, t_batch, = replay_buffer.sample_batch(32)

                # Calculate targets
                #target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(32):
                    y_i.append(r_batch[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (32, 1)))
                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])
                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

    def run(self, delete_func, load_func):
        with tf.Session() as sess:
            np.random.seed(RANDOM_SEED)
            tf.set_random_seed(RANDOM_SEED)

            state_dim = (100,100)
            action_dim = 4
            action_bound = 1.0
            # Ensure action bound is symmetric
            #assert (self.env.action_space.high == -self.env.action_space.low)

            actor = ActorNetwork(sess, state_dim, action_dim, action_bound, 0.0001, 0.01, 32)

            critic = CriticNetwork(sess, state_dim, action_dim, 0.001, 0.001, 0.99, actor.get_num_trainable_vars())

            actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

            self.train(sess, actor, critic, actor_noise, delete_func, load_func)


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    RL = RLBrain(env)
    RL.rl()
