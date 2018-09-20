import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque
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

RANDOM_SEED = 1234

class DQN:
    def __init__(self, env):
        self.env     = env
        self.memory  = deque(maxlen=2000)
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(8, 8), activation='relu', input_shape=(84, 84, 1)))
        model.add(Conv2D(32, kernel_size=(4, 4), activation='relu'))
        model.add(Dense(400, activation="relu"))
        model.add(Dense(300, activation="relu"))
        model.add(Dense(self.a_dim))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            random_action = np.random.uniform(low=-1, high=1.0, size=(4))
            return Quaternion(x=a[0][0], y=a[0][1], z=a[0][2], w=a[0][3])
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

class RLBrain:
    def __init__(self, bax_env, starting_angles, block_pose):
        self.env = bax_env
        self.starting_angles = starting_angles
        self.block_pose = block_pose

    def train(self, delete_model_func, load_model_func):
        gamma   = 0.9
        epsilon = .95
        trials  = 5000
        trial_len = 500

        # updateTargetNetwork = 1000
        dqn_agent = DQN(env=self.env)
        steps = []
        for trial in range(trials):
            for step in range(trial_len):
                cur_state = self.env.reset(self.starting_angles)
                action = dqn_agent.act(cur_state)
                reward, done, _ = self.env.step(action)
                dqn_agent.remember(cur_state, action[0], action[1], action[2], action[3], reward, done)

                dqn_agent.replay()       # internally iterates default (prediction) model
                dqn_agent.target_train() # iterates target model
                if done:
                    break
            if step >= 199:
                print("Failed to complete in trial {}".format(trial))
                if step % 10 == 0:
                    dqn_agent.save_model("trial-{}.model".format(trial))
            else:
                print("Completed in {} trials".format(trial))
                dqn_agent.save_model("success.model")
                break

if __name__ == "__main__":
    main()
