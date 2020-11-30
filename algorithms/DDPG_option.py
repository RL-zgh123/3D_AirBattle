import argparse
import os
import pickle
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.append('..')
from envs.myEnv2 import AirBattle
from algorithms.Offense import Offense

np.random.seed(1)
tf.set_random_seed(1)

FIG_NUM = 0
EXPLORE = 10
RANDOM_DECAY = 0.9
RANDOM_DECAY_GAP = 1000
MAX_EPISODES = 4000
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 100000
BATCH_SIZE = 128
LR_O = 0.001  # learning rate for option
LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.99  # reward discount
RFACTOR = 10  # reward shaping factor
AFACTOR = 1  # offense action bound shaping factor
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][1]
RENDER = False
OUTPUT_GRAPH = True

parser = argparse.ArgumentParser()
parser.add_argument('--episode', type=int, default=MAX_EPISODES)
parser.add_argument('--fig', type=int, default=FIG_NUM)
parser.add_argument('--explore', type=float, default=EXPLORE)
parser.add_argument('--decay', type=float, default=RANDOM_DECAY)
parser.add_argument('--gap', type=int, default=RANDOM_DECAY_GAP)
parser.add_argument('--batch', type=int, default=BATCH_SIZE)
parser.add_argument('--esteps', type=int, default=MAX_EP_STEPS)
parser.add_argument('--memory', type=int, default=MEMORY_CAPACITY)
parser.add_argument('--lro', type=float, default=LR_O)
parser.add_argument('--lra', type=float, default=LR_A)
parser.add_argument('--lrc', type=float, default=LR_C)
parser.add_argument('--gamma', type=float, default=GAMMA)
parser.add_argument('--r_factor', type=float, default=RFACTOR)
parser.add_argument('--a_factor', type=float, default=AFACTOR)
args = parser.parse_args()


def print_args():
    print(
        '\nfig_num: {}\nmax_episodes:{}\nexplore: {}\ndecay: {}\ngap: {}\nbatch: {}\nep_steps: {}\nmemory size: {}\nLR_O: {}\nLR_A: {}\nLR_C: {}\ngamma: {}\nr_factor: {}\na_factor: {}\n'.format(
            args.fig, args.episode, args.explore, args.decay, args.gap, args.batch,
            args.esteps,
            args.memory,
            args.lro, args.lra, args.lrc, args.gamma, args.r_factor, args.a_factor))


class Option(object):
    def __init__(self, name, sess, option_dim, state_dim, learning_rate=0.001):
        self.sess = sess
        self.name = name
        self.op_dim = option_dim
        self.lr = learning_rate

        self.s = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
        self.s_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')

        # # double target structure
        # with tf.variable_scope('option'):
        #     self.o = self._build_net(self.s, 'eval_net', trainable=True)
        #     self.o_ = self._build_net(self.s_, 'target_net', trainable=False)
        #
        # self.e_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        #                                   scope='option/eval_net')
        # self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
        #                                   scope='option/target_net')

        # single structure
        with tf.variable_scope(self.name + '/option') as scope:
            self.o_v = self._build_net(self.s)
            # self.o = tf.random.categorical(tf.math.log(self.o_v), 1)
            self.o = tf.random.categorical(self.o_v, 1)
            scope.reuse_variables()
            self.o_v_ = self._build_net(self.s_)
            # self.o_ = tf.random.categorical(tf.math.log(self.o_v_), 1)
            self.o_ = tf.random.categorical(self.o_v_, 1)
        self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        scope=self.name + '/option/net')

    def _build_net(self, s, scope='net', trainable=True):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            h1 = tf.layers.dense(s, 30, activation=tf.nn.relu,
                                 kernel_initializer=init_w,
                                 bias_initializer=init_b,
                                 name='h1',
                                 trainable=trainable)
            values = tf.layers.dense(h1, self.op_dim,
                                     activation=tf.nn.sigmoid,
                                     kernel_initializer=init_w,
                                     bias_initializer=init_b,
                                     name='value',
                                     trainable=trainable)
        return values

    def learn(self, s):
        self.sess.run(self.train_op, {self.s: s})

    def add_grad_to_graph(self, o_grads):
        with tf.variable_scope('option_grads'):
            self.option_grads = tf.gradients(ys=self.o_v, xs=self.params,
                                             grad_ys=o_grads)

        with tf.variable_scope('O_train'):
            self.train_op = tf.train.AdamOptimizer(-self.lr).apply_gradients(
                zip(self.option_grads, self.params))

    def get_option(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.o, {self.s: s})[0]

    def get_option_value(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.o_v, {self.s: s})[0]


class Actor(object):
    def __init__(self, name, sess, option_dim, action_dim, state_dim, action_bound, s, s_,
                 o,
                 o_, learning_rate=0.001, replacement=REPLACEMENT):
        self.name = name
        self.sess = sess
        self.a_dim = action_dim
        self.s_dim = state_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0
        self.op_dim = option_dim

        self.s = s
        self.s_ = s_

        with tf.variable_scope(self.name + '/Actor'):
            self.o = o
            self.o_ = o_
            self.a = self._build_net(self.s, self.o, scope='eval_net',
                                     trainable=True)
            self.a_ = self._build_net(self.s_, self.o_, scope='target_net',
                                      trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          scope=self.name + '/Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          scope=self.name + '/Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in
                                 zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t +
                                           self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, o, scope, trainable):
        self.option_onehot = tf.squeeze(tf.one_hot(o, self.op_dim, dtype=tf.float32),
                                        [1])
        # print('onehot', self.option_onehot)
        s = tf.reshape(s, [-1, 1, self.s_dim])
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            nl1 = 30

            w1 = tf.get_variable('w1', [self.op_dim, self.s_dim * nl1],
                                 initializer=init_w)
            b1 = tf.get_variable('b1', [self.op_dim, 1 * nl1], initializer=init_b)
            w1_onehot = tf.reshape(tf.matmul(self.option_onehot, w1),
                                   [-1, self.s_dim, nl1])
            b1_onehot = tf.reshape(tf.matmul(self.option_onehot, b1), [-1, 1, nl1])
            h1 = tf.nn.relu(tf.matmul(s, w1_onehot) + b1_onehot)

            with tf.variable_scope('a'):
                w2 = tf.get_variable('w2', [self.op_dim, nl1 * self.a_dim],
                                     initializer=init_w)
                b2 = tf.get_variable('b2', [self.op_dim, 1 * self.a_dim],
                                     initializer=init_b)
                w2_onehot = tf.reshape(tf.matmul(self.option_onehot, w2),
                                       [-1, nl1, self.a_dim])
                b2_onehot = tf.reshape(tf.matmul(self.option_onehot, b2),
                                       [-1, 1, self.a_dim])
                actions = tf.nn.tanh(tf.matmul(h1, w2_onehot) + b2_onehot)
                actions = tf.squeeze(actions, [1])
                scaled_a = tf.multiply(actions, self.action_bound,
                                       name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s, o):
        self.sess.run(self.train_op, feed_dict={self.s: s, self.o: o})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, s, o):
        s = s[np.newaxis, :]
        return self.sess.run(self.a, feed_dict={self.s: s, self.o: o})[0]

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params,
                                             grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(
                -self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(
                zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, name, sess, option_dim, state_dim, action_dim, gamma, a, a_, s, s_,
                 o_v, o_v_, learning_rate,
                 replacement):
        self.name = name
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.op_dim = option_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        self.s = s
        self.r = tf.placeholder(tf.float32, [None, 1], name='r')
        self.s_ = s_
        self.o_v = o_v
        self.o_v_ = o_v_

        with tf.variable_scope(self.name + '/Critic'):
            self.a = a
            self.q = self._build_net(self.s, self.a, self.o_v, 'eval_net',
                                     trainable=True)
            self.q_ = self._build_net(self.s_, a_, self.o_v_, 'target_net',
                                      trainable=False)  # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope=self.name + '/Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope=self.name + '/Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = self.r + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]

        with tf.variable_scope('o_grads'):
            self.o_grads = tf.gradients(self.q, o_v)[0]

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in
                                     zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t +
                                               self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, o_v, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1],
                                       initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1],
                                       initializer=init_w, trainable=trainable)
                w1_o = tf.get_variable('w1_o', [self.op_dim, n_l1],
                                       initializer=init_w,
                                       trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b,
                                     trainable=trainable)
                net = tf.nn.relu(
                    tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + tf.matmul(o_v,
                                                                        w1_o) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w,
                                    bias_initializer=init_b,
                                    trainable=trainable)  # Q(s,a)
        return q

    def learn(self, s, o_v, a, r, s_, o_v_):
        self.sess.run(self.train_op,
                      feed_dict={self.s: s, self.a: a, self.r: r, self.s_: s_,
                                 self.o_v: o_v, self.o_v_: o_v_})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, o, o_v, a, r, s_, o_, o_v_):
        transition = np.hstack((s, o, o_v, a, [r], s_, o_, o_v_))
        index = self.pointer % self.capacity
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


if __name__ == '__main__':
    env = AirBattle()
    env.reinforce_enemy(factor=args.a_factor)
    option_dim = 2
    state_dim = env.n_features
    action_dim = env.n_actions
    action_bound = env.action_bound  # action的激活函数是tanh
    offense = Offense(action_bound, 0, args.a_factor)

    sess = tf.Session()
    option = Option('agent0', sess, option_dim, state_dim, args.lro)
    actor = Actor('agent0', sess, option_dim, action_dim, state_dim, action_bound,
                  option.s, option.s_, option.o,
                  option.o_, args.lra, REPLACEMENT)
    critic = Critic('agent0', sess, option_dim, state_dim, action_dim, args.gamma,
                    actor.a, actor.a_, option.s, option.s_, option.o_v, option.o_v_,
                    args.lrc,
                    REPLACEMENT)
    with tf.variable_scope('agent0'):
        actor.add_grad_to_graph(critic.a_grads)
        option.add_grad_to_graph(critic.o_grads)
    sess.run(tf.global_variables_initializer())
    # print('sess', sess)

    M = Memory(args.memory,
               dims=2 * (state_dim + 1 + option_dim) + action_dim + 1)  # (s, o, o_v)
    mr = deque(maxlen=200)
    mr_shaping = deque(maxlen=200)
    all_ep_r = []
    all_ep_r_shaping = []

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("../logs/", sess.graph)

    for i in range(MAX_EPISODES):
        if i % 50 == 0:
            print_args()

        if RENDER:
            env.render()

        s, info = env.reset()
        ep_reward = 0
        ep_reward_shaping = 0

        for j in range(args.esteps):
            o = option.get_option(s)
            o_v = option.get_option_value(s)
            a = actor.choose_action(s, o[np.newaxis, :])
            # print(s.shape, a.shape)
            a = np.clip(np.random.normal(a, args.explore), -2,
                        2)  # add randomness for exploration

            # a0 = np.random.rand(action_dim)
            a0 = offense.get_action(s, info)

            s_, r, done, info = env.step(a, a0, o, o_v)
            o_ = option.get_option(s_)
            o_v_ = option.get_option_value(s_)

            # reward shaping based on o
            # o=0 encouraging offense, while o=1 encouraging defense
            r1 = r
            if (o[0] == 0 and r < 0) or (o[0] == 1 and r > 0):
                r1 /= args.r_factor

            M.store_transition(s, o, o_v, a, r1, s_, o_, o_v_)

            if M.pointer == args.memory:
                print('\nBegin training\n')

            if M.pointer > args.memory:
                if M.pointer % args.gap == 0:
                    args.explore *= args.decay  # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :state_dim]
                b_o = b_M[:, state_dim: state_dim + 1]
                b_ov = b_M[:, state_dim + 1: state_dim + 1 + option_dim]
                b_a = b_M[:,
                      state_dim + 1 + option_dim: state_dim + 1 + option_dim + action_dim]
                b_r = b_M[:,
                      -option_dim - state_dim - 2: -option_dim - state_dim - 1]
                b_s_ = b_M[:, -option_dim - state_dim - 1:-option_dim - 1]
                b_o_ = b_M[:, -option_dim - 1:-option_dim]
                b_ov_ = b_M[:, -option_dim:]

                option.learn(b_s)
                critic.learn(b_s, b_ov, b_a, b_r, b_s_, b_ov_)
                actor.learn(b_s, b_o)

            s = s_
            ep_reward += r
            ep_reward_shaping += r1

            if j == args.esteps - 1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward),
                      'Mean reward: %.2f' % np.round(np.mean(list(mr)), 2),
                      'Mean shaping reward: %.2f' % np.round(
                          np.mean(list(mr_shaping)), 2))

        mr.append(ep_reward)
        mr_shaping.append(ep_reward_shaping)
        all_ep_r.append(np.round(np.mean(list(mr)), 2))
        all_ep_r_shaping.append(np.round(np.mean(list(mr_shaping)), 2))

    plt.figure()
    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.savefig('result_{}.jpg'.format(args.fig))

    plt.figure()
    plt.plot(np.arange(len(all_ep_r_shaping)), all_ep_r_shaping)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode shaping reward')
    plt.savefig('result_shaping_{}.jpg'.format(args.fig))

    plt.show()

    # save sess as ckpt
    relative_path = '../results'
    file_name = args.fig
    file_path = '{}/option_{}.ckpt'.format(relative_path, file_name)
    saver = tf.train.Saver()
    saver.save(sess, file_path)
    print('Session has been saved sucessfully in {}'.format(file_path))

    # save data as pkl
    d = {"mean episode reward": all_ep_r,
         "mean episode shaping reward": all_ep_r_shaping}
    with open(os.path.join(relative_path, "option_data_{}.pkl".format(file_name)),
              "wb") as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
