import argparse
import os
import pickle
import random
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.append('..')
from envs.myEnv2 import AirBattle
from algorithms.buffer import ReplayBuffer, ReservoirBuffer

np.random.seed(1)
tf.set_random_seed(1)

FIG_NUM = 0
EXPLORE = 10
ETA = 0.1  # nfsp choose best policy threshold
RANDOM_DECAY = 0.9
RANDOM_DECAY_GAP = 1000
MAX_EPISODES = 4000
MAX_EP_STEPS = 200
MULTI_STEPS = 1  # GAE steps
MEMORY_CAPACITY = 100000
BATCH_SIZE = 128
LR_SL = 0.001  # learning rate for SL net
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

parser = argparse.ArgumentParser()
parser.add_argument('--eta', type=float, default=ETA)
parser.add_argument('--multi_steps', type=int, default=MULTI_STEPS)
parser.add_argument('--episode', type=int, default=MAX_EPISODES)
parser.add_argument('--fig', type=int, default=FIG_NUM)
parser.add_argument('--explore', type=float, default=EXPLORE)
parser.add_argument('--decay', type=float, default=RANDOM_DECAY)
parser.add_argument('--gap', type=int, default=RANDOM_DECAY_GAP)
parser.add_argument('--batch', type=int, default=BATCH_SIZE)
parser.add_argument('--esteps', type=int, default=MAX_EP_STEPS)
parser.add_argument('--memory', type=int, default=MEMORY_CAPACITY)
parser.add_argument('--lrsl', type=float, default=LR_SL)
parser.add_argument('--lro', type=float, default=LR_O)
parser.add_argument('--lra', type=float, default=LR_A)
parser.add_argument('--lrc', type=float, default=LR_C)
parser.add_argument('--gamma', type=float, default=GAMMA)
parser.add_argument('--r_factor', type=float, default=RFACTOR)
parser.add_argument('--a_factor', type=float, default=AFACTOR)
args = parser.parse_args()


def print_args():
    print(
        '\nfig_num: {}\neta: {}\nmulti_steps: {}\nmax_episodes:{}\nexplore: {}\ndecay: {}\ngap: {}\nbatch: {}\nep_steps: {}\nmemory size: {}\nLR_SL: {}\nLR_O: {}\nLR_A: {}\nLR_C: {}\ngamma: {}\nr_factor: {}\na_factor: {}\n'.format(
            args.fig, args.eta, args.multi_steps, args.episode, args.explore,
            args.decay, args.gap, args.batch,
            args.esteps,
            args.memory,
            args.lrsl, args.lro, args.lra, args.lrc,
            args.gamma, args.r_factor, args.a_factor))


# GAE reward
def multi_step_reward(rewards, gamma):
    res = 0.
    for idx, reward in enumerate(rewards):
        res += reward * (gamma ** idx)
    return res


# exchange state order from friend to enemy
def exchange_order(state, num_friend, num_enemy, agent_features):
    new_state = np.zeros(state.shape)
    new_state[:num_enemy * agent_features] = state[num_friend * agent_features:(
                                                                                           num_friend + num_enemy) * agent_features]
    new_state[
    num_enemy * agent_features:(num_enemy + num_friend) * agent_features] = state[
                                                                            :num_friend * agent_features]
    new_state[(num_friend + num_enemy) * agent_features:] = state[(
                                                                              num_friend + num_enemy) * agent_features:]
    return new_state


def split_batch(b_M, state_dim, option_dim, action_dim):
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
    return b_s, b_o, b_ov, b_a, b_r, b_s_, b_o_, b_ov_


class Option(object):
    def __init__(self, name, sess, option_dim, state_dim, learning_rate=0.001):
        self.sess = sess
        self.name = name
        self.op_dim = option_dim
        self.lr = learning_rate

        # single structure
        with tf.variable_scope(self.name + '/option') as scope:
            self.s = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
            self.s_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')
            self.o_v = self._build_net(self.s)
            self.o = tf.random.categorical(self.o_v, 1)
            scope.reuse_variables()
            self.o_v_ = self._build_net(self.s_)
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
    def __init__(self, name, sess, option_dim, action_dim, state_dim, action_bound,
                 s, s_,
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

        with tf.variable_scope(self.name + '/Actor'):
            self.s = s
            self.s_ = s_
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
    def __init__(self, name, sess, option_dim, state_dim, action_dim, gamma, a, a_,
                 s, s_,
                 o_v, o_v_, learning_rate,
                 replacement):
        self.sess = sess
        self.name = name
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.op_dim = option_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        with tf.variable_scope(self.name + '/Critic'):
            self.s = s
            self.r = tf.placeholder(tf.float32, [None, 1], name='r')
            self.s_ = s_
            self.o_v = o_v
            self.o_v_ = o_v_
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


class Policy(object):
    def __init__(self, name, sess, state_dim, action_dim, action_bound,
                 learning_rate=0.001):
        self.sess = sess
        self.name = name
        self.lr = learning_rate
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        with tf.variable_scope(self.name + 'policy') as scope:
            self.s = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
            self.sl_action = tf.placeholder(tf.float32, shape=[None, action_dim],
                                            name='a')
            self.action = self._build_net(self.s, scope='action', trainable=True)

            with tf.variable_scope('train'):
                self.loss = tf.reduce_mean(
                    tf.squared_difference(self.sl_action, self.action))
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _build_net(self, s, scope='net', trainable=True):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            nl1 = 50
            nl2 = 30

            h1 = tf.layers.dense(s, nl1, activation=tf.nn.relu,
                                 kernel_initializer=init_w,
                                 bias_initializer=init_b,
                                 name='h1')
            h2 = tf.layers.dense(h1, nl2, activation=tf.nn.relu,
                                 kernel_initializer=init_w,
                                 bias_initializer=init_b,
                                 name='h2')
            action = tf.layers.dense(h2, self.action_dim, activation=tf.nn.tanh,
                                     kernel_initializer=init_w,
                                     bias_initializer=init_b,
                                     name='output')
            scaled_a = tf.multiply(action, self.action_bound, name='scaled_a')
        return scaled_a

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.action, {self.s: s})[0]

    def learn(self, s, a):
        self.sess.run(self.train_op, {self.s: s, self.sl_action: a})


if __name__ == '__main__':
    env = AirBattle()
    env.reinforce_enemy(factor=args.a_factor)
    option_dim = 2
    friend_num = len(env.friend)
    enemy_num = len(env.enemy)
    agent_features = env.n_space_dim
    state_dim = env.n_features
    action_dim = env.n_actions
    action_bound = env.action_bound  # action的激活函数是tanh
    # offense = Offense(action_bound, 0, args.a_factor)

    sess = tf.Session()
    # agent1 network
    policy1 = Policy('agent1', sess, state_dim, action_dim, action_bound, args.lrsl)
    option1 = Option('agent1', sess, option_dim, state_dim, args.lro)
    actor1 = Actor('agent1', sess, option_dim, action_dim, state_dim, action_bound,
                   option1.s, option1.s_, option1.o,
                   option1.o_, args.lra, REPLACEMENT)
    critic1 = Critic('agent1', sess, option_dim, state_dim, action_dim, args.gamma,
                     actor1.a, actor1.a_, option1.s, option1.s_, option1.o_v,
                     option1.o_v_,
                     args.lrc,
                     REPLACEMENT)
    actor1.add_grad_to_graph(critic1.a_grads)
    option1.add_grad_to_graph(critic1.o_grads)

    # agent2 network
    policy2 = Policy('agent2', sess, state_dim, action_dim, action_bound, args.lrsl)
    option2 = Option('agent2', sess, option_dim, state_dim, args.lro)
    actor2 = Actor('agent2', sess, option_dim, action_dim, state_dim, action_bound,
                   option2.s, option2.s_, option2.o,
                   option2.o_, args.lra, REPLACEMENT)
    critic2 = Critic('agent2', sess, option_dim, state_dim, action_dim, args.gamma,
                     actor2.a, actor2.a_, option2.s, option2.s_, option2.o_v,
                     option2.o_v_,
                     args.lrc,
                     REPLACEMENT)
    actor2.add_grad_to_graph(critic2.a_grads)
    option2.add_grad_to_graph(critic2.o_grads)
    sess.run(tf.global_variables_initializer())

    # replay buffer and reservior buffer
    Replay1 = ReplayBuffer(args.memory,
                           dims=2 * (
                                   state_dim + 1 + option_dim) + action_dim + 1)  # (s, o, o_v)
    Replay2 = ReplayBuffer(args.memory,
                           dims=2 * (state_dim + 1 + option_dim) + action_dim + 1)

    Reservoir1 = ReservoirBuffer(args.memory)
    Reservoir2 = ReservoirBuffer(args.memory)

    # logger
    mr = deque(maxlen=200)
    mr_shaping = deque(maxlen=200)
    all_ep_r = []
    all_ep_r_shaping = []

    for i in range(MAX_EPISODES):
        if i % 50 == 0:
            print_args()

        # if RENDER:
        #     env.render()

        s_f, info = env.reset()
        s_e = exchange_order(s_f, friend_num, enemy_num, agent_features)

        ep_reward = 0
        ep_reward_shaping = 0

        for j in range(args.esteps):
            is_best_response = False

            o1 = option1.get_option(s_f)
            o_v1 = option1.get_option_value(s_f)
            o2 = option2.get_option(s_e)
            o_v2 = option2.get_option_value(s_e)

            # Action selection is decided by a combination of best response and average strategy
            if random.random() > args.eta:
                p1_action = policy1.choose_action(s_f)
                p2_action = policy2.choose_action(s_e)

            else:
                is_best_response = True

                p1_action = actor1.choose_action(s_f, o1[np.newaxis, :])
                p1_action = np.clip(np.random.normal(p1_action, args.explore), -2,
                                    2)  # add randomness for exploration

                p2_action = actor2.choose_action(s_e, o2[np.newaxis, :])
                p2_action = np.clip(np.random.normal(p2_action, args.explore), -2,
                                    2)  # add randomness for exploration

            # interaction with env
            s_f_, r, done, info = env.step(p1_action, p2_action, o1,
                                           o_v1)  # (o, o_v) not interact with env
            s_e_ = exchange_order(s_f_, friend_num, enemy_num, agent_features)

            o1_ = option1.get_option(s_f_)
            o_v1_ = option1.get_option_value(s_f_)
            o2_ = option2.get_option(s_e_)
            o_v2_ = option2.get_option_value(s_e_)

            # reward shaping based on o
            # o=0 encouraging offense, while o=1 encouraging defense
            r1 = r
            if (o1[0] == 0 and r1 < 0) or (o1[0] == 1 and r1 > 0):
                r1 /= args.r_factor

            r2 = -r
            if (o2[0] == 0 and r2 < 0) or (o2[0] == 1 and r2 > 0):
                r2 /= args.r_factor

            # store buffer
            Replay1.store_transition(s_f, o1, o_v1, p1_action, r1, s_f_, o1_, o_v1_)
            Replay2.store_transition(s_e, o2, o_v2, p2_action, r2, s_e_, o2_, o_v2_)

            if is_best_response:
                Reservoir1.store_transition(s_f, p1_action)
                Reservoir2.store_transition(s_e, p2_action)

            # training
            if Replay1.pointer == args.memory:
                print('\nBegin training\n')

            if Replay1.pointer > args.memory:
                if Replay1.pointer % args.gap == 0:
                    args.explore *= args.decay  # decay the action randomness

                b_M1 = Replay1.sample(BATCH_SIZE)
                b_s1, b_o1, b_ov1, b_a1, b_r1, b_s1_, b_o1_, b_ov1_ = split_batch(
                    b_M1, state_dim, option_dim, action_dim)
                option1.learn(b_s1)
                critic1.learn(b_s1, b_ov1, b_a1, b_r1, b_s1_, b_ov1_)
                actor1.learn(b_s1, b_o1)

                b_M2 = Replay2.sample(BATCH_SIZE)
                b_s2, b_o2, b_ov2, b_a2, b_r2, b_s2_, b_o2_, b_ov2_ = split_batch(
                    b_M2, state_dim, option_dim, action_dim)
                option2.learn(b_s2)
                critic2.learn(b_s2, b_ov2, b_a2, b_r2, b_s2_, b_ov2_)
                actor2.learn(b_s2, b_o2)

                b_s1, b_a1 = Reservoir1.sample(BATCH_SIZE)
                policy1.learn(b_s1, b_a1)

                b_s2, b_a2 = Reservoir2.sample(BATCH_SIZE)
                policy2.learn(b_s2, b_a2)

            s_f = s_f_
            s_e = s_e_
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
    file_path = '{}/nfsp_{}.ckpt'.format(relative_path, file_name)
    saver = tf.train.Saver()
    saver.save(sess, file_path)
    print('Session has been saved sucessfully in {}'.format(file_path))

    # save data as pkl
    d = {"mean episode reward": all_ep_r,
         "mean episode shaping reward": all_ep_r_shaping}
    with open(os.path.join(relative_path, "nfsp_data_{}.pkl".format(file_name)),
              "wb") as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
