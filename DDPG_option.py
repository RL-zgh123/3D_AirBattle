import argparse
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from myEnv import AirBattle

np.random.seed(1)
tf.set_random_seed(1)

#####################  hyper parameters  ####################

EXPLORE = 10
RANDOM_DECAY = 0.9
RANDOM_DECAY_GAP = 1000
MAX_EPISODES = 3000
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128
LR_O = 0.001  # learning rate for option
LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.99  # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][1]  # you can try different target replacement strategies
RENDER = False
OUTPUT_GRAPH = True

parser = argparse.ArgumentParser()
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
args = parser.parse_args()


def print_args():
    print(
        '\nexplore: {}\ndecay: {}\ngap: {}\nbatch: {}\nep_steps: {}\nmemory size: {}\nLR_O: {}\nLR_A: {}\nLR_C: {}\ngamma: {}\n'.format(
            args.explore, args.decay, args.gap, args.batch, args.esteps, args.memory,
            args.lro, args.lra, args.lrc, args.gamma))


class Option(object):
    def __init__(self, sess, option_dim, learning_rate):
        self.sess = sess
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
        with tf.variable_scope('option') as scope:
            self.o_v = self._build_net(self.s)
            print('self.o_v', self.o_v)
            # self.o = tf.reshape(tf.argmax(self.o_v, 1), [-1, 1])
            self.o = tf.random.categorical(tf.math.log(self.o_v), 1)
            print('self.o', self.o)
            scope.reuse_variables()
            self.o_v_ = self._build_net(self.s_)
            # self.o_ = tf.reshape(tf.argmax(self.o_v_, 1), [-1, 1])
            self.o_ = tf.random.categorical(tf.math.log(self.o_v_), 1)
        self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='option/net')

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
        print('o_v', self.sess.run(self.o_v, {self.s: s}))
        return self.sess.run(self.o, {self.s: s})


class Actor(object):
    def __init__(self, sess, option_dim, action_dim, action_bound, learning_rate, replacement, s, s_, o,
                 o_):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0
        self.op_dim = option_dim

        self.s = s
        self.s_ = s_

        with tf.variable_scope('Actor'):
            self.o = o
            self.o_ = o_
            self.a = self._build_net(self.s, self.o, scope='eval_net',
                                     trainable=True)
            print('self.a', self.a)
            self.a_ = self._build_net(self.s_, self.o_, scope='target_net',
                                      trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in
                                 zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t +
                                           self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, o, scope, trainable):
        self.option_onehot = tf.squeeze(tf.one_hot(o, self.op_dim, dtype=tf.float32), [1])
        # print('onehot', self.option_onehot)
        s = tf.reshape(s, [-1, 1, state_dim])
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            nl1 = 30

            w1 = tf.get_variable('w1', [self.op_dim, state_dim * nl1], initializer=init_w)
            b1 = tf.get_variable('b1', [self.op_dim, 1 * nl1], initializer=init_b)
            w1_onehot = tf.reshape(tf.matmul(self.option_onehot, w1), [-1, state_dim, nl1])
            b1_onehot = tf.reshape(tf.matmul(self.option_onehot, b1), [-1, 1, nl1])
            h1 = tf.nn.relu(tf.matmul(s, w1_onehot) + b1_onehot)

            with tf.variable_scope('a'):
                w2 = tf.get_variable('w2', [self.op_dim, nl1 * self.a_dim], initializer=init_w)
                b2 = tf.get_variable('b2', [self.op_dim, 1 * self.a_dim], initializer=init_b)
                w2_onehot = tf.reshape(tf.matmul(self.option_onehot, w2),
                                       [-1, nl1, self.a_dim])
                b2_onehot = tf.reshape(tf.matmul(self.option_onehot, b2),
                                       [-1, 1, self.a_dim])
                actions = tf.nn.tanh(tf.matmul(h1, w2_onehot) + b2_onehot)
                actions = tf.squeeze(actions, [1])
                scaled_a = tf.multiply(actions, self.action_bound,
                                       name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s, o):  # batch update
        self.sess.run(self.train_op, feed_dict={self.s: s, self.o: o})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, s, o):
        s = s[np.newaxis, :]  # single state
        return self.sess.run(self.a, feed_dict={self.s: s, self.o: o})[
            0]  # single action

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
    def __init__(self, sess, option_dim, state_dim, action_dim, learning_rate, gamma,
                 replacement, a, a_, s, s_, o_v, o_v_):
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

        with tf.variable_scope('Critic'):
            self.a = a
            self.q = self._build_net(self.s, self.a, self.o_v, 'eval_net', trainable=True)
            self.q_ = self._build_net(self.s_, a_, self.o_v_, 'target_net',
                                      trainable=False)  # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope='Critic/target_net')

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
                w1_o = tf.get_variable('w1_o', [self.op_dim, n_l1], initializer=init_w,
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

    def store_transition(self, s, o, a, r, s_, o_):
        # o [?,1] two dimensions, [[o]]
        transition = np.hstack((s, o[0], a, [r], s_, o_[0]))
        index = self.pointer % self.capacity
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


if __name__ == '__main__':
    env = AirBattle()
    option_dim = 2
    state_dim = env.n_features
    action_dim = env.n_actions
    action_bound = env.action_bound  # action的激活函数是tanh

    sess = tf.Session()
    option = Option(sess, option_dim, args.lro)
    actor = Actor(sess, option_dim, action_dim, action_bound, args.lra, REPLACEMENT, option.s, option.s_, option.o,
                  option.o_)
    critic = Critic(sess, option_dim, state_dim, action_dim, args.lrc, args.gamma, REPLACEMENT,
                    actor.a,
                    actor.a_, option.s, option.s_, option.o_v, option.o_v_)
    actor.add_grad_to_graph(critic.a_grads)
    option.add_grad_to_graph(critic.o_grads)
    sess.run(tf.global_variables_initializer())

    M = Memory(args.memory, dims=2 * (state_dim + 1) + action_dim + 1)  # (s, o)
    mr = deque(maxlen=200)
    all_ep_r = []

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    for i in range(MAX_EPISODES):
        if i % 50 == 0:
            print_args()

        if RENDER:
            env.render()

        s = env.reset()
        ep_reward = 0

        for j in range(args.esteps):
            o = option.get_option(s)
            a = actor.choose_action(s, o)
            a = np.clip(np.random.normal(a, args.explore), -2,
                        2)  # add randomness for exploration
            a0 = np.random.rand(action_dim)
            s_, r, done, info = env.step(a, a0)
            o_ = option.get_option(s_)

            # o=0 encouraging offense, while o=1 encouraging defense
            if (o == 0 and r < 0) or (o == 1 and r > 0):
                r /= 5
            M.store_transition(s, o, a, r, s_, o_)
            print('o', o, 'a', a)

            if M.pointer == args.memory:
                print('\nBegin training\n')

            if M.pointer > args.memory:
                if M.pointer % args.gap == 0:
                    args.explore *= args.decay  # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :state_dim]
                b_o = b_M[:, state_dim: state_dim + 1]
                b_a = b_M[:, state_dim + 1: state_dim + 1 + action_dim]
                b_r = b_M[:, -state_dim - 2: -state_dim - 1]
                b_s_ = b_M[:, -state_dim - 1:-1]
                b_o_ = b_M[:, -1:]

                option.learn(b_s)
                critic.learn(b_s, b_o, b_a, b_r, b_s_, b_o_)
                actor.learn(b_s, b_o)

            s = s_
            ep_reward += r

            if j == args.esteps - 1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward),
                      'Mean reward: %.2f' % np.round(np.mean(list(mr)), 2), )

        mr.append(ep_reward)
        all_ep_r.append(np.round(np.mean(list(mr)), 2))

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()
