import argparse
import numpy as np
import tensorflow as tf
from myEnv import AirBattle
from collections import deque

np.random.seed(1)
tf.set_random_seed(1)


#####################  hyper parameters  ####################

EXPLORE = 10
RANDOM_DECAY = 0.99
RANDOM_DECAY_GAP = 1000
MAX_EPISODES = 10000
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 100000
BATCH_SIZE = 128
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
parser.add_argument('--lra', type=float, default=LR_A)
parser.add_argument('--lrc', type=float, default=LR_C)
parser.add_argument('--gamma', type=float, default=GAMMA)
args = parser.parse_args()


def print_args():
    print(
        '\nexplore: {}\ndecay: {}\ngap: {}\nbatch: {}\nep_steps: {}\nmemory size: {}\nLR_A: {}\nLR_C: {}\ngamma: {}\n'.format(
            args.explore, args.decay, args.gap, args.batch, args.esteps, args.memory,
            args.lra, args.lrc, args.gamma))


###############################  Actor  ####################################


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        self.s = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
        self.r = tf.placeholder(tf.float32, [None, 1], name='r')
        self.s_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')

        with tf.variable_scope('Actor'):
            self.a = self._build_net(self.s, scope='eval_net', trainable=True)
            self.a_ = self._build_net(self.s_, scope='target_net', trainable=False)

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

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s, 30, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l1',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh,
                                          kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a',
                                          trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound,
                                       name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):  # batch update
        self.sess.run(self.train_op, feed_dict={self.s: s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]  # single state
        return self.sess.run(self.a, feed_dict={self.s: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # self.a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params,
                                             grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(
                -self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(
                zip(self.policy_grads, self.e_params))


###############################  Critic  ####################################

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma,
                 replacement, a, a_, s, s_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        self.s = s
        self.r = tf.placeholder(tf.float32, [None, 1], name='r')
        self.s_ = s_

        with tf.variable_scope('Critic'):
            self.a = a
            self.q = self._build_net(self.s, self.a, 'eval_net', trainable=True)
            self.q_ = self._build_net(self.s_, a_, 'target_net',
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
            self.a_grads = tf.gradients(self.q, a)[
                0]  # tensor of gradients of each sample (None, a_dim)

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in
                                     zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t +
                                               self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1],
                                       initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1],
                                       initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b,
                                     trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w,
                                    bias_initializer=init_b,
                                    trainable=trainable)  # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op,
                      feed_dict={self.s: s, self.a: a, self.r: r, self.s_: s_})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1


#####################  Memory  ####################

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

if __name__ == '__main__':
    env = AirBattle()
    state_dim = env.n_features
    action_dim = env.n_actions
    action_bound = env.action_bound  # action的激活函数是tanh

    sess = tf.Session()
    actor = Actor(sess, action_dim, action_bound, args.lra, REPLACEMENT)
    critic = Critic(sess, state_dim, action_dim, args.lrc, args.gamma, REPLACEMENT, actor.a,
                    actor.a_, actor.s, actor.s_)
    actor.add_grad_to_graph(critic.a_grads)
    sess.run(tf.global_variables_initializer())

    M = Memory(args.memory, dims=2 * state_dim + action_dim + 1)
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
            # Add exploration noise
            a = actor.choose_action(s)
            a = np.clip(np.random.normal(a, args.explore), -2,
                        2)  # add randomness to action selection for exploration

            a0 = np.random.rand(action_dim)
            s_, r, done, info = env.step(a, a0)
            M.store_transition(s, a, r, s_)

            if M.pointer > args.memory:
                if M.pointer % args.gap == 0:
                    args.explore *= args.decay  # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :state_dim]
                b_a = b_M[:, state_dim: state_dim + action_dim]
                b_r = b_M[:, -state_dim - 1: -state_dim]
                b_s_ = b_M[:, -state_dim:]

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)

            s = s_
            ep_reward += r

            if j == args.esteps - 1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward),
                      'Mean reward: %.2f' % np.round(np.mean(list(mr)), 2), )

        mr.append(ep_reward)
        if i == 0:
            all_ep_r.append(ep_reward)
        else:
            all_ep_r.append(all_ep_r[-1]*0.9+ep_reward*0.1)

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()
