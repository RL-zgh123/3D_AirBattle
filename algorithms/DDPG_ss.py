import tensorflow as tf
import numpy as np
import gym

np.random.seed(1)
tf.set_random_seed(1)

#####################  hyper parameters  ####################

MAX_EPISODES = 500
MAX_EP_STEPS = 200
LR_S = 0.0002  # learning rate for assistNet
LR_A = 0.002  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][1]  # you can try different target replacement strategies
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
OUTPUT_GRAPH = True
ENV_NAME = 'Pendulum-v0'


class AssistNet(object):
    def __init__(self, sess, state_dim, phi_dim, action_dim, learning_rate):
        self.sess = sess
        self.a_dim = action_dim
        self.s_dim = state_dim
        self.p_dim = phi_dim
        self.lr = learning_rate

        self.init_w = tf.random_normal_initializer(0., 0.3)
        self.init_b = tf.constant_initializer(0.1)

        # build placeholder
        self._build_ph()

        # build phi(s)
        with tf.variable_scope("phi") as scope:
            self.phi_s = self._build_phi(self.s)
            scope.reuse_variables()
            self.phi_s_ = self._build_phi(self.s_)

        # predict action
        with tf.variable_scope("predict_action") as scope:
            self.pre_a = self._predict_action()

        self._build_train_op()

    def _build_ph(self):
        self.s = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
        self.s_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')
        self.a = tf.placeholder(tf.float32, shape=[None, action_dim], name='a')

    def _build_phi(self, s, scope='net0', trainable=True):
        with tf.variable_scope(scope) as scope:
            phi = tf.layers.dense(s, self.p_dim, activation=tf.nn.relu,
                                  kernel_initializer=self.init_w,
                                  bias_initializer=self.init_b,
                                  name='l1',
                                  trainable=trainable)
        return phi

    def _predict_action(self, scope='net0', trainable=True):
        with tf.variable_scope(scope) as scope:
            unit_phi_s = tf.concat([self.phi_s, self.phi_s_], axis=1)
            pre_a = tf.layers.dense(unit_phi_s, self.a_dim, activation=tf.nn.relu,
                                    kernel_initializer=self.init_w,
                                    bias_initializer=self.init_b,
                                    name='l2',
                                    trainable=trainable
                                    )
        return pre_a

    def _build_train_op(self):
        self.loss = tf.reduce_mean(tf.squared_difference(self.a, self.pre_a))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def learn(self, s, s_, a):
        self.sess.run(self.train_op, {self.s: s, self.s_: s_, self.a: a})

    def get_phi(self, s):
        return self.sess.run(self.phi_s, {self.s: s[np.newaxis, :]})


class Actor(object):
    def __init__(self, sess, phi_dim, action_dim, action_bound, learning_rate,
                 replacement):
        self.sess = sess
        self.a_dim = action_dim
        self.p_dim = phi_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        self._build_ph()

        with tf.variable_scope('Actor'):
            self.a = self._build_net(self.phi_s, scope='eval_net', trainable=True)
            self.a_ = self._build_net(self.phi_s_, scope='target_net',
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

    def _build_ph(self):
        self.phi_s = tf.placeholder(tf.float32, [None, self.p_dim], name='phi_s')
        self.phi_s_ = tf.placeholder(tf.float32, [None, self.p_dim], name='phi_s_')

    def _build_net(self, phi_s, scope='net0', trainable=True):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            # net = tf.layers.dense(s, 30, activation=tf.nn.relu,
            #                       kernel_initializer=init_w, bias_initializer=init_b, name='l1',
            #                       trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(phi_s, self.a_dim, activation=tf.nn.tanh,
                                          kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a',
                                          trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound,
                                       name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, phi_s):  # batch update
        self.sess.run(self.train_op, feed_dict={self.phi_s: phi_s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, phi_s):
        return self.sess.run(self.a, feed_dict={self.phi_s: phi_s})[
            0]  # single action

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


class Critic(object):
    def __init__(self, sess, phi_dim, action_dim, learning_rate, gamma, replacement,
                 a, a_, phi, phi_):
        self.sess = sess
        self.p_dim = phi_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        self.phi_s = phi
        self.phi_s_ = phi_
        self.r = tf.placeholder(tf.float32, [None, 1], name='r')

        with tf.variable_scope('Critic'):
            self.a = a
            self.q = self._build_net(self.phi_s, self.a, 'eval_net', trainable=True)
            self.q_ = self._build_net(self.phi_s_, a_, 'target_net',
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

    def _build_net(self, phi_s, a, scope='net0', trainable=True):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.p_dim, n_l1],
                                       initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1],
                                       initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b,
                                     trainable=trainable)
                net = tf.nn.relu(tf.matmul(phi_s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w,
                                    bias_initializer=init_b,
                                    trainable=trainable)  # Q(s,a)
        return q

    def learn(self, phi_s, a, r, phi_s_):
        self.sess.run(self.train_op,
                      feed_dict={self.phi_s: phi_s, self.a: a, self.r: r,
                                 self.phi_s_: phi_s_})
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

    def store_transition(self, s, phi_s, a, r, s_, phi_s_):
        transition = np.hstack((s, phi_s[0], a, [r], s_, phi_s_[0]))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


if __name__ == '__main__':

    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    state_dim = env.observation_space.shape[0]
    phi_dim = 30
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    sess = tf.Session()

    # Create assistnet, actor, critic.
    assist = AssistNet(sess, state_dim, phi_dim, action_dim, LR_S)
    actor = Actor(sess, phi_dim, action_dim, action_bound, LR_A, REPLACEMENT)
    critic = Critic(sess, phi_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a,
                    actor.a_, actor.phi_s, actor.phi_s_)
    actor.add_grad_to_graph(critic.a_grads)

    sess.run(tf.global_variables_initializer())

    M = Memory(MEMORY_CAPACITY, dims=2 * (state_dim + phi_dim) + action_dim + 1)

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    var = 3  # control exploration

    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0

        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            phi_s = assist.get_phi(s)
            a = actor.choose_action(phi_s)
            a = np.clip(np.random.normal(a, var), -2,
                        2)  # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)
            phi_s_ = assist.get_phi(s_)

            M.store_transition(s, phi_s, a, r / 10, s_, phi_s_)

            if M.pointer > MEMORY_CAPACITY:
                var *= .9995  # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :state_dim]
                b_ps = b_M[:, state_dim:state_dim + phi_dim]
                b_a = b_M[:, state_dim + phi_dim: state_dim + phi_dim + action_dim]
                b_r = b_M[:, -state_dim - phi_dim - 1: -state_dim - phi_dim]
                b_s_ = b_M[:, -state_dim - phi_dim:-phi_dim]
                b_ps_ = b_M[:, -phi_dim:]

                assist.learn(b_s, b_s_, b_a)
                actor.learn(b_ps)
                critic.learn(b_ps, b_a, b_r, b_ps_)

            s = s_
            ep_reward += r

            if j == MAX_EP_STEPS - 1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward),
                      'Explore: %.2f' % var, )
                if ep_reward > -300:
                    RENDER = False
                break
