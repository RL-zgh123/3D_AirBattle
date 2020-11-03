import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from myEnv import AirBattle

EP_MAX = 10000
EP_LEN = 1000
GAMMA = 0.9
A_LR = 0.001
C_LR = 0.002
BATCH = 128
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
ENV = AirBattle()
S_DIM, A_DIM = ENV.n_features, ENV.n_actions
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    dict(name='clip', epsilon=0.2),  # Clipped surrogate objective
][1]


class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)  # state-value
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.critic_opt = tf.train.AdamOptimizer(C_LR)
            self.ctrain_op = self.critic_opt.minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in
                                    zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv

            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -tf.reduce_mean(surr - self.tflam * kl)

            else:
                self.aloss = -tf.reduce_mean(
                    tf.minimum(surr, tf.clip_by_value(ratio, 1. - METHOD['epsilon'],
                                                1. + METHOD['epsilon']) * self.tfadv))

            with tf.variable_scope('atrain'):
                self.actor_opt = tf.train.AdamOptimizer(A_LR)
                self.atrain_op = self.actor_opt.minimize(
                    self.aloss)

            tf.summary.FileWriter("log/", self.sess.graph)

            self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage,
                            {self.tfs: s, self.tfdc_r: r})  # 得到advantage value

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run([self.atrain_op, self.kl_mean],
                                      {self.tfs: s, self.tfa: a,
                                       self.tfadv: adv,
                                       self.tflam: METHOD['lam']})
                if kl > 4 * METHOD['kl_target']:
                    break
                elif kl < METHOD[
                    'kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                    METHOD['lam'] /= 2
                elif kl > METHOD['kl_target'] * 1.5:
                    METHOD['lam'] *= 2
                METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)  # sometimes explode, this clipping is my solution

        else:  # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op,
                           {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in
             range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in
         range(C_UPDATE_STEPS)]




    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus,
                                    trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)  # 一个正态分布
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    # def get_gradients(self, s, a, r):
    #     adv = self.sess.run(self.advantage,
    #                         {self.tfs: s, self.tfdc_r: r})  # 得到advantage value
    #     self.actor_grads_and_vars_op = \
    #     self.actor_opt.compute_gradients(self.aloss, tf.trainable_variables(scope = 'pi'))
    #     self.critic_grads_and_vars_op = \
    #     self.critic_opt.compute_gradients(self.closs, tf.trainable_variables(scope = 'critic'))
    #
    #     print('pi variables:', tf.trainable_variables(scope = 'pi'))
    #     print('actor_op:', len(self.actor_grads_and_vars_op))
    #     return self.sess.run([self.actor_grads_and_vars_op, self.critic_grads_and_vars_op],
    #                          {self.tfs: s, self.tfa: a, self.tfdc_r: r,
    #                           self.tfadv: adv})

if __name__ == '__main__':
    env = ENV
    ppo = PPO()
    all_ep_r = []

    for ep in range(EP_MAX):
        s = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        for t in range(EP_LEN):  # in one episode
            # env.render()
            a = ppo.choose_action(s)  # 根据一个正态分布，选择一个action
            a1 = np.random.rand(A_DIM)
            s_, r, done, _ = env.step(a, a1)
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append((r + 8) / 8)  # normalize reward, find to be useful
            s = s_
            ep_r += r

            # update ppo
            if (t + 1) % BATCH == 0 or t == EP_LEN - 1:
                v_s_ = ppo.get_v(s_)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)  # v(s) = r + gamma * v(s+1)
                discounted_r.reverse()

                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(
                    discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                ppo.update(bs, ba, br)
                # actor_grads_and_vars, critic_grads_and_vars = ppo.get_gradients(bs, ba, br)
                # print("actor_grads : ", actor_grads_and_vars[0][0].shape, actor_grads_and_vars[0][1].shape)
                # print("actor_vars : ", actor_grads_and_vars[1])
                # print("actor_vars_get:", ppo.a_variables.get_weights())
                # print("all : ",  actor_grads_and_vars)

        if ep == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
        print(
            'Ep: %i' % ep,
            "|Ep_r: %i" % ep_r,
            ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
        )

    # plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    # plt.xlabel('Episode')
    # plt.ylabel('Moving averaged episode reward')
    # plt.show()
