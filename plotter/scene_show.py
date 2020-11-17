import sys

import numpy as np
import tensorflow as tf

sys.path.append('..')
from envs.myEnv import AirBattle
from algorithms.DDPG_option import Option, Actor
from algorithms.offense import Offense


class Model(object):
    def __init__(self, file_path):
        self.env = AirBattle()
        action_dim = self.env.n_actions
        state_dim = self.env.n_features
        option_dim = 2
        action_bound = self.env.action_bound
        self.offense = Offense(action_bound, 0)

        sess = tf.Session()
        self.option = Option(sess, option_dim)
        self.actor = Actor(sess, option_dim, action_dim, action_bound, self.option.s,
                           self.option.s_, self.option.o, self.option.o_)
        # sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, file_path)

        self.step_count = 0
        self.max_steps = 200

    def _get_action(self, o_n, info):
        # a0 = np.random.rand(self.n_actions)
        # a1 = np.random.rand(self.n_actions)

        op = self.option.get_option(o_n)
        a0 = self.actor.choose_action(o_n, op[np.newaxis, :])
        a1 = self.offense.get_action(o_n, info)
        return a0, a1

    # one single rollout
    def rollout(self):
        print('Enter rollout.')
        steps = 0
        o_n, info = self.env.reset()
        print('Rollout reset sucessfully!')
        r_all = 0
        while True:
            self.step_count += 1
            steps += 1
            act0, act1 = self._get_action(o_n, info)
            o_n_next, r_n, d_n, info = self.env.step(act0, act1)
            r_all += r_n
            o_n = o_n_next
            if steps == self.max_steps or d_n:
                break
        return steps, r_all


if __name__ == '__main__':
    iterations = 100
    relative_path = '../results'
    file_name = 0
    file_path = '{}/{}.ckpt'.format(relative_path, file_name)
    model = Model(file_path)
    dic = {'win': 0, 'equal': 0, 'lose': 0}

    for i in range(iterations):
        steps, r_all = model.rollout()
        print('Iteration {}, steps: {}, r_all: {}'.format(i, steps, r_all))
        if r_all > 0:
            model.env.render(steps)
            dic['win'] += 1
        elif r_all == 0:
            dic['equal'] += 1
        else:
            dic['lose'] += 1
    print('total wins = {}, equal = {}, lose = {}'.format(dic['win'], dic['equal'],
                                                          dic['lose']))
