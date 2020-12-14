import sys

import numpy as np
import tensorflow as tf

sys.path.append('..')
from envs.myEnv2 import AirBattle
from algorithms.DDPG_option import Option, Actor
from algorithms.Offense import Offense


class Model(object):
    def __init__(self, file_path):
        self.env = AirBattle()
        action_dim = self.env.n_actions
        state_dim = self.env.n_features
        option_dim = 2
        action_bound = self.env.action_bound
        self.offense = Offense(action_bound, 0)

        sess = tf.Session()
        self.option = Option('agent0', sess, option_dim, state_dim)
        self.actor = Actor('agent0', sess, option_dim, action_dim, state_dim, action_bound, self.option.s,
                           self.option.s_, self.option.o, self.option.o_)
        # sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, file_path)
        print('Successfully load session data!')

        self.step_count = 0
        self.max_steps = 200

    def _get_action(self, o_n, info):
        # a0 = np.random.rand(self.n_actions)
        # a1 = np.random.rand(self.n_actions)

        op = self.option.get_option(o_n)
        op_value = self.option.get_option_value(o_n)
        # if op_value[0] - op_value[1] < 1:
        #     print(op_value)
        a0 = self.actor.choose_action(o_n, op[np.newaxis, :])
        a1 = self.offense.get_action(o_n, info)
        return a0, a1, op, op_value

    # one single rollout
    def rollout(self):
        print('Enter rollout.')
        steps = 0
        o_n, info = self.env.reset()
        print('Rollout reset sucessfully!')
        r_all = 0
        option_old = 0
        option_count = 0
        option_list = []
        while True:
            self.step_count += 1
            steps += 1
            act0, act1, option, option_value = self._get_action(o_n, info)
            option_list.append(option[0])
            if option[0] != option_old:
                option_count += 1
                option_old = option[0]
            o_n_next, r_n, d_n, info = self.env.step(act0, act1, option, option_value)
            r_all += r_n
            o_n = o_n_next
            if steps == self.max_steps or d_n:
                break
        return steps, r_all, option_count, option_list


if __name__ == '__main__':
    iterations = 100
    relative_path = '../results'
    file_name = 'option_0'
    file_path = '{}/{}.ckpt'.format(relative_path, file_name)
    model = Model(file_path)
    dic = {'win': 0, 'equal': 0, 'lose': 0}
    option_total = []

    for i in range(iterations):
        steps, r_all, option_count, option_list = model.rollout()
        print('Iteration {}, steps: {}, r_all: {}'.format(i, steps, r_all))
        if option_count > 3:
            print('steps:', steps)
            print('option_list:', option_list)
            if steps != model.max_steps:
                option_total.append(option_list)
        if r_all > 0:
            # model.env.render(steps)
            dic['win'] += 1
        elif r_all == 0:
            dic['equal'] += 1
        else:
            dic['lose'] += 1
    print(option_total, len(option_total))
    print('total wins = {}, equal = {}, lose = {}'.format(dic['win'], dic['equal'],
                                                          dic['lose']))
