import numpy as np
import sys
import tensorflow as tf

sys.path.append('..')
from envs.myEnv import AirBattle
from algorithms.DDPG_option import Option, Actor
from algorithms.offense import Offense

class Model(object):
    def __init__(self):
        self.env = AirBattle()
        action_dim = self.env.n_actions
        state_dim = self.env.n_features
        option_dim = 2
        action_bound = self.env.action_bound

        sess = tf.Session()
        self.option_model = Option()
        self.actor_model = Actor()




        self.step_count = 0
        self.max_steps = 200



    # random action
    def _get_action(self, o_n):
        act0 = np.random.rand(self.n_actions)
        act1 = np.random.rand(self.n_actions)
        return act0, act1

    def load_data(self, relative_path, file_name):




    # one single rollout
    def rollout(self):
        print('Enter rollout.')
        steps = 0
        o_n = self.env.reset()
        print('Rollout reset sucessfully!')
        r_all = 0
        while True:
            self.step_count += 1
            steps += 1
            act0, act1 = self._get_action(o_n)
            o_n_next, r_n, d_n, _ = self.env.step(act0, act1)
            r_all += r_n
            o_n = o_n_next
            if steps == self.max_steps or d_n:
                break
        return steps, r_all

if __name__ == '__main__':
    iterations = 100
    model = Model()
    dic = {'win':0, 'equal':0, 'lose':0}
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
    print('total wins = {}, equal = {}, lose = {}'.format(dic['win'], dic['equal'], dic['lose']))




if __name__ == '__main__':