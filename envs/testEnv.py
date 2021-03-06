
import numpy as np
from envs.myEnv2 import AirBattle

class Model(object):
    def __init__(self):
        self.env = AirBattle()
        self.n_actions = self.env.n_actions
        self.step_count = 0
        self.max_steps = 200

    # random action
    def _get_action(self, o_n):
        act0 = np.random.rand(self.n_actions)
        act1 = np.random.rand(self.n_actions)
        return act0, act1

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
            option, option_value = 0, [0, 0]
            o_n_next, r_n, d_n, _ = self.env.step(act0, act1, option, option_value)
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