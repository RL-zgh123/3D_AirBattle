import numpy as np
class Offense(object):
    def __init__(self, action_bound, i_enemy = 0, factor=0.25):
        self.i_enemy = i_enemy
        self.action_bound = action_bound * factor # reserve optimization space for intelligent agent friend

    def get_action(self, s, info):
        n_f = info['N_friend']
        action_dim = info['action_dim']
        entity_dim = info['entity_dim']
        index = n_f + self.i_enemy  # reference player index
        enemy_pos = s[index * entity_dim:index * entity_dim + action_dim]

        distance_list = []
        for i in range(n_f):
            friend_pos = s[i * entity_dim:i * entity_dim + action_dim]
            distance = np.linalg.norm(enemy_pos - friend_pos)
            distance_list.append(distance)
        near_index = np.argmin(distance_list)
        near_friend_pos = s[near_index * entity_dim:near_index * entity_dim + action_dim]

        near_vector = near_friend_pos - enemy_pos # point to friend
        near_vector_norm = near_vector / np.linalg.norm(near_vector, ord=1)
        action = near_vector_norm * self.action_bound
        # print(near_friend_pos, '\n', enemy_pos, '\n', near_vector, action)
        return 100 * action


if __name__ == '__main__':
    info = {}
    info['N_friend'] = 1
    info['action_dim'] = 3
    info['entity_dim'] = 3
    state = np.array([1, 1, 1, 2, 2, 2], dtype=np.float32)
    action_bound = 100 * np.ones(3)
    offense = Offense(action_bound)
    action = offense.get_action(state, info)
    print(action)
