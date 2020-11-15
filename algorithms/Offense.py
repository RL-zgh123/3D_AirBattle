class Offense(object):
    def __init__(self, action_bound, i_enemy = 0):
        self.i_enemy = i_enemy
        self.action_bound = action_bound * 0.5 # reserve optimization space for intelligent agent friend

    def get_action(self, s, info):
        n_f = info['N_friend']
        n_e = info['N_enemy']
        action_dim = info['action_dim']
        entity_dim = info['entity_dim']
        index = n_f + self.i_enemy  # reference player index
        enemy_pos = s[index * entity_dim:index * entity_dim + action_dim]

        distance_list = []
        for i in range(n_f):
            friend_pos = s[i * entity_dim:i * entity_dim + action_dim]
            distance = np.linalog.norm(enemy_pos - friend_pos)
            distance_list.append(distnace)
        near_index = np.argmin(distance_list)
        near_friend_pos = s[near_index * entity_dim:near_index * entity_dim + action_dim]

        near_vector = near_friend_pos - enemy_pos # point to friend
        near_vector_norm = np.linalog.norm(near_vector, ord=1)





if __name__ == '__main__':
    offense = Offense()
