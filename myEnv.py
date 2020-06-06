import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

class entity(object):
    def __init__(self, movable=True, is_friend=True):
        self.action_dim = 6
        self.movable = movable
        self.is_friend = is_friend
        self.radius = 0.5 if self.movable else 1
        self.vel = np.zeros(self.action_dim, dtype = np.float32)
        self.acc = np.zeros(self.action_dim, dtype = np.float32)
        if self.movable:
            self.pos = np.array([2, 2, 2, 0, 0, 0], dtype = np.float32) if self.is_friend else np.array([-2, -2, -2, 0, 0, 0], dtype = np.float32)
        else:
            self.pos = np.array([0, 0, 0, 0, 0, 0], dtype = np.float32)
        self.max_pos = np.array([4, 4, 4, 2*math.pi, 2*math.pi, 2*math.pi], dtype = np.float32)
        self.min_pos = -np.array([4, 4, 4, 2*math.pi, 2*math.pi, 2*math.pi], dtype = np.float32)
        self.max_vel = np.array([0.5, 0.5, 0.5, math.pi/6, math.pi/6, math.pi/6], dtype = np.float32)
        self.min_vel = -np.array([0.5, 0.5, 0.5, math.pi/6, math.pi/6, math.pi/6], dtype = np.float32)
        self.max_acc = 2 * np.array([0.5, 0.5, 0.5, math.pi / 6, math.pi / 6, math.pi / 6], dtype = np.float32)
        self.min_acc = - 2 * np.array([0.5, 0.5, 0.5, math.pi / 6, math.pi / 6, math.pi / 6], dtype = np.float32)

    def reset_state(self):
        self.vel = np.zeros(self.action_dim, dtype = np.float32)
        self.acc = np.zeros(self.action_dim, dtype = np.float32)
        if self.movable:
            self.pos = np.concatenate((8*np.random.rand(3)-4, 6*np.random.rand(3)-3))
        else:
            self.pos = np.array([0, 0, 0, 0, 0, 0], dtype = np.float32)

    # limit vel range
    def vel_range_limit(self):
        for i in range(self.action_dim):
            self.vel[i] = self.max_vel[i] if self.vel[i] > self.max_vel[i] \
                else self.min_vel[i] if self.vel[i] < self.min_vel[i] else self.vel[i]

    # limit pos range
    def pos_range_limit(self):
        for i in range(self.action_dim):
            self.pos[i] = self.max_pos[i] if self.pos[i] > self.max_pos[i] \
                else self.min_pos[i] if self.pos[i] < self.min_pos[i] else self.pos[i]

    # limit acc range
    def acc_range_limit(self):
        for i in range(self.action_dim):
            self.acc[i] = self.max_acc[i] if self.acc[i] > self.max_acc[i] \
                else self.min_acc[i] if self.acc[i] < self.min_acc[i] else self.acc[i]


# only support two agents(friend and enemy)
class AirBattle(object):
    def __init__(self):
        self.dt = 0.1
        self.friend = [entity(is_friend = True)]
        self.enemy = [entity(is_friend = False)]
        self.hinder = [entity(movable = False)]
        self.agents = self.friend + self.enemy
        self.entities = self.agents + self.hinder
        self.n_actions = 6
        self.n_features = self._get_state().shape[0]

    # detect collision
    # no requirement for entity0 and entity1
    def _collision_detect(self, entity0, entity1):
        delta_pos = (entity0.pos - entity1.pos)[:3]
        distance = np.linalg.norm(delta_pos) # modular value
        return True if distance < (entity0.radius + entity1.radius) else False

    # detect kill
    # both agent0 and agent1 must be agent
    # return kill flag, win agent, lose agent
    def _kill_detect(self, agent0, agent1):
        # orientation of agent
        direct0 = agent0.pos[-3:]
        direct1 = agent1.pos[-3:]
        mo_direct0 = np.linalg.norm(direct0)
        mo_direct1 = np.linalg.norm(direct1)

        # relative orientation of two agents
        dist01 = agent1.pos[:3] - agent0.pos[:3]
        dist10 = -dist01
        mo_dist01 = np.linalg.norm(dist01)
        mo_dist10 = np.linalg.norm(dist10)

        # intersection space angle
        cos_angle01 = np.dot(dist01, direct0.T) / (mo_dist01 * mo_direct0)
        cos_angle10 = np.dot(dist10, direct1.T) / (mo_dist10 * mo_direct1)

        if cos_angle01 < math.pi/4 and cos_angle10 >= math.pi/4:
            return True, agent0, agent1
        elif cos_angle01 >= math.pi/4 and cos_angle10 < math.pi/4:
            return True, agent1, agent0
        # both in fire range
        # elif cos_angle01 < math.pi/4 and cos_angle10 < math.pi/4:
        #     return True, agent1, agent0
        else:
            return False, None, None

    # rebound if not kill
    # agent0 must be agent, no requirement for agent1
    def _rebound(self, agent0, agent1):
        # relative orientation of two agents
        dist01 = agent1.pos[:3] - agent0.pos[:3]
        dist10 = -dist01
        mo_dist01 = np.linalg.norm(dist01)
        mo_dist10 = np.linalg.norm(dist10)

        overlay = agent0.radius + agent1.radius - mo_dist01
        if agent0.movable and agent1.movable:
            return - dist01 / mo_dist01 * overlay, - dist10 / mo_dist10 * overlay
        # agent0.movable and not agent1.movable
        else:
            return - dist01 / mo_dist01 * overlay, 0

    # rotate the orientation point
    def _rotate(self, pos):
        x, y, z, a, b, c = pos
        rotatex = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        rotatey = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
        rotatez = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
        R = np.dot(np.dot(rotatex, rotatey), rotatez)
        x1 = x * R[0][0] + y * R[0][1] + z * R[0][2]
        y1 = x * R[1][0] + y * R[1][1] + z * R[1][2]
        z1 = x * R[2][0] + y * R[2][1] + z * R[2][2]
        return np.array([x1, y1, z1])

    def _update_state(self, act0, act1):
        # pos update(regard as uniform motion)
        delta_friend_pos = np.hstack((self.friend[0].vel[:3] * self.dt, self.friend[0].pos[-3:]))
        delta_enemy_pos = np.hstack((self.enemy[0].vel[:3] * self.dt, self.enemy[0].pos[-3:]))
        self.friend[0].pos[:3] += self._rotate(delta_friend_pos)
        self.enemy[0].pos[:3] += self._rotate(delta_enemy_pos)
        self.friend[0].pos[-3:] += self.friend[0].vel[-3:] * self.dt
        self.enemy[0].pos[-3:] += self.enemy[0].vel[-3:] * self.dt
        self.friend[0].pos_range_limit()
        self.enemy[0].pos_range_limit()

        # vel update
        self.friend[0].acc = act0
        self.friend[0].acc_range_limit()
        self.friend[0].vel += self.friend[0].acc * self.dt
        self.friend[0].vel_range_limit()

        self.enemy[0].acc = act1
        self.enemy[0].acc_range_limit()
        self.enemy[0].vel += self.enemy[0].acc * self.dt
        self.enemy[0].vel_range_limit()

    def _get_state(self):
        return np.hstack((self.friend[0].pos, self.friend[0].vel, self.friend[0].radius,
                          self.enemy[0].pos, self.enemy[0].vel, self.enemy[0].radius,
                          self.hinder[0].pos, self.hinder[0].vel, self.hinder[0].radius))

    def reset(self):
        while True:
            # keeping reset until no collision
            for entity in self.entities:
                entity.reset_state()
            flag = False

            for entity in self.entities:
                for other in self.entities:
                    if entity is not other:
                        flag = self._collision_detect(entity, other)
                        # print(entity.pos, other.pos, flag)
                    if flag:
                        break
                if flag:
                    break
            if not flag:
                break
        return self._get_state()

    # return o_n_next, a_n, r_n, i_n
    def step(self, act0, act1):
        self._update_state(act0, act1)

        done = False
        for agent in self.agents:
            for entity in self.entities:
                if agent is entity:
                    continue
                elif not self._collision_detect(agent, entity):
                    continue

                if entity.movable:
                    done, win, lose = self._kill_detect(agent, entity)
                    if done:
                        reward = 10 if win == self.friend[0] else -10
                        return self._get_state(), reward, done, None
                else:
                    rebound0, rebound1 = self._rebound(agent, entity)
                    agent.pos[:3] += rebound0
                    entity.pos[:3] += rebound1
        return  self._get_state(), 0, done, None
