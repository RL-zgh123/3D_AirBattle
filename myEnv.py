import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class entity(object):
    def __init__(self, movable=True, is_friend=True):
        self.action_dim = 6
        self.movable = movable
        self.is_friend = is_friend
        self.radius = 1.75 if self.movable else 3
        self.vel = np.zeros(self.action_dim, dtype=np.float32)
        self.acc = np.zeros(self.action_dim, dtype=np.float32)
        if self.movable:
            self.pos = np.array([2, 2, 2, 0, 0, 0],
                                dtype=np.float32) if self.is_friend else np.array(
                [-2, -2, -2, 0, 0, 0], dtype=np.float32)
        else:
            self.pos = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.max_pos = np.array([4, 4, 4, 2 * math.pi, 2 * math.pi, 2 * math.pi],
                                dtype=np.float32)
        self.min_pos = -self.max_pos
        self.max_vel = np.array(
            [3.0, 3.0, 3.0, math.pi / 3, math.pi / 3, math.pi / 3], dtype=np.float32)
        self.min_vel = -self.max_vel
        self.max_acc = 2 * np.array(
            [2.0, 2.0, 2.0, math.pi / 4, math.pi / 4, math.pi / 4], dtype=np.float32)
        self.min_acc = - self.max_acc

    def reset_state(self):
        self.vel = np.zeros(self.action_dim, dtype=np.float32)
        self.acc = np.zeros(self.action_dim, dtype=np.float32)
        if self.movable:
            self.pos = np.concatenate(
                (8 * np.random.rand(3) - 4, 6 * np.random.rand(3) - 3))
        else:
            self.pos = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

    # limit vel range
    def vel_range_limit(self):
        for i in range(self.action_dim):
            self.vel[i] = self.max_vel[i] if self.vel[i] > self.max_vel[i] \
                else self.min_vel[i] if self.vel[i] < self.min_vel[i] else self.vel[
                i]

    # limit pos range
    def pos_range_limit(self):
        for i in range(self.action_dim):
            if i < self.action_dim / 2:
                self.pos[i] = self.max_pos[i] if self.pos[i] > self.max_pos[i] \
                    else self.min_pos[i] if self.pos[i] < self.min_pos[i] else \
                    self.pos[
                        i]
            else:
                self.pos[i] = self.pos[i] - self.max_pos[i] if self.pos[i] > \
                                                               self.max_pos[i] else \
                    self.pos[i] - self.min_pos[i] if self.pos[i] < self.min_pos[
                        i] else \
                        self.pos[i]

    # limit acc range
    def acc_range_limit(self):
        for i in range(self.action_dim):
            self.acc[i] = self.max_acc[i] if self.acc[i] > self.max_acc[i] \
                else self.min_acc[i] if self.acc[i] < self.min_acc[i] else self.acc[
                i]


# only support two agents(friend and enemy)
class AirBattle(object):
    def __init__(self):
        self.dt = 0.1
        self.friend = [entity(is_friend=True)]
        self.enemy = [entity(is_friend=False)]
        self.hinder = [entity(movable=False)]
        self.agents = self.friend + self.enemy
        self.entities = self.agents + self.hinder
        self.n_actions = 6
        self.n_features = self._get_state().shape[0]
        self.action_bound = self.friend[0].max_acc
        self._cursor = 0
        self._store = np.empty([10000] + [len(self.entities) * self.n_actions])
        self._count = 0

    # detect collision
    # no requirement for entity0 and entity1
    def _collision_detect(self, entity0, entity1):
        delta_pos = (entity0.pos - entity1.pos)[:3]
        distance = np.linalg.norm(delta_pos)  # modular value
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

        threshold = math.sqrt(2) / 2

        # flog, win, lose
        if cos_angle01 > threshold and cos_angle10 <= threshold:
            return True, agent0, agent1
        elif cos_angle01 <= threshold and cos_angle10 > threshold:
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
        mo_dist01 = max(np.linalg.norm(dist01), 0.1)
        mo_dist10 = max(np.linalg.norm(dist10), 0.1)

        overlay = agent0.radius + agent1.radius - mo_dist01
        if agent1.movable:
            return - dist01 / mo_dist01 * overlay, - dist10 / mo_dist10 * overlay

        # agent0.movable and not agent1.movable
        else:
            return - dist01 / mo_dist01 * overlay, 0

    # rotate the orientation point
    def _rotate(self, pos):
        x, y, z, a, b, c = pos
        rotatex = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)],
                            [0, math.sin(a), math.cos(a)]])
        rotatey = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0],
                            [-math.sin(b), 0, math.cos(b)]])
        rotatez = np.array(
            [[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0],
             [0, 0, 1]])
        R = np.dot(np.dot(rotatex, rotatey), rotatez)
        x1 = x * R[0][0] + y * R[0][1] + z * R[0][2]
        y1 = x * R[1][0] + y * R[1][1] + z * R[1][2]
        z1 = x * R[2][0] + y * R[2][1] + z * R[2][2]
        return np.array([x1, y1, z1])

    def _update_state(self, act0, act1):
        # pos update(regard as uniform motion)
        delta_friend_pos = np.hstack(
            (self.friend[0].vel[:3] * self.dt, self.friend[0].pos[-3:]))
        delta_enemy_pos = np.hstack(
            (self.enemy[0].vel[:3] * self.dt, self.enemy[0].pos[-3:]))

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
        return np.hstack(
            (self.friend[0].pos, self.friend[0].vel, self.friend[0].radius,
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
        self._store_state()
        done = False

        # rebound judgement
        for i in range(0, len(self.agents)):
            for j in range(0, len(self.entities)):
                if self.agents[i] == self.entities[j] or not self._collision_detect(
                        self.agents[i], self.entities[j]):
                    continue

                elif self.entities[j].movable:
                    done, win, lose = self._kill_detect(self.agents[i],
                                                        self.entities[j])
                    if done:
                        reward = 10 if win == self.friend[0] else -10
                        return self._get_state(), reward, done, None
                    else:
                        rebound0, rebound1 = self._rebound(self.agents[i],
                                                           self.entities[j])
                        self.agents[i].pos[:3] += rebound0
                        self.entities[j].pos[:3] += rebound1

                else:
                    rebound0, _ = self._rebound(self.agents[i], self.entities[j])
                    self.agents[i].pos[:3] += rebound0

        return self._get_state(), 0, done, None

    def _store_state(self):
        state = np.array([])
        for entity in self.entities:
            state = np.concatenate((state, entity.pos), axis=0)
        self._store[self._cursor % self._store.shape[0]] = state
        self._cursor += 1

    def _update(self, num, pf, pe, pc):
        ax = plt.axes(projection='3d')
        # Setting the axes properties
        ax.set_xlim3d([-4.0, 4.0])
        ax.set_xlabel('X')
        ax.set_ylim3d([-4.0, 4.0])
        ax.set_ylabel('Y')
        ax.set_zlim3d([-4.0, 4.0])
        ax.set_zlabel('Z')
        ax.set_title('3D Experiment')

        tf = self._store[num][: self.n_actions]
        te = self._store[num][self.n_actions: 2 * self.n_actions]
        tc = self._store[num][-self.n_actions:]

        pf = self._rotate([pf[0], pf[1], pf[2], tf[3], tf[4], tf[5]])
        pe = self._rotate([pe[0], pe[1], pe[2], te[3], te[4], te[5]])
        pc = self._rotate([pc[0], pc[1], pc[2], tc[3], tc[4], tc[5]])
        ax.plot_surface(pf[0] + tf[0], pf[1] + tf[1], pf[2] + tf[2], color='r')
        ax.plot_surface(pe[0] + te[0], pe[1] + te[1], pe[2] + te[2],
                        color='lightyellow')
        ax.plot_surface(pc[0] + tc[0], pc[1] + tc[1], pc[2] + tc[2], color='navy')
        return ax

    def _generate_ball(self, radius, completed=True):
        if not completed:
            u = np.linspace(0, 2 * np.pi, 25)
            v = np.linspace(-np.pi, np.pi / 4, 25)
        else:
            u = np.linspace(0, 2 * np.pi, 25)
            v = np.linspace(-np.pi, np.pi, 25)
        xc = radius * np.outer(np.cos(u), np.sin(v))
        yc = radius * np.outer(np.sin(u), np.sin(v))
        zc = radius * np.outer(np.ones(np.size(u)), np.cos(v))
        return [xc, yc, zc]

    def render(self, gap):
        pf = self._generate_ball(self.friend[0].radius, False)
        pe = self._generate_ball(self.enemy[0].radius, False)
        pc = self._generate_ball(self.hinder[0].radius, True)
        fig = plt.figure()

        # num = gap
        num = 100
        self._store = self._store[
                      self._cursor % self._store.shape[0] - num:self._cursor %
                                                                self._store.shape[0]]
        ani = FuncAnimation(fig, self._update, num, fargs=(pf, pe, pc),
                            interval=1, blit=False)
        # plt.savefig('demo{}_{}.gif'.format(self._count * gap, (self._count + 1) * gap))
        plt.show()
        plt.close()
        self._cursor = 0
        self._store = np.empty([10000] + [len(self.entities)] * self.n_actions)
        self._count += 1
