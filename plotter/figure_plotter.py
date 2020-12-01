import pickle
import os
import seaborn as sns # 0.9.0
import matplotlib.pyplot as plt
import numpy as np


class Fig_Plotter(object):
    def plot_single(self, relative_path, file_name):
        file = file_name + '.pkl'
        with open(os.path.join(relative_path, file), "rb") as f:
            data = pickle.load(f)

        for k in data.keys():
            fig = plt.figure()
            x = data[k]
            # x = self.smooth(x, sm=2)
            print(np.array(x).shape)
            length = np.array(x).shape[0]
            sns.set(style="darkgrid", font_scale=1.5)
            sns.tsplot(time=range(length), data=x, color="r", condition=k)
        plt.show()

    def plot_multi_demo(self):
        def getdata():
            basecond = [[18, 20, 19, 18, 13, 4, 1],
                        [20, 17, 12, 9, 3, 0, 0],
                        [20, 20, 20, 12, 5, 3, 0]]

            cond1 = [[18, 19, 18, 19, 20, 15, 14],
                     [19, 20, 18, 16, 20, 15, 9],
                     [19, 20, 20, 20, 17, 10, 0],
                     [20, 20, 20, 20, 7, 9, 1]]

            cond2 = [[20, 20, 20, 20, 19, 17, 4],
                     [20, 20, 20, 20, 20, 19, 7],
                     [19, 20, 20, 19, 19, 15, 2]]

            cond3 = [[20, 20, 20, 20, 19, 17, 12],
                     [18, 20, 19, 18, 13, 4, 1],
                     [20, 19, 18, 17, 13, 2, 0],
                     [19, 18, 20, 20, 15, 6, 0]]

            return basecond, cond1, cond2, cond3
        data = getdata()
        fig = plt.figure()
        xdata = np.array([0, 1, 2, 3, 4, 5, 6]) / 5
        linestyle = ['-', '--', ':', '-.']
        color = ['r', 'g', 'b', 'k']
        label = ['algo1', 'algo2', 'algo3', 'algo4']

        for i in range(4):
            sns.tsplot(time=xdata, data=data[i], color=color[i], linestyle=linestyle[i],
                       condition=label[i])
        plt.show()

    def plot_multi(self, relative_path, file_name, num_file, episodes, ymin, ymax):
        """
        画路径下data文件包含的对应的单条图线的置信区间图
        Args:
            relative_path: 相对文件路径
            file_name: 文件名
            num_file: 一共读取文件数目
            episodes: 需要读取的横轴长度（xmax）
            ymin: y轴起始值
            ymax: y轴最大值

        Returns:

        """
        all_data = {}
        for i in range(num_file):
            print(i)
            file = '{}_{}.pkl'.format(file_name, i)
            with open(os.path.join(relative_path, file), "rb") as f:
                data = pickle.load(f)

            for key in data.keys():
                if key not in all_data.keys():
                    all_data[key] = np.array(data[key])[np.newaxis, :episodes]
                else:
                    all_data[key] = np.concatenate([all_data[key], np.array(data[key])[np.newaxis, :episodes]], axis=0)
            print(i)

        fig = plt.figure()
        xdata = np.array(range(episodes))
        linestyle = ['-', ':']
        color = ['r', 'g']
        label = ['mean reward', 'mean shaping reward']

        for i, key in enumerate(all_data.keys()):
            ax = sns.tsplot(time=xdata, data=all_data[key], color=color[i],
                       linestyle=linestyle[i],
                       condition=label[i])
            ax.legend(fontsize=15)
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Reward')

        plt.axis([0, episodes, ymin, ymax])
        plt.show()

    def plot_compare(self, relative_path, file_name_list, key_word, num_file, episodes):
        all_data = {}
        for file_name in file_name_list:
            print(file_name)
            for i in range(num_file):
                print(i)
                file = '{}_{}.pkl'.format(file_name, i)
                with open(os.path.join(relative_path, file), "rb") as f:
                    data = pickle.load(f)

                key = key_word + file_name
                add_data = np.array(data[key_word])
                if key not in all_data.keys():
                    all_data[key] = add_data[np.newaxis, :episodes]
                else:
                    all_data[key] = np.concatenate([all_data[key], add_data[np.newaxis, :episodes]], axis=0)

        fig = plt.figure()
        xdata = np.array(range(episodes))
        linestyle = ['-', '-']
        color = ['r', 'g']
        label = file_name_list

        for i, key in enumerate(all_data.keys()):
            ax = sns.tsplot(time=xdata, data=all_data[key], color=color[i],
                       linestyle=linestyle[i],
                       condition=label[i])
            ax.legend(fontsize=15)
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Reward')

        plt.axhline(y=15.0, color='b', linestyle='--')
        plt.axhline(y=10.0, color='b', linestyle='--')
        plt.axis([0, episodes, -5, 25])
        plt.show()

    def smooth(self, data, sm=1):
        if sm > 1:
            smooth_data = []
            for d in data:
                y = np.ones(sm)*1.0/sm
                d = np.convolve(y, d, "same")
                smooth_data.append(d)
        return smooth_data

if __name__ == '__main__':
    plotter = Fig_Plotter()
    # plotter.plot_multi_demo()
    # plotter.plot_single('../results', 'option_data_0')
    plotter.plot_multi('../results/nfsp', 'nfsp_data', 4, 17, 0, 0.8)
    # plotter.plot_compare('../results/option', ['option_data', 'option_origin'], 'mean episode reward', 4, 3500)
