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
            length = np.array(x).shape[1]
            sns.set(style="darkgrid", font_scale=1.5)
            sns.tsplot(time=range(length), data=x, color="r", condition="file_name")
        plt.show()

    def plot_multi(self):
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
    plotter.plot_multi()