from sklearn import manifold
import matplotlib.pyplot as plot
import os

color = [['deepskyblue', 'violet', 'orangered'],
         ['blue', 'purple', 'red']]

class TSNE():
    def __init__(self, save_path):
        super(TSNE, self).__init__()
        self.save_path = save_path + "/tsne/"

    def print(self, X, y, string, epoch, immediately_show=False):
        y = list(map(int, y))
        plot.figure(figsize=(15, 15))
        for k in range(len(X)):
            x = X[k]
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
            X_tsne = tsne.fit_transform(x)

            x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
            for i in range(X_norm.shape[0]):
                plot.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=color[k][y[i]],
                          fontdict={'weight': 'bold', 'size': 9})

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        path = self.save_path + string + "_" + str(epoch)
        plot.savefig(path)
        if immediately_show:
            plot.show()