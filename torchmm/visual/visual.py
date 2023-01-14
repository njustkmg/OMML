from torchmm.visual.backbone import TSNE

MethodMap = {
    'tsne': TSNE,
}

class Visual():
    def __init__(self, method, save_path):
        super(Visual, self).__init__()
        self.method = MethodMap[method.lower()](save_path)

    def plot(self, X, y, str, epoch, immediately_show=False):
        self.method.print(X, y, str, epoch, immediately_show=immediately_show)
