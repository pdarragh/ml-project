from .margin_perceptron import *


class AggressiveMarginPerceptron(MarginPerceptron):
    def __init__(self, n_features: int, margin: float):
        super().__init__(n_features, learning_rate=0, margin=margin)

    def train(self, data_pair: DataPair):
        y = data_pair.label
        x = data_pair.vector
        wx = self.w * x.transpose()
        ywx = y * wx
        if ywx <= self.u:
            t = self.u - ywx
            b = (x * x.transpose()).toarray().item() + 1
            n = t / b
            nyx = (n * y) * x
            self.w = self.w + nyx
