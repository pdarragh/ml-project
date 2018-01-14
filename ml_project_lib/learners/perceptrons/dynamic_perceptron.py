from .simple_perceptron import *


class DynamicPerceptron(SimplePerceptron):
    def __init__(self, n_features: int, learning_rate: float):
        super().__init__(n_features, learning_rate=learning_rate)

    def train(self, data_pair: DataPair):
        super().train(data_pair)
        self.t += 1

    @property
    def n(self):
        return self._n / (self.t + 1)
