from .dynamic_perceptron import *


class MarginPerceptron(DynamicPerceptron):
    def __init__(self, n_features: int, learning_rate: float, margin: float):
        super().__init__(n_features, learning_rate=learning_rate)
        self.u = margin
