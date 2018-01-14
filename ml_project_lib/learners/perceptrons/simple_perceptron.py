from .base_perceptron import *


class SimplePerceptron(BasePerceptron):
    def __init__(self, n_features: int, learning_rate: float):
        super().__init__(n_features)
        self._n = learning_rate

    def train(self, data_pair: DataPair):
        y = data_pair.label
        x = data_pair.vector
        if y * ((self.w * x.transpose()) + self.b) < self.u:
            # Update according to:
            #   w <- w + nyx
            #   b <- b + ny
            ny = self.n * y
            nyx = ny * x
            self.w = self.w + nyx
            self.b += ny

    def test(self, data_pair: DataPair) -> TestResult:
        y = data_pair.label
        x = data_pair.vector
        p = -1 if (self.w * x.transpose()) + self.b < 0 else 1
        return self._test_result_from_guess_and_actual(p, y)
