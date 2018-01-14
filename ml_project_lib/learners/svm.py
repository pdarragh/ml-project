from .parameterized_learner import ParameterizedLearner
from ..math_and_types import *


class SVM(ParameterizedLearner):
    def __init__(self, n_features: int, learning_rate: float, tradeoff: float):
        self.g = learning_rate
        self.c = tradeoff
        self.t = 0
        shape = (1, n_features)
        self.w = ZerosVector(shape)

    def train(self, data_pair: DataPair):
        self.t += 1
        gt = self.g / (1 + self.t)
        y = data_pair.label
        x = data_pair.vector
        if (y * (self.w * x.transpose())) <= 1:
            self.w = ((1 - gt) * self.w) + ((gt * self.c * y) * x)
        else:
            self.w *= (1 - gt)

    def test(self, data_pair: DataPair) -> TestResult:
        y = data_pair.label
        x = data_pair.vector
        p = -1 if (self.w * x.transpose()) < 0 else 1
        return self._test_result_from_guess_and_actual(p, y)
