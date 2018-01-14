from .parameterized_learner import ParameterizedLearner
from ..math_and_types import *


class LogisticRegression(ParameterizedLearner):
    def __init__(self, n_features: int, learning_rate: float, tradeoff: float):
        self.g = learning_rate
        self.c = tradeoff
        self.t = 0
        shape = (1, n_features)
        self.w = ZerosVector(shape)

    @staticmethod
    def _sigmoid(wx: float) -> float:
        if wx > MAX_EXP_ARG:
            # The wx component is too great, which collapses the fraction to 1.
            return 1
        if wx < -MAX_EXP_ARG:
            # The wx component is too low, which collapses the fraction to 0.
            return 0
        return 1 / (1 + exp(-wx))

    def _del_j(self, y: Label, x: SparseMatrix, wx: float) -> SparseMatrix:
        n1 = -1 * (y * x)
        try:
            d1 = 1 + exp(y * wx)
            f1 = n1 / d1
        except OverflowError:
            f1 = 0
        n2 = 2 * self.w
        d2 = power(self._sigmoid(wx), 2)
        f2 = n2 / d2
        return f1 + f2

    def train(self, data_pair: DataPair):
        self.t += 1
        gt = self.g / (1 + self.t)
        y = data_pair.label
        x = data_pair.vector
        wx = self.w * x.transpose()
        if wx > MAX_EXP_ARG or wx < MIN_EXP_ARG:
            # Skip this example.
            return
        self.w = self.w - gt * self._del_j(y, x, wx)

    def test(self, data_pair: DataPair) -> TestResult:
        y = data_pair.label
        x = data_pair.vector
        wx = self.w * x.transpose()
        sig = self._sigmoid(wx)
        if sig > 0.5:
            t = 1
        else:
            t = -1
        return self._test_result_from_guess_and_actual(t, y)
