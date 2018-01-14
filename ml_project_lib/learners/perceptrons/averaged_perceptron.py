from .simple_perceptron import *


class AveragedPerceptron(SimplePerceptron):
    def __init__(self, n_features: int, learning_rate: float):
        super().__init__(n_features, learning_rate=learning_rate)
        shape = (1, n_features)
        self.a = ZerosVector(shape)
        self.ba = 0

    def train(self, data_pair: DataPair):
        super().train(data_pair)
        self.a = self.a + self.w
        self.ba += self.b

    def test(self, data_pair: DataPair) -> TestResult:
        y = data_pair.label
        x = data_pair.vector
        t = -1 if (self.a * x.transpose()) + self.ba < 0 else 1
        return self._test_result_from_guess_and_actual(t, y)
