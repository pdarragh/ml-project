from ..parameterized_learner import ParameterizedLearner
from ...math_and_types import *

from abc import abstractmethod


UNI_LOW = -0.1
UNI_HIGH = 0.1


class BasePerceptron(ParameterizedLearner):
    def __init__(self, n_features: int):
        shape = (1, n_features)
        self.w = uniform(UNI_LOW, UNI_HIGH, size=shape)
        self.u = 0
        self._n = 0
        self.t = 0
        self.b = uniform(UNI_LOW, UNI_HIGH)

    @staticmethod
    def _small_random_value(low=UNI_LOW, high=UNI_HIGH) -> float:
        return uniform(low, high)

    @property
    def n(self):
        return self._n

    @abstractmethod
    def train(self, data_pair: DataPair):
        pass

    @abstractmethod
    def test(self, data_pair: DataPair) -> TestResult:
        pass
