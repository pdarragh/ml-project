from ..math_and_types import *

from abc import ABC, abstractmethod


class ParameterizedLearner(ABC):
    @abstractmethod
    def train(self, data_pair: DataPair):
        pass

    @abstractmethod
    def test(self, data_pair: DataPair) -> TestResult:
        pass

    @staticmethod
    def _test_result_from_guess_and_actual(guess: int, actual: int) -> TestResult:
        return TestResult(guess, actual)
