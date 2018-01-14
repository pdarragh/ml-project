from .forest import Forest
from ..parameterized_learner import ParameterizedLearner
from ...math_and_types import *


class SimpleForestLearner(ParameterizedLearner):
    def __init__(self, forest: Forest):
        self.forest = forest

    def train(self, data_pair: DataPair):
        pass

    def test(self, data_pair: DataPair) -> TestResult:
        count = 0
        for tree in self.forest.trees:
            count += tree.predict(data_pair)
        if count > 0:
            p = 1
        elif count < 0:
            p = -1
        else:
            toss = randint(0, 1)
            if toss == 1:
                p = 1
            else:
                p = -1
        return self._test_result_from_guess_and_actual(p, data_pair.label)
