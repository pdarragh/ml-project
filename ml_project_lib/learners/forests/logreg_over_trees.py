from .forest import Forest
from ..parameterized_learner import ParameterizedLearner
from ..logreg import LogisticRegression
from ...math_and_types import *


class LogregOverTrees(ParameterizedLearner):
    def __init__(self, forest: Forest, learning_rate: float, tradeoff: float):
        self.forest = forest
        self.logreg = LogisticRegression(len(self.forest.trees), learning_rate, tradeoff)

    def _feature_transformation(self, data_pair: DataPair) -> DataPair:
        length = len(self.forest.trees)
        shape = (1, length)
        features = ZerosVector(shape)
        for i in range(length):
            features[0, i] = self.forest.trees[i].predict(data_pair)
        new_pair = DataPair(data_pair.label, SparseRowMatrix(features))
        return new_pair

    def train(self, data_pair: DataPair):
        new_pair = self._feature_transformation(data_pair)
        self.logreg.train(new_pair)

    def test(self, data_pair: DataPair) -> TestResult:
        new_pair = self._feature_transformation(data_pair)
        return self.logreg.test(new_pair)
