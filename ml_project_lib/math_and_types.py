from enum import Enum, auto
from numpy import argmax, exp, floor, log, log2, ndarray, ones, power, zeros
from numpy.random import uniform
from random import randint, shuffle
from scipy.sparse import csr_matrix, spmatrix
from typing import Any, Dict, Iterator, List, NamedTuple
from sys import float_info


__all__ = [
    'argmax', 'exp', 'floor', 'log', 'log2', 'power', 'MAX_EXP_ARG', 'MIN_EXP_ARG',
    'randint', 'shuffle', 'uniform',
    'Dict', 'List',
    'NumpyArray', 'ZerosVector', 'OnesVector', 'SparseMatrix', 'SparseRowMatrix',
    'Label', 'DataPair', 'Data', 'ParameterDict', 'HyperParameterDict', 'DataFolder', 'TestResult', 'TestResultTally',
]


MAX_EXP_ARG = log(float_info.max)       # e^x    <= float_info.max
MIN_EXP_ARG = -log(1 / float_info.min)  # e^(-x) >= float_info.min


NumpyArray = ndarray
ZerosVector = zeros
OnesVector = ones

SparseMatrix = spmatrix
SparseRowMatrix = csr_matrix


Label = int
DataPair = NamedTuple('DataPair', [('label', Label), ('vector', SparseMatrix)])
Data = List[DataPair]
ParameterDict = Dict[str, Any]
HyperParameterDict = Dict[str, List[Any]]


class DataFolder:
    def __init__(self, length: int, folds: int, with_replacement=False):
        self.length = length
        self.boundary = int(floor(self.length * (folds - 1) / folds))
        if with_replacement:
            self.indices = [randint(0, self.length) for _ in range(self.length)]
        else:
            self.indices = list(range(self.length))
        shuffle(self.indices)

    def _yield_indices_from_range(self, low: int, high: int) -> Iterator[int]:
        current = low
        while current < high:
            current += 1
            yield self.indices[current - 1]

    def training_indices(self) -> Iterator[int]:
        return self._yield_indices_from_range(0, self.boundary)

    def testing_indices(self) -> Iterator[int]:
        return self._yield_indices_from_range(self.boundary + 1, self.length)

    @property
    def training_length(self):
        return self.boundary

    @property
    def testing_length(self):
        return self.length - self.boundary


class TestResultType(Enum):
    TRUE_POSITIVE = auto()
    TRUE_NEGATIVE = auto()
    FALSE_POSITIVE = auto()
    FALSE_NEGATIVE = auto()


class TestResult:
    def __init__(self, guess: Label, actual: Label):
        self.guess = guess
        self.actual = actual
        if guess == actual == 1:
            self.result = TestResultType.TRUE_POSITIVE
        elif guess == actual == -1:
            self.result = TestResultType.TRUE_NEGATIVE
        elif guess == 1 and actual == -1:
            self.result = TestResultType.FALSE_POSITIVE
        elif guess == -1 and actual == 1:
            self.result = TestResultType.FALSE_NEGATIVE
        else:
            raise ValueError(f"invalid test value; guess: {guess}, actual: {actual}")


class TestResultTally:
    def __init__(self):
        self._tp = 0  # true positives
        self._tn = 0  # true negatives
        self._fp = 0  # false positives
        self._fn = 0  # false negatives

    def tally_result(self, test_result: TestResult):
        result = test_result.result
        if result == TestResultType.TRUE_POSITIVE:
            self._tp += 1
        elif result == TestResultType.TRUE_NEGATIVE:
            self._tn += 1
        elif result == TestResultType.FALSE_POSITIVE:
            self._fp += 1
        elif result == TestResultType.FALSE_NEGATIVE:
            self._fn += 1
        else:
            raise ValueError(f"invalid TestResult: {result}")

    @property
    def total_tests(self) -> int:
        return self._tp + self._tn + self._fp + self._fn

    @property
    def precision(self) -> float:
        return self._tp / (self._tp + self._fp)

    @property
    def recall(self) -> float:
        return self._tp / (self._tp + self._fn)

    @property
    def accuracy(self) -> float:
        if self.total_tests == 0:
            raise RuntimeError("empty testing results")
        return (self._tp + self._tn) / self.total_tests
