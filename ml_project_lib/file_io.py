from .math_and_types import *

from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from typing import Tuple


def _read_label(raw_label: float) -> Label:
    if raw_label > 0:
        return 1
    else:
        return -1


def convert_matrices_to_data(data_matrix: NumpyArray, label_array: NumpyArray) -> Data:
    data: Data = []
    for i in range(len(label_array)):
        pair = DataPair(_read_label(label_array[i]), data_matrix[i])
        data.append(pair)
    return data


def convert_data_to_matrices(data: Data, n_features: int) -> Tuple[NumpyArray, NumpyArray]:
    if not data:
        raise ValueError
    n_examples = len(data)
    labels_shape = (n_examples)
    examples_shape = (n_examples, n_features)
    labels = ZerosVector(labels_shape)
    examples = ZerosVector(examples_shape)
    for i, dp in enumerate(data):
        labels[i] = dp.label
        examples[i] += dp.vector
    return examples, labels


def read_file(filename: str, n_features: int) -> Tuple[Data, List[int]]:
    id_filename = f'{filename}.id'
    data = read_liblinear_file(filename, n_features)
    ids = read_id_file(id_filename)
    return data, ids


def read_id_file(filename: str) -> List[int]:
    ids: List[int] = []
    with open(filename) as f:
        for line in f:
            ident = int(line.strip())
            ids.append(ident)
    return ids


def read_liblinear_file(filename: str, n_features: int) -> Data:
    tup = load_svmlight_file(filename, n_features=n_features)
    svs: SparseRowMatrix = tup[0]
    labels: NumpyArray = tup[1]
    return convert_matrices_to_data(svs, labels)


def write_csv(filename: str, ids: List[int], predictions: List[Label]):
    with open(filename, 'w') as f:
        f.write("Id,Prediction\n")
        for ident, label in zip(ids, predictions):
            f.write(f"{ident},{int(label > 0)}\n")


def write_liblinear(filename: str, data: Data, n_features: int):
    examples, labels = convert_data_to_matrices(data, n_features)
    dump_svmlight_file(examples, labels, filename, zero_based=False)
