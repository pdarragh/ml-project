from ...progress import ProgressMeter
from ...math_and_types import *

from collections import Counter, defaultdict
from datetime import datetime
from numpy import save
from typing import Tuple


def _entropy(data: Data) -> float:
    counts = Counter(pair.label for pair in data)
    p = {x: counts[x] / sum(counts.values()) for x in counts.keys()}
    h_s = sum(-p[x] * log2(p[x]) for x in counts.keys())
    return h_s


def _compute_information_gains(data: Data, feature_count: int, skip_features=None) -> NumpyArray:
    if skip_features is None:
        skip_features = []
    shape1 = (1, feature_count)
    shape2 = (2, feature_count)
    counts = ZerosVector(shape2)
    all_ones = OnesVector(shape1)
    # Count the number of each feature value.
    for data_pair in data:
        v = data_pair.vector
        # Addition has to be done this way for type compatibility.
        counts[0] = counts[0] + (all_ones - v)
        counts[1] = counts[1] + v
    # Features are unimportant if all examples are the same. Discard these.
    features = []
    for i in range(feature_count):
        # Check if we should be skipping this feature.
        if i in skip_features:
            continue
        # Then check if either the zeroes count or the ones count is empty.
        if not counts[0, i] or not counts[1, i]:
            continue
        features.append(i)
    # Calculate the probability of each feature's potential value.
    probs = ZerosVector(shape2)
    values = (0, 1)
    for feature in features:
        total = counts[0, feature] + counts[1, feature]
        for value in values:
            probs[value, feature] = counts[value, feature] / total
    # Now compute information gain for each of these features.
    igs = ZerosVector(feature_count)
    h_s = _entropy(data)
    for feature in features:
        ig_as = h_s - sum(probs[t, feature] * _entropy([pair for pair in data if pair.vector[0, feature] == t])
                          for t in values)
        igs[feature] = ig_as
    return igs


def _make_threshold_matrix_data(data: Data, feature_count: int) -> Tuple[Data, NumpyArray]:
    # Compute the means of the features to use as a threshold.
    # Note that this will still work for binary features.
    means = ZerosVector(feature_count)
    for pair in data:
        means += pair.vector
    means /= len(data)
    # Now determine the binary values for the data.
    shape = (len(data), feature_count)
    dest = ZerosVector(shape)
    for i, pair in enumerate(data):
        comp = pair.vector > means
        for j in range(feature_count):
            if comp[0, j]:
                dest[i, j] = 1
    # Construct a new set of data for use in the tree.
    spm = SparseRowMatrix(dest)  # TODO: This is only done because of assumptions made on the DataPair.vector type.
    new_data: Data = []
    for i, pair in enumerate(data):
        new_pair = DataPair(pair.label, spm[i])
        new_data.append(new_pair)
    return new_data, means


class TreeNode:
    def __init__(self, feature_count: int, examples: Data, depth: int, means: NumpyArray, skip_features: List,
                 tree_no: int, parent=None):
        self.means = means
        self.tree_no = tree_no
        self.parent: TreeNode = parent
        self.children: Dict[Label, TreeNode] = {}
        if not examples:
            raise ValueError("no examples from which to build tree")
        counts = Counter(pair.label for pair in examples)
        # Assign the majority label.
        self.label = max(counts, key=lambda c: counts[c])
        if depth <= 0 or len(counts.keys()) == 1:
            # Don't go any deeper.
            self.feature = None
        else:
            # Find the feature with the greatest information gain and use it here.
            igs = _compute_information_gains(examples, feature_count, skip_features)
            self.feature: int = argmax(igs)
            new_skip_features = [f for f in skip_features if f != self.feature]
            self.label = None
            # Split the data based on that feature.
            indices = defaultdict(list)
            for i, pair in enumerate(examples):
                v = pair.vector
                indices[v[0, self.feature]].append(i)
            for k, idcs in indices.items():
                self.children[k] = TreeNode(feature_count, [examples[i] for i in idcs], depth - 1, means,
                                            new_skip_features, self.tree_no, self)
            if len(self.children) == 1:
                # Something went wrong... so convert this node to a leaf.
                self.children = {}
                self.feature = None
                self.label = max(counts, key=lambda c: counts[c])

    def _get_features(self, features: List[int]):
        features.append(self.feature)
        if self.parent is not None:
            self.parent._get_features(features)

    def __repr__(self):
        features = []
        self._get_features(features)
        return str(self.tree_no) + ': [' + ', '.join(map(str, reversed(features))) + ']'

    def predict(self, data_pair: DataPair) -> Label:
        if self.is_leaf:
            return self.label
        else:
            # Produce a binary value for this feature.
            x = data_pair.vector
            v = 0
            if x[0, self.feature] > self.means[0, self.feature]:
                v = 1
            try:
                result = self.children[v].predict(data_pair)
            except Exception:
                print(repr(self))
                raise
            if result is not None:
                return result
            else:
                raise RuntimeError

    @property
    def is_leaf(self):
        return len(self.children) == 0


class Forest:
    def __init__(self, base_data: Data, n_features: int, max_tree_depth: int, max_tree_count: int, example_count: int,
                 verbose=True, trees=None):
        if trees:
            self.trees = trees
        else:
            self.trees: List[TreeNode] = []
            self._build_trees(base_data, max_tree_count, example_count, max_tree_depth, n_features, verbose)

    def _build_trees(self, data: Data, tree_count: int, example_count: int, max_depth: int, feature_count: int,
                     verbose: bool):
        threshold_data, threshold_means = _make_threshold_matrix_data(data, feature_count)
        try:
            pm = ProgressMeter(tree_count, leader='Training tree: ', print_out=verbose)
            for t in range(tree_count):
                # Each tree needs to be built from the requisite number of examples. Sample those with replacement.
                training_buffer = []
                while len(training_buffer) < example_count:
                    index = randint(0, len(threshold_data) - 1)
                    training_buffer.append(threshold_data[index])
                # Now that we have the examples, build the tree.
                root = TreeNode(feature_count, training_buffer, max_depth, threshold_means, list(), tree_no=t)
                self.trees.append(root)
                pm.update()
            pm.finish()
        finally:
            tfmt = '%Y%m%d-%H%M%S'
            save(f'trees/{datetime.now().strftime(tfmt)}_d{max_depth}_c{tree_count}_e{example_count}.npy', self.trees)
