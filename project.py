#!/usr/bin/env python3

from ml_project_lib import *

from datetime import datetime
from sys import exit
from typing import List, Type


FOREST = None

TREE_DEPTH = 25
TREE_COUNT = 1000
EXAMPLE_COUNT = 100


def get_forest(training_data: Data, feature_count: int, tree_depth: int, tree_count: int,
               example_count: int):
    global FOREST
    if FOREST is None:
        FOREST = Forest(training_data, feature_count, tree_depth, tree_count, example_count)
    return FOREST


def _run_test(name: str, learner_class: Type[ParameterizedLearner], training_data: Data,
              testing_data: Data, hyper_parameters: HyperParameterDict, extra_parameters: ParameterDict,
              testing_ids: List[int]):
    print(name)
    (training_tally,
     testing_tally,
     kcv_acc,
     best_params,
     predictions) = test_parameterized_learner(learner_class, hyper_parameters, extra_parameters, training_data,
                                               testing_data)
    print(f"  k-CV accuracy: {kcv_acc}")
    print(f"  Training set accuracy: {training_tally.accuracy}")
    print(f"  Testing set accuracy: {testing_tally.accuracy}")
    tfmt = '%Y%m%d-%H%M%S'
    filename = f'Result_{name.replace(" ", "-")}_{datetime.now().strftime(tfmt)}.csv'
    write_csv(filename, testing_ids, predictions)


def run_baseline(training_data: Data, testing_data: Data):
    print(f"Majority Baseline")
    mb_label, test_acc = test_majority_baseline(training_data, testing_data)
    print(f"  Baseline label:       {mb_label}")
    print(f"  Testing accuracy:     {test_acc * 100:0.2f}% ({test_acc})")


def run_test(learner_name: str, training_data: Data, testing_data: Data, testing_ids: List[int], feature_count: int,
             tree_depth: int, tree_count: int, example_count: int):
    if learner_name == 'all':
        for name in LEARNERS:
            run_test(name, training_data, testing_data, testing_ids, feature_count, tree_depth, tree_count, example_count)
        return
    learner = LEARNERS[learner_name]
    extra_parameters = {}
    if 'feature_count' in learner['extra_parameters']:
        extra_parameters['n_features'] = feature_count
    if 'forest' in learner['extra_parameters']:
        extra_parameters['forest'] = get_forest(training_data, feature_count, tree_depth, tree_count, example_count)
    _run_test(learner['name'], learner['class'], training_data, testing_data, learner['hyper_parameters'],
              extra_parameters, testing_ids)


LEARNERS = {
    'svm': {
        'name': 'SVM with SGD',
        'class': SVM,
        'hyper_parameters': {
            'learning_rate': [eval(f'1e{_}') for _ in range(1, -5, -1)],
            'tradeoff': [eval(f'1e{_}') for _ in range(1, -5, -1)],
        },
        'extra_parameters': ['feature_count'],
    },
    'logreg': {
        'name': 'Logistic Regression',
        'class': LogisticRegression,
        'hyper_parameters': {
            'learning_rate': [eval(f'1e{_}') for _ in range(0, -6, -1)],
            'tradeoff': [eval(f'1e{_}') for _ in range(-1, 5)],
        },
        'extra_parameters': ['feature_count'],
    },
    'tree': {
        'name': 'Bagged Forest',
        'class': SimpleForestLearner,
        'hyper_parameters': {},
        'extra_parameters': ['forest'],
    },
    'tree-svm': {
        'name': 'SVM Over Trees',
        'class': SVMOverTrees,
        'hyper_parameters': {
            'learning_rate': [eval(f'1e{_}') for _ in range(0, -6, -1)],
            'tradeoff': [eval(f'1e{_}') for _ in range(1, -6, -1)],
        },
        'extra_parameters': ['forest'],
    },
    'tree-logreg': {
        'name': 'LogReg Over Trees',
        'class': LogregOverTrees,
        'hyper_parameters': {
            'learning_rate': [eval(f'1e{_}') for _ in range(0, -6, -1)],
            'tradeoff': [eval(f'1e{_}') for _ in range(-1, 4)],
        },
        'extra_parameters': ['forest'],
    },
    'perceptron': {
        'name': 'Simple Perceptron',
        'class': SimplePerceptron,
        'hyper_parameters': {
            'learning_rate': [eval(f'1e{_}') for _ in range(1, -4, -1)],
        },
        'extra_parameters': ['feature_count'],
    },
    'perceptron-dynamic': {
        'name': 'Dynamic Perceptron',
        'class': DynamicPerceptron,
        'hyper_parameters': {
            'learning_rate': [eval(f'1e{_}') for _ in range(1, -4, -1)],
        },
        'extra_parameters': ['feature_count'],
    },
    'perceptron-margin': {
        'name': 'Margin Perceptron',
        'class': MarginPerceptron,
        'hyper_parameters': {
            'learning_rate': [eval(f'1e{_}') for _ in range(1, -4, -1)],
            'margin': [eval(f'1e{_}') for _ in range(1, -4, -1)],
        },
        'extra_parameters': ['feature_count'],
    },
    'perceptron-averaged': {
        'name': 'Averaged Perceptron',
        'class': AveragedPerceptron,
        'hyper_parameters': {
            'learning_rate': [eval(f'1e{_}') for _ in range(1, -4, -1)],
        },
        'extra_parameters': ['feature_count'],
    },
    'perceptron-aggressive': {
        'name': 'Aggressive Margin Perceptron',
        'class': AggressiveMarginPerceptron,
        'hyper_parameters': {
            'margin': [eval(f'1e{_}') for _ in range(1, -4, -1)],
        },
        'extra_parameters': ['feature_count'],
    },
}


def print_result(test_result: TestResult, kcv_accuracy=None):
    result = test_result.result
    print(f"  Hyper-parameters:")
    print(f"    margin:        {result.best_margin if result.best_margin is not None else 'N/A'}")
    print(f"    learning rate: {result.best_learning_rate if result.best_learning_rate is not None else 'N/A'}")
    if kcv_accuracy is not None:
        print(f"  Cross-validation accuracy:    {kcv_accuracy * 100:0.2f}% ({kcv_accuracy})")
    print(f"  Updates performed:            {result.updates}")
    print(f"  Test set accuracy:            {result.test_accuracy * 100:0.2f}% ({result.test_accuracy})")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('training_data', help="The training data file, in liblinear format.")
    parser.add_argument('testing_data', help="The testing data file, in liblinear format.")
    parser.add_argument('--learner', choices=list(LEARNERS.keys()) + ['baseline', 'all'], default='all',
                        help="Which learner to use; default = 'all'.")
    parser.add_argument('--vector_size', type=int, default=16, help="The number of features.")
    parser.add_argument('-c', '--tree-count', type=int, default=TREE_COUNT, help="Max number of trees to build.")
    parser.add_argument('-d', '--tree-depth', type=int, default=TREE_DEPTH, help="Max depth of each tree.")
    parser.add_argument('-e', '--forest-examples', type=int, default=EXAMPLE_COUNT,
                        help="Number of examples to use for each tree.")
    parser.add_argument('--build-trees', action='store_true', help="Build trees for the data and quit.")
    parser.add_argument('--tree-file', help="A file containing all the trees, if they have been computed before.")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Print out extra information to convince you that computation has not halted "
                             "unexpectedly..")
    args = parser.parse_args()

    features = args.vector_size
    data_for_training, ids_for_training = read_file(args.training_data, n_features=features)
    data_for_testing, ids_for_testing = read_file(args.testing_data, n_features=features)

    trees = args.tree_count
    depth = args.tree_depth
    examples = args.forest_examples

    set_verbosity(args.verbose)

    if args.build_trees:
        get_forest(data_for_training, features, depth, trees, examples)
        exit(0)

    if args.tree_file:
        from numpy import load
        raw_trees = load(args.tree_file)
        trees = raw_trees.tolist()
        print(f"Loaded trees from file: {args.tree_file}")
        FOREST = Forest([], 0, 0, 0, 0, None, trees)

    learn = args.learner
    if learn in ('all', 'baseline'):
        run_baseline(data_for_training, data_for_testing)
    if learn != 'baseline':
        run_test(args.learner, data_for_training, data_for_testing, ids_for_testing, features, depth, trees, examples)
