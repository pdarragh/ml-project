from .math_and_types import *
from .learners import ParameterizedLearner
from .progress import ProgressMeter

from itertools import chain, product
from typing import Tuple, Type


FOLDS = 5
TRAINING_EPOCHS = 3
TESTING_EPOCHS = 10
VERBOSE = True


def set_verbosity(verbosity: bool):
    global VERBOSE
    VERBOSE = verbosity


def print_out(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def cross_validate(learner_class: Type[ParameterizedLearner], parameters: HyperParameterDict, training_data: Data,
                   k=FOLDS, e=TRAINING_EPOCHS) -> float:
    tallies = []
    df = DataFolder(len(training_data), folds=k)
    print_out(f"  Cross-validation for {learner_class.__name__}")
    print_out(f"    With parameters: {parameters}")
    print_out()
    for i in range(k):
        print_out(f"    Fold: {i+1}/{k}")
        learner = learner_class(**parameters)
        pm = ProgressMeter(total=e*df.training_length, print_out=VERBOSE)
        for t in range(e):
            pm.set_leader(f'      Training epoch {t+1}/{e}: ')
            for index in df.training_indices():
                training_pair = training_data[index]
                learner.train(training_pair)
                pm.update()
        pm.finish()
        tally = TestResultTally()
        pm = ProgressMeter(total=df.testing_length, leader=f'      Testing: ', print_out=VERBOSE)
        for index in df.testing_indices():
            testing_pair = training_data[index]
            try:
                tally.tally_result(learner.test(testing_pair))
            except Exception:
                print()
                print(f"error with training index: {index}")
                raise
            pm.update()
        pm.finish()
        tallies.append(tally)
        print_out(f"      Testing accuracy: {tally.accuracy}")
    avg_accuracy = sum(t.accuracy for t in tallies) / len(tallies)
    print_out(f"    Average accuracy: {avg_accuracy}")
    print_out()
    return avg_accuracy


def k_fold_cross_validate(learner_class: Type[ParameterizedLearner], hyper_parameters: HyperParameterDict,
                          extra_parameters: ParameterDict, training_data: Data) -> Tuple[float, HyperParameterDict]:
    cross_validation_results = []
    hp_permutations = product(*hyper_parameters.values())
    for permutation in hp_permutations:
        parameters = {parameter: value for parameter, value in zip(hyper_parameters.keys(), permutation)}
        parameters.update(extra_parameters)
        avg_accuracy = cross_validate(learner_class, parameters, training_data)
        tup = (avg_accuracy, parameters)
        cross_validation_results.append(tup)
    best_accuracy, best_params = max(cross_validation_results, key=lambda t: t[0])
    return best_accuracy, best_params


def test_parameterized_learner(learner_class: Type[ParameterizedLearner], hyper_parameters: HyperParameterDict,
                               extra_parameters: ParameterDict, training_data: Data, testing_data: Data,
                               e=TESTING_EPOCHS) -> Tuple[TestResultTally, TestResultTally, float, HyperParameterDict,
                                                          List[Label]]:
    best_accuracy, best_params = k_fold_cross_validate(learner_class, hyper_parameters, extra_parameters, training_data)
    print_out(f"  Selected hyper-parameters:")
    for param, val in best_params.items():
        print_out(f"    {param}: {val}")
    print_out()
    all_params = {k: v for k, v in chain(best_params.items(), extra_parameters.items())}
    learner = learner_class(**all_params)
    pm = ProgressMeter(total=e*len(training_data), leader=f'  Training: ', print_out=VERBOSE)
    for t in range(e):
        for training_pair in training_data:
            learner.train(training_pair)
            pm.update()
    pm.finish()
    # Run trained classifier on training data.
    training_tally = TestResultTally()
    pm = ProgressMeter(total=len(training_data), leader='  Testing on training data: ', print_out=VERBOSE)
    for pair in training_data:
        training_tally.tally_result(learner.test(pair))
        pm.update()
    pm.finish()
    # Run trained classifier on testing data.
    testing_tally = TestResultTally()
    pm = ProgressMeter(total=len(testing_data), leader=f'  Testing on testing data: ', print_out=VERBOSE)
    predictions = []
    for pair in testing_data:
        result = learner.test(pair)
        predictions.append(result.guess)
        testing_tally.tally_result(result)
        pm.update()
    pm.finish()
    print_out()
    return training_tally, testing_tally, best_accuracy, best_params, predictions


def test_majority_baseline(training_data: Data, testing_data: Data) -> Tuple[int, float]:
    pm = ProgressMeter(total=len(training_data), leader='  Training: ', print_out=VERBOSE)
    # Count the labels in the training set to determine a predictor.
    count = 0
    for dt in training_data:
        count += dt.label
        pm.update()
    pm.finish()
    # Labels should be {-1, 1}, so a count of 0 means equal numbers.
    if count > 0:
        label = 1
    elif count < 0:
        label = -1
    else:
        if randint(1, 2) == 1:
            label = 1
        else:
            label = -1
    # Now go through the test data to predict.
    total = 0
    correct = 0
    pm = ProgressMeter(total=len(testing_data), leader='  Testing: ', print_out=VERBOSE)
    for dt in testing_data:
        total += 1
        if dt.label == label:
            correct += 1
        pm.update()
    pm.finish()
    test_accuracy = correct / total
    return label, test_accuracy
