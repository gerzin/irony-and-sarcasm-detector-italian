import sklearn
import sklearn.metrics
import numpy as np


def computePrecision(tp, fp):
    """
    Return the precision score given true positive and false positive
    """
    if (tp + fp) != 0:
        return tp / (tp + fp)
    return 0


def computeRecall(tp, fn):
    """
    Return the recall score given true positive and false negative
    """
    if (tp + fn) != 0:
        return tp / (tp + fn)
    return 0


def computeF1(tp, fp, fn, tn):
    """
    Return the f1 score given the four components of confusion matrix
    """
    p = computePrecision(tp, fp)
    r = computeRecall(tp, fn)
    if (p + r) != 0:
        return (2 * p * r) / (p + r)
    return 0


def computeAvgF1(confusion_matrix):
    """
    Return the average F1, intended as the mean between first and second class's f1 score
    """
    [tp, fp], [fn, tn] = confusion_matrix
    f_pos = computeF1(tp, fp, fn, tn)
    f_neg = computeF1(tn, fn, fp, tp)
    return (f_pos + f_neg) / 2


def computeAvgF1B(confusion_matrix):
    """
    Return the F1 score from the confusion matrix
    """
    [tp, fp], [fn, tn] = confusion_matrix
    f_pos = computeF1(tp, fp, fn, tn)
    return f_pos


def model_test(model, x_test, y_test):
    """
    params:
    - model: the model to test
    - x_test: input data
    - y_test: target data
    return:
    - Average F1 of the prediction
    """
    y_pred = model.predict(x_test)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_test, np.rint(y_pred))

    return computeAvgF1(confusion_matrix)


def model_test_2out(model, x_test, y_test):
    """
    params:
    - model: the model to test
    - x_test: input data
    - y_test: target data
    return:
    - Average F1 of the prediction
    """
    y_pred = model.predict(x_test)
    y_pred_1, y_pred2 = zip(*y_pred)
    y_test_1 = y_test['irony']
    y_test2 = y_test['sarcasm']
    confusion_matrix_1 = sklearn.metrics.confusion_matrix(y_test_1, np.rint(y_pred_1))
    confusion_matrix_2 = sklearn.metrics.confusion_matrix(y_test2, np.rint(y_pred2))
    AvgF1B = (computeAvgF1(confusion_matrix_1) * 2 + computeAvgF1B(confusion_matrix_2)) / 3
    return computeAvgF1(confusion_matrix_1), AvgF1B


def computePerformanceTaskB_2output(model, x_test, y_test, y_test_A):
    """
    params:
    - model: the model to test
    - x_test: input data
    - y_test: target data (irony, sarcasm)
    - y_test_A: target data (irony)
    return:
    - F1 score reached in task A and task B
    """
    y_pred = model.predict(x_test)
    y_pred_1, y_pred_2 = zip(*y_pred)
    y_pred_1 = np.rint(y_pred_1)
    y_pred_2 = np.rint(y_pred_2)
    y_pred_round = list(zip(y_pred_1, y_pred_2))

    f1_taskA = computeAvgF1(sklearn.metrics.confusion_matrix(y_test_A, y_pred_1))

    y_test_1 = y_test['irony']
    y_test_2 = y_test['sarcasm']
    y_test_round = list(zip(y_test_1, y_test_2))
    # f1 per classe (0,0)
    y_pred_collapsed_1 = []
    for e in y_pred_round:
        if e == (0, 0):
            y_pred_collapsed_1.append(1)
        else:
            y_pred_collapsed_1.append(0)
    y_true_collapsed_1 = []
    for e in y_test_round:
        if e == (0, 0):
            y_true_collapsed_1.append(1)
        else:
            y_true_collapsed_1.append(0)
    tn_1, fp_1, fn_1, tp_1 = sklearn.metrics.confusion_matrix(y_true_collapsed_1, y_pred_collapsed_1).ravel()
    F1_1 = computeF1(tp_1, fp_1, fn_1, tn_1)

    # f1 per classe (1,0)
    y_pred_collapsed_2 = []
    for e in y_pred_round:
        if e == (1, 0):
            y_pred_collapsed_2.append(1)
        else:
            y_pred_collapsed_2.append(0)
    y_true_collapsed_2 = []
    for e in y_test_round:
        if e == (1, 0):
            y_true_collapsed_2.append(1)
        else:
            y_true_collapsed_2.append(0)
    tn_2, fp_2, fn_2, tp_2 = sklearn.metrics.confusion_matrix(y_true_collapsed_2, y_pred_collapsed_2).ravel()
    F1_2 = computeF1(tp_2, fp_2, fn_2, tn_2)

    # f1 per classe (1,1)
    y_pred_collapsed_3 = []
    for e in y_pred_round:
        if e == (1, 1):
            y_pred_collapsed_3.append(1)
        else:
            y_pred_collapsed_3.append(0)
    y_true_collapsed_3 = []
    for e in y_test_round:
        if e == (1, 1):
            y_true_collapsed_3.append(1)
        else:
            y_true_collapsed_3.append(0)
    tn_3, fp_3, fn_3, tp_3 = sklearn.metrics.confusion_matrix(y_true_collapsed_3, y_pred_collapsed_3).ravel()
    F1_3 = computeF1(tp_3, fp_3, fn_3, tn_3)

    return [f1_taskA, (F1_1 + F1_2 + F1_3) / 3]


def computePerformanceTaskB_2output_predicted(y_pred, y_test, y_test_A):
    """
    params:
    - y_pred: prediction of the model
    - y_test: target data (irony, sarcasm)
    - y_test_A: target data (irony)
    return:
    - F1 score reached in task A and task B
    """
    y_pred_1, y_pred_2 = zip(*y_pred)
    y_pred_1 = np.rint(y_pred_1)
    y_pred_2 = np.rint(y_pred_2)
    y_pred_round = list(zip(y_pred_1, y_pred_2))

    f1_taskA = computeAvgF1(sklearn.metrics.confusion_matrix(y_test_A, y_pred_1))

    y_test_1 = y_test['irony']
    y_test_2 = y_test['sarcasm']
    y_test_round = list(zip(y_test_1, y_test_2))
    # f1 per classe (0,0)
    y_pred_collapsed_1 = []
    for e in y_pred_round:
        if e == (0, 0):
            y_pred_collapsed_1.append(1)
        else:
            y_pred_collapsed_1.append(0)
    y_true_collapsed_1 = []
    for e in y_test_round:
        if e == (0, 0):
            y_true_collapsed_1.append(1)
        else:
            y_true_collapsed_1.append(0)
    tn_1, fp_1, fn_1, tp_1 = sklearn.metrics.confusion_matrix(y_true_collapsed_1, y_pred_collapsed_1).ravel()
    F1_1 = computeF1(tp_1, fp_1, fn_1, tn_1)

    # f1 per classe (1,0)
    y_pred_collapsed_2 = []
    for e in y_pred_round:
        if e == (1, 0):
            y_pred_collapsed_2.append(1)
        else:
            y_pred_collapsed_2.append(0)
    y_true_collapsed_2 = []
    for e in y_test_round:
        if e == (1, 0):
            y_true_collapsed_2.append(1)
        else:
            y_true_collapsed_2.append(0)
    tn_2, fp_2, fn_2, tp_2 = sklearn.metrics.confusion_matrix(y_true_collapsed_2, y_pred_collapsed_2).ravel()
    F1_2 = computeF1(tp_2, fp_2, fn_2, tn_2)

    # f1 per classe (1,1)
    y_pred_collapsed_3 = []
    for e in y_pred_round:
        if e == (1, 1):
            y_pred_collapsed_3.append(1)
        else:
            y_pred_collapsed_3.append(0)
    y_true_collapsed_3 = []
    for e in y_test_round:
        if e == (1, 1):
            y_true_collapsed_3.append(1)
        else:
            y_true_collapsed_3.append(0)
    tn_3, fp_3, fn_3, tp_3 = sklearn.metrics.confusion_matrix(y_true_collapsed_3, y_pred_collapsed_3).ravel()
    F1_3 = computeF1(tp_3, fp_3, fn_3, tn_3)

    return [f1_taskA, (F1_1 + F1_2 + F1_3) / 3]


def computePerformanceTaskB_2model(model1, model2, x_test, y_test):
    """
    params:
    - model1: model trained on irony
    - model2: model trained on sarcasm
    - x_test: input data
    - y_test: target data (irony, sarcasm)
    return:
    - F1 score reached in task A and task B
    """
    y_pred_1 = model1.predict(x_test)
    y_pred_2 = model2.predict(x_test)
    y_pred_1 = np.rint(y_pred_1)
    y_pred_2 = np.rint(y_pred_2)
    y_pred_round = list(zip(y_pred_1, y_pred_2))

    y_test_1 = y_test['irony']
    y_test_2 = y_test['sarcasm']
    y_test_round = list(zip(y_test_1, y_test_2))
    # f1 per classe (0,0)
    y_pred_collapsed_1 = []
    for e in y_pred_round:
        if e == (0, 0):
            y_pred_collapsed_1.append(1)
        else:
            y_pred_collapsed_1.append(0)
    y_true_collapsed_1 = []
    for e in y_test_round:
        if e == (0, 0):
            y_true_collapsed_1.append(1)
        else:
            y_true_collapsed_1.append(0)
    tn_1, fp_1, fn_1, tp_1 = sklearn.metrics.confusion_matrix(y_true_collapsed_1, y_pred_collapsed_1).ravel()
    F1_1 = computeF1(tp_1, fp_1, fn_1, tn_1)

    # f1 per classe (1,0)
    y_pred_collapsed_2 = []
    for e in y_pred_round:
        if e == (1, 0):
            y_pred_collapsed_2.append(1)
        else:
            y_pred_collapsed_2.append(0)
    y_true_collapsed_2 = []
    for e in y_test_round:
        if e == (1, 0):
            y_true_collapsed_2.append(1)
        else:
            y_true_collapsed_2.append(0)
    tn_2, fp_2, fn_2, tp_2 = sklearn.metrics.confusion_matrix(y_true_collapsed_2, y_pred_collapsed_2).ravel()
    F1_2 = computeF1(tp_2, fp_2, fn_2, tn_2)

    # f1 per classe (1,1)
    y_pred_collapsed_3 = []
    for e in y_pred_round:
        if e == (1, 1):
            y_pred_collapsed_3.append(1)
        else:
            y_pred_collapsed_3.append(0)
    y_true_collapsed_3 = []
    for e in y_test_round:
        if e == (1, 1):
            y_true_collapsed_3.append(1)
        else:
            y_true_collapsed_3.append(0)
    tn_3, fp_3, fn_3, tp_3 = sklearn.metrics.confusion_matrix(y_true_collapsed_3, y_pred_collapsed_3).ravel()
    F1_3 = computeF1(tp_3, fp_3, fn_3, tn_3)

    return (F1_1 + F1_2 + F1_3) / 3


def f1_taskA_2output(model, x, y):
    y_pred = model.predict(x_test)
    y_pred_1, y_pred_2 = zip(*y_pred)