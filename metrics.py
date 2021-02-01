import numpy as np

# modified from https://github.com/yl2019lw/ImPloc/blob/revision/util/npmetrics.py

epsilon = 1e-8 # to aviod zero-divison

# Example-based metrics

# gt (numpy boolean) shape: N x nlabel
# predict (numpy boolean) shape: N x nlabel 
def example_subset_accuracy(gt, predict):
    ex_equal = np.all(np.equal(gt, predict), axis=1).astype("float32")
    return np.mean(ex_equal)


def example_accuracy(gt, predict):
    ex_and = np.sum(np.logical_and(gt, predict), axis=1).astype("float32")
    ex_or = np.sum(np.logical_or(gt, predict), axis=1).astype("float32")
    return np.mean(ex_and / (ex_or+epsilon))


def example_precision(gt, predict):
    ex_and = np.sum(np.logical_and(gt, predict), axis=1).astype("float32")
    ex_predict = np.sum(predict, axis=1).astype("float32")
    return np.mean(ex_and / (ex_predict + epsilon))


def example_recall(gt, predict):
    ex_and = np.sum(np.logical_and(gt, predict), axis=1).astype("float32")
    ex_gt = np.sum(gt, axis=1).astype("float32")
    return np.mean(ex_and / (ex_gt + epsilon))


def example_f1(gt, predict, beta=1):
    p = example_precision(gt, predict)
    r = example_recall(gt, predict)
    return ((1+beta**2) * p * r) / ((beta**2)*(p + r + epsilon))


# Label-based metrics

def _label_quantity(gt, predict):
    tp = np.sum(np.logical_and(gt, predict), axis=0)
    fp = np.sum(np.logical_and(1-gt, predict), axis=0)
    tn = np.sum(np.logical_and(1-gt, 1-predict), axis=0)
    fn = np.sum(np.logical_and(gt, 1-predict), axis=0)
    return np.stack([tp, fp, tn, fn], axis=0).astype("float")


def label_accuracy_macro(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp_tn = np.add(quantity[0], quantity[2])
    tp_fp_tn_fn = np.sum(quantity, axis=0)
    return np.mean(tp_tn / (tp_fp_tn_fn + epsilon))


def label_accuracy_micro(gt, predict):
    quantity = _label_quantity(gt, predict)
    sum_tp, sum_fp, sum_tn, sum_fn = np.sum(quantity, axis=1)
    return (sum_tp + sum_tn) / (
            sum_tp + sum_fp + sum_tn + sum_fn + epsilon)


def label_precision_macro(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp = quantity[0]
    tp_fp = np.add(quantity[0], quantity[1])
    return np.mean(tp / (tp_fp + epsilon))


def label_precision_micro(gt, predict):
    quantity = _label_quantity(gt, predict)
    sum_tp, sum_fp, sum_tn, sum_fn = np.sum(quantity, axis=1)
    return sum_tp / (sum_tp + sum_fp + epsilon)


def label_recall_macro(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp = quantity[0]
    tp_fn = np.add(quantity[0], quantity[3])
    return np.mean(tp / (tp_fn + epsilon))


def label_recall_micro(gt, predict):
    quantity = _label_quantity(gt, predict)
    sum_tp, sum_fp, sum_tn, sum_fn = np.sum(quantity, axis=1)
    return sum_tp / (sum_tp + sum_fn + epsilon)


def label_f1_macro(gt, predict, beta=1):
    quantity = _label_quantity(gt, predict)
    tp = quantity[0]
    fp = quantity[1]
    fn = quantity[3]
    return np.mean((1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp  + epsilon))


def label_f1_micro(gt, predict, beta=1):
    quantity = _label_quantity(gt, predict)
    tp = np.sum(quantity[0])
    fp = np.sum(quantity[1])
    fn = np.sum(quantity[3])
    return (1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp + epsilon)


# test code
if __name__ == "__main__":
    gt = np.load("gt.npy")
    predict = np.load("predict.npy")
    subset_acc = example_subset_accuracy(gt, predict)
    ex_acc = example_accuracy(gt, predict)
    ex_precision = example_precision(gt, predict)
    ex_recall = example_recall(gt, predict)
    ex_f1 = example_f1(gt, predict)

    lab_acc_ma = label_accuracy_macro(gt, predict)
    lab_acc_mi = label_accuracy_micro(gt, predict)
    lab_precision_ma = label_precision_macro(gt, predict)
    lab_precision_mi = label_precision_micro(gt, predict)
    lab_recall_ma = label_recall_macro(gt, predict)
    lab_recall_mi = label_recall_micro(gt, predict)
    lab_f1_ma = label_f1_macro(gt, predict)
    lab_f1_mi = label_f1_micro(gt, predict)

    print("subset acc:        %.4f\n" %subset_acc)
    print("example acc:       %.4f\n" %ex_acc)
    print("example precision: %.4f\n" %ex_precision)
    print("example recall:    %.4f\n" %ex_recall)
    print("example f1:        %.4f\n" %ex_f1)

    print("label acc macro:   %.4f\n" %lab_acc_ma)
    print("label acc micro:   %.4f\n" %lab_acc_mi)
    print("label prec macro:  %.4f\n" %lab_precision_ma)
    print("label prec micro:  %.4f\n" %lab_precision_mi)
    print("label rec macro:   %.4f\n" %lab_recall_ma)
    print("label rec micro:   %.4f\n" %lab_recall_mi)
    print("label f1 macro:    %.4f\n" %lab_f1_ma)
    print("label f1 micro:    %.4f\n" %lab_f1_mi)