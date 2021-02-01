# Multi-Label-Metrics
The metrics for evaluating the performance of multi-class multi-label model.

Refer to [1] [2]，evaluation metrics for multi-class multi-label model are divided into two major categories: example-based metrics and label-based metrics.
### Example-based Metrics
1. Subset accuracy
   $$subsetacc(h)=\frac{1}{p} \sum_{i=1}^{p} I\left[h\left(x_{i}\right)=Y_{i}\right]$$

```python
def example_subset_accuracy(gt, predict):
    ex_equal = np.all(np.equal(gt, predict), axis=1).astype("float32")
    return np.mean(ex_equal)
```

2. Example accuracy
   $$\operatorname{Accuracy}_{\operatorname{exam}}(h)=\frac{1}{p} \sum_{i=1}^{p} \frac{\left|Y_{i} \cap h\left(x_{i}\right)\right|}{\left|Y_{i} \cup h\left(x_{i}\right)\right|}$$

```python
def example_accuracy(gt, predict):
    ex_and = np.sum(np.logical_and(gt, predict), axis=1).astype("float32")
    ex_or = np.sum(np.logical_or(gt, predict), axis=1).astype("float32")
    return np.mean(ex_and / (ex_or+epsilon))
```

3. Example precision
   $$\text { Precision }_{\text {exam }}(h)=\frac{1}{p} \sum_{i=1}^{p} \frac{\left|Y_{i} \cap h\left(x_{i}\right)\right|}{\left|h\left(x_{i}\right)\right|}$$

```python
def example_precision(gt, predict):
    ex_and = np.sum(np.logical_and(gt, predict), axis=1).astype("float32")
    ex_predict = np.sum(predict, axis=1).astype("float32")
    return np.mean(ex_and / (ex_predict + epsilon))
```

4. Example recall
   $$\text { Recall }_{\text exam}(h)=\frac{1}{p} \sum_{i=1}^{p} \frac{\left|Y_{i} \cap h\left(x_{i}\right)\right|}{\left|Y_{i}\right|}$$

```python
def example_recall(gt, predict):
    ex_and = np.sum(np.logical_and(gt, predict), axis=1).astype("float32")
    ex_gt = np.sum(gt, axis=1).astype("float32")
    return np.mean(ex_and / (ex_gt + epsilon))
```

5. Example F1 (with $\beta$)
   $$F_{\text {exam }}^{\beta}(h)=\frac{\left(1+\beta^{2}\right) \cdot \text { Precsion }_{\text {exam }}(h) \cdot \text { Recall }_{\text {exam }}(h)}{\beta^{2} \cdot \text { Precision }_{\text {exam }}(h)+\text { Recall }_{\text {exam }}(h)}$$

```python
def example_f1(gt, predict, beta=1):
    p = example_precision(gt, predict)
    r = example_recall(gt, predict)
    return ((1+beta**2) * p * r) / ((beta**2)*(p + r + epsilon))
```

### Label-based Metrics


- calculate $TP,TN,FP,FN$
  $$\begin{array}{l}T P_{j}=\left|\left\{x_{i} \mid y_{j} \in Y_{i} \wedge y_{j} \in h\left(x_{i}\right), 1 \leq i \leq p\right\}\right| \\ F P_{j}=\left|\left\{x_{i} \mid y_{j} \notin Y_{i} \wedge y_{j} \in h\left(x_{i}\right), 1 \leq i \leq p\right\}\right| \\ T N_{j}=\left|\left\{x_{i} \mid y_{j} \notin Y_{i} \wedge y_{j} \notin h\left(x_{i}\right), 1 \leq i \leq p\right\}\right| \\ F N_{j}=\left|\left\{x_{i} \mid y_{j} \in Y_{i} \wedge y_{j} \notin h\left(x_{i}\right), 1 \leq i \leq p\right\}\right|\end{array}$$

```python
def _label_quantity(gt, predict):
    tp = np.sum(np.logical_and(gt, predict), axis=0)
    fp = np.sum(np.logical_and(1-gt, predict), axis=0)
    tn = np.sum(np.logical_and(1-gt, 1-predict), axis=0)
    fn = np.sum(np.logical_and(gt, 1-predict), axis=0)
    return np.stack([tp, fp, tn, fn], axis=0).astype("float")
```

- calculate Accuracy, Precision, Recall, and F1
  $$\begin{array}{c}\text { Accuracy }\left(T P_{j}, F P_{j}, T N_{j}, F N_{j}\right)=\frac{T P_{j}+T N_{j}}{T P_{j}+F P_{j}+T N_{j}+F N_{j}} \\ \text { Precision }\left(T P_{j}, F P_{j}, T N_{j}, F N_{j}\right)=\frac{T P_{j}}{T P_{j}+F P_{j}} \\ \operatorname{Recall}\left(T P_{j}, F P_{j}, T N_{j}, F N_{j}\right)=\frac{T P_{j}}{T P_{j}+F N_{j}} \\ F^{\beta}\left(T P_{j}, F P_{j}, T N_{j}, F N_{j}\right)=\frac{\left(1+\beta^{2}\right) \cdot T P_{j}}{\left(1+\beta^{2}\right) T P_{j}+\beta^{2} \cdot F N_{j}+F P_{j}}\end{array}$$

- Marco-average and Micro-average

  $$B_{\operatorname{macro}}(h)=\frac{1}{q} \sum_{j=1}^{q} B\left(T P_{j}, F P_{j}, T N_{j}, F N_{j}\right) \\ B_{\text {micro }}(h)=B\left(\sum_{j=1}^{q} T P_{j}, \sum_{j=1}^{q} F P_{j}, \sum_{j=1}^{q} T N_{j}, \sum_{j=1}^{q} F N_{j}\right)$$

1. Label accuracy

  - Macro

```python
def label_accuracy_macro(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp_tn = np.add(quantity[0], quantity[2])
    tp_fp_tn_fn = np.sum(quantity, axis=0)
    return np.mean(tp_tn / (tp_fp_tn_fn + epsilon))
```
  - Micro 

```python
def label_accuracy_micro(gt, predict):
    quantity = _label_quantity(gt, predict)
    sum_tp, sum_fp, sum_tn, sum_fn = np.sum(quantity, axis=1)
    return (sum_tp + sum_tn) / (
            sum_tp + sum_fp + sum_tn + sum_fn + epsilon)
```

2. Label precision

- Macro

```python
def label_precision_macro(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp = quantity[0]
    tp_fp = np.add(quantity[0], quantity[1])
    return np.mean(tp / (tp_fp + epsilon))
```

  - Micro
```python
def label_precision_micro(gt, predict):
    quantity = _label_quantity(gt, predict)
    sum_tp, sum_fp, sum_tn, sum_fn = np.sum(quantity, axis=1)
    return sum_tp / (sum_tp + sum_fp + epsilon)
```

3. Label recall
  - Macro
```python
def label_recall_macro(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp = quantity[0]
    tp_fn = np.add(quantity[0], quantity[3])
    return np.mean(tp / (tp_fn + epsilon))
```
  - Micro
```python
def label_recall_micro(gt, predict):
    quantity = _label_quantity(gt, predict)
    sum_tp, sum_fp, sum_tn, sum_fn = np.sum(quantity, axis=1)
    return sum_tp / (sum_tp + sum_fn + epsilon)
```
4. Label F1
- Macro
```python
def label_f1_macro(gt, predict, beta=1):
    quantity = _label_quantity(gt, predict)
    tp = quantity[0]
    fp = quantity[1]
    fn = quantity[3]
    return np.mean((1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp))
```
- Micro
```python
def label_f1_micro(gt, predict, beta=1):
    quantity = _label_quantity(gt, predict)
    tp = np.sum(quantity[0])
    fp = np.sum(quantity[1])
    fn = np.sum(quantity[3])
    return (1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp)
```




## Reference
[1] M. Zhang and Z. Zhou, "A Review on Multi-Label Learning Algorithms," in *IEEE Transactions on Knowledge and Data Engineering*, vol. 26, no. 8, pp. 1819-1837, Aug. 2014, doi: 10.1109/TKDE.2013.39.

[2]Wei Long, Yang Yang, Hong-Bin Shen, ImPLoc: a multi-instance deep learning model for the prediction of protein subcellular localization based on immunohistochemistry images, *Bioinformatics*, Volume 36, Issue 7, 1 April 2020, Pages 2244–2250, https://doi.org/10.1093/bioinformatics/btz909
