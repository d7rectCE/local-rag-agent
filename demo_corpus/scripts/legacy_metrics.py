# Старый ручной подсчёт метрик из первой версии проекта.
# УСТАРЕЛО: используйте ml_toolkit.metrics.calc_metrics.


def confusion_counts(y_true, y_pred, positive=1):
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if p == positive and t == positive:
            tp += 1
        elif p == positive:
            fp += 1
        elif t == positive:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def f1_manual(y_true, y_pred, positive=1):
    """F1 для бинарной задачи без sklearn."""
    tp, fp, fn, _ = confusion_counts(y_true, y_pred, positive)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    print(f1_manual([1, 0, 1, 1, 0], [1, 0, 0, 1, 1]))
