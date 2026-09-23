import jittor as jt
import numpy as np


def compute_accuracy(output, target, topk=(1,)):
    logits = output.numpy() if isinstance(output, jt.Var) else np.asarray(output)
    labels = target.numpy() if isinstance(target, jt.Var) else np.asarray(target)
    order = np.argsort(logits, axis=1)[:, ::-1]
    result = []
    for k in topk:
        correct = (order[:, :k] == labels.reshape(-1, 1)).any(axis=1).mean() * 100.0
        result.append(jt.array([correct], dtype="float32"))
    return result


__all__ = ["compute_accuracy"]
