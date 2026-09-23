import os.path as osp
from collections import OrderedDict, defaultdict

import jittor as jt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from dassl.utils import Registry, check_availability


EVALUATOR_REGISTRY = Registry("EVALUATOR")


@EVALUATOR_REGISTRY.register()
class Classification:
    def __init__(self, cfg, lab2cname=None, **kwargs):
        self.cfg = cfg
        self._lab2cname = lab2cname
        self._per_class_res = defaultdict(list) if cfg.TEST.PER_CLASS_RESULT else None
        self.reset()

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, model_output, ground_truth):
        output = model_output.numpy() if isinstance(model_output, jt.Var) else np.asarray(model_output)
        target = ground_truth.numpy() if isinstance(ground_truth, jt.Var) else np.asarray(ground_truth)
        target = target.reshape(-1).astype(np.int64)
        pred = output.argmax(axis=1)
        matches = pred == target
        self._correct += int(matches.sum())
        self._total += int(target.shape[0])
        self._y_true.extend(target.tolist())
        self._y_pred.extend(pred.tolist())
        if self._per_class_res is not None:
            for label, match in zip(target, matches):
                self._per_class_res[int(label)].append(int(match))

    def evaluate(self):
        accuracy = 100.0 * self._correct / self._total
        error = 100.0 - accuracy
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true),
        )
        results = OrderedDict(
            accuracy=accuracy,
            error_rate=error,
            macro_f1=macro_f1,
        )
        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {accuracy:.1f}%\n"
            f"* error: {error:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )
        if self._per_class_res is not None:
            print("=> per-class result")
            values = []
            for label in sorted(self._per_class_res):
                matches = self._per_class_res[label]
                value = 100.0 * sum(matches) / len(matches)
                values.append(value)
                print(
                    f"* class: {label} ({self._lab2cname[label]})\t"
                    f"total: {len(matches):,}\tcorrect: {sum(matches):,}\tacc: {value:.1f}%"
                )
            results["perclass_accuracy"] = float(np.mean(values))
        if self.cfg.TEST.COMPUTE_CMAT:
            matrix = confusion_matrix(self._y_true, self._y_pred, normalize="true")
            path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pkl")
            jt.save(matrix, path)
            print(f"Confusion matrix is saved to {path}")
        return results


def build_evaluator(cfg, **kwargs):
    available = EVALUATOR_REGISTRY.registered_names()
    check_availability(cfg.TEST.EVALUATOR, available)
    if cfg.VERBOSE:
        print(f"Loading evaluator: {cfg.TEST.EVALUATOR}")
    return EVALUATOR_REGISTRY.get(cfg.TEST.EVALUATOR)(cfg, **kwargs)


__all__ = ["build_evaluator", "EVALUATOR_REGISTRY"]
