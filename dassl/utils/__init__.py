import errno
import json
import os
import os.path as osp
import platform
import random
import shutil
import sys
import time
import warnings
from collections import OrderedDict, defaultdict
from difflib import SequenceMatcher

import jittor as jt
import numpy as np
import PIL
from PIL import Image


class Registry:
    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def register(self, obj=None, force=False):
        def wrapper(item):
            name = item.__name__
            if name in self._obj_map and not force:
                raise KeyError(f'Object "{name}" is already registered in {self._name}')
            self._obj_map[name] = item
            return item

        return wrapper if obj is None else wrapper(obj)

    def get(self, name):
        if name not in self._obj_map:
            raise KeyError(f'Object "{name}" does not exist in {self._name}')
        return self._obj_map[name]

    def registered_names(self):
        return list(self._obj_map.keys())


class AverageMeter:
    def __init__(self, ema=False):
        self.ema = ema
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        if isinstance(val, jt.Var):
            val = float(val.item())
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.avg * 0.9 + self.val * 0.1 if self.ema else self.sum / self.count


class MetricMeter:
    def __init__(self, delimiter=" "):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, input_dict):
        if input_dict is None:
            return
        if not isinstance(input_dict, dict):
            raise TypeError("MetricMeter.update expects a dictionary")
        for key, value in input_dict.items():
            if isinstance(value, jt.Var):
                value = float(value.item())
            self.meters[key].update(value)

    def __str__(self):
        return self.delimiter.join(
            f"{name} {meter.val:.4f} ({meter.avg:.4f})"
            for name, meter in self.meters.items()
        )


class Logger:
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, "w")

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()


def setup_logger(output=None):
    if output is None:
        return
    fpath = output if output.endswith((".txt", ".log")) else osp.join(output, "log.txt")
    if osp.exists(fpath):
        fpath += time.strftime("-%Y-%m-%d-%H-%M-%S")
    sys.stdout = Logger(fpath)


def mkdir_if_missing(dirname):
    if not dirname:
        return
    try:
        os.makedirs(dirname)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise


def check_isfile(fpath):
    result = osp.isfile(fpath)
    if not result:
        warnings.warn(f'No file found at "{fpath}"')
    return result


def read_json(fpath):
    with open(fpath, "r") as file:
        return json.load(file)


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, "w") as file:
        json.dump(obj, file, indent=4, separators=(",", ": "))


def read_image(path):
    return Image.open(path).convert("RGB")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def collect_env_info():
    return "\n".join(
        [
            f"Platform: {platform.platform()}",
            f"Python: {platform.python_version()}",
            f"Jittor: {jt.__version__}",
            f"CUDA enabled: {bool(jt.flags.use_cuda)}",
            f"Pillow: {PIL.__version__}",
        ]
    )


def listdir_nohidden(path, sort=False):
    items = [item for item in os.listdir(path) if not item.startswith(".")]
    if sort:
        items.sort()
    return items


def tolist_if_not(value):
    return value if isinstance(value, list) else [value]


def check_availability(requested, available):
    if requested in available:
        return
    nearest = None
    if available:
        nearest = max(available, key=lambda item: SequenceMatcher(None, requested, item).ratio())
    raise ValueError(f"Expected one of {available}, got [{requested}] (do you mean [{nearest}]?)")


def count_num_param(model=None, params=None):
    values = model.parameters() if model is not None else params
    if values is None:
        raise ValueError("model or params is required")
    return int(sum(np.prod(param.shape) for param in values))


def _load_state(model, state_dict, strict=True):
    current = model.state_dict()
    filtered = OrderedDict()
    unexpected = []
    mismatched = []
    for key, value in state_dict.items():
        key = key[7:] if key.startswith("module.") else key
        if key not in current:
            unexpected.append(key)
            continue
        array = value.numpy() if isinstance(value, jt.Var) else np.asarray(value)
        if tuple(array.shape) != tuple(current[key].shape):
            mismatched.append((key, tuple(array.shape), tuple(current[key].shape)))
            continue
        filtered[key] = array
    missing = sorted(set(current) - set(filtered))
    if strict and (missing or unexpected or mismatched):
        raise ValueError(
            f"State mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}, "
            f"shape={mismatched[:10]}"
        )
    model.load_state_dict(filtered)
    return missing, unexpected, mismatched


def save_checkpoint(state, save_dir, is_best=False, remove_module_from_keys=True, model_name=""):
    mkdir_if_missing(save_dir)
    epoch = state["epoch"]
    if not model_name:
        model_name = f"model.pth.tar-{epoch}"
    fpath = osp.join(save_dir, model_name)
    jt.save(state, fpath)
    print(f"Checkpoint saved to {fpath}")
    with open(osp.join(save_dir, "checkpoint"), "w") as file:
        file.write(osp.basename(fpath) + "\n")
    if is_best:
        best_path = osp.join(save_dir, "model-best.pth.tar")
        shutil.copy(fpath, best_path)
        print(f'Best checkpoint saved to "{best_path}"')


def load_checkpoint(fpath):
    if not fpath or not osp.isfile(fpath):
        raise FileNotFoundError(f'File is not found at "{fpath}"')
    return jt.load(fpath)


def resume_from_checkpoint(fdir, model, optimizer=None, scheduler=None):
    with open(osp.join(fdir, "checkpoint"), "r") as file:
        fpath = osp.join(fdir, file.readline().strip())
    print(f'Loading checkpoint from "{fpath}"')
    checkpoint = load_checkpoint(fpath)
    _load_state(model, checkpoint["state_dict"], strict=True)
    print("Loaded model weights")
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Loaded optimizer")
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded scheduler")
    start_epoch = int(checkpoint["epoch"])
    print(f"Previous epoch: {start_epoch}")
    return start_epoch


def load_pretrained_weights(model, weight_path):
    checkpoint = load_checkpoint(weight_path)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected, mismatched = _load_state(model, state_dict, strict=False)
    loaded = len(model.state_dict()) - len(missing)
    print(
        f"Successfully loaded pretrained weights from {weight_path} "
        f"(loaded={loaded}, missing={len(missing)}, unexpected={len(unexpected)}, "
        f"mismatched={len(mismatched)})"
    )
