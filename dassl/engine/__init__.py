import datetime
import os
import os.path as osp
import random
import time
from collections import OrderedDict

import jittor as jt
import numpy as np
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tqdm import tqdm

from dassl.data import DataManager
from dassl.evaluation import build_evaluator
from dassl.utils import (
    AverageMeter,
    MetricMeter,
    Registry,
    _load_state,
    check_availability,
    load_checkpoint,
    load_pretrained_weights,
    mkdir_if_missing,
    save_checkpoint,
    tolist_if_not,
)


TRAINER_REGISTRY = Registry("TRAINER")


def build_trainer(cfg):
    available = TRAINER_REGISTRY.registered_names()
    check_availability(cfg.TRAINER.NAME, available)
    if cfg.VERBOSE:
        print(f"Loading trainer: {cfg.TRAINER.NAME}")
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)


class _SummaryWriter:
    def __init__(self, log_dir):
        self.writer = EventFileWriter(log_dir)

    def add_scalar(self, tag, value, step):
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])
        self.writer.add_event(Event(wall_time=time.time(), step=int(step or 0), summary=summary))

    def close(self):
        self.writer.close()


def _checkpoint_path(directory, epoch=None):
    if epoch is not None:
        return osp.join(directory, f"model.pth.tar-{int(epoch)}")
    pointer = osp.join(directory, "checkpoint")
    if osp.isfile(pointer):
        with open(pointer, "r") as file:
            return osp.join(directory, file.readline().strip())
    best = osp.join(directory, "model-best.pth.tar")
    if osp.isfile(best):
        return best
    raise FileNotFoundError(f"No checkpoint found in {directory}")


def _optimizer_state_dict(optimizer):
    state = dict(optimizer.state_dict())
    defaults = dict(state["defaults"])
    defaults["param_groups"] = [
        {key: value for key, value in group.items() if key not in {"params", "grads"}}
        for group in defaults.get("param_groups", ())
    ]
    state["defaults"] = defaults
    return state


class TrainerBase:
    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        self.global_step = 0
        self.resume_batch = 0

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if name in self._models:
            raise KeyError(f"Duplicate model name: {name}")
        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        available = list(self._models)
        if names is None:
            return available
        names = tolist_if_not(names)
        for name in names:
            if name not in available:
                raise KeyError(name)
        return names

    def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=""):
        for name in self.get_model_names():
            state = {
                "state_dict": self._models[name].state_dict(to="numpy"),
                "epoch": int(epoch) + 1,
                "next_epoch": int(getattr(self, "_next_epoch", int(epoch) + 1)),
                "next_batch": int(getattr(self, "_next_batch", 0)),
                "global_step": int(self.global_step),
                "optimizer": _optimizer_state_dict(self._optims[name]) if self._optims[name] else None,
                "scheduler": self._scheds[name].state_dict() if self._scheds[name] else None,
                "val_result": val_result,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "jittor_seed": int(jt.get_seed()),
            }
            save_checkpoint(
                state,
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def resume_model_if_exist(self, directory):
        paths = []
        for name in self.get_model_names():
            model_dir = osp.join(directory, name)
            if not osp.isdir(model_dir) or not osp.isfile(osp.join(model_dir, "checkpoint")):
                print("No checkpoint found, train from scratch")
                return 0
            paths.append((name, _checkpoint_path(model_dir)))
        print(f"Found checkpoint at {directory} (will resume training)")
        metadata = None
        for name, path in paths:
            print(f'Loading checkpoint from "{path}"')
            checkpoint = load_checkpoint(path)
            _load_state(self._models[name], checkpoint["state_dict"], strict=True)
            print("Loaded model weights")
            if self._optims[name] is not None and checkpoint.get("optimizer") is not None:
                self._optims[name].load_state_dict(checkpoint["optimizer"])
                print("Loaded optimizer")
            if self._scheds[name] is not None and checkpoint.get("scheduler") is not None:
                self._scheds[name].load_state_dict(checkpoint["scheduler"])
                print("Loaded scheduler")
            metadata = checkpoint
        self.global_step = int(metadata.get("global_step", 0))
        self.resume_batch = int(metadata.get("next_batch", 0))
        start_epoch = int(metadata.get("next_epoch", metadata["epoch"]))
        if "python_random_state" in metadata:
            random.setstate(metadata["python_random_state"])
        if "numpy_random_state" in metadata:
            np.random.set_state(metadata["numpy_random_state"])
        if "jittor_seed" in metadata:
            jt.set_global_seed(int(metadata["jittor_seed"]))
        print(f"Previous epoch: {metadata['epoch']}")
        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
        for name in self.get_model_names():
            path = _checkpoint_path(osp.join(directory, name), epoch)
            checkpoint = load_checkpoint(path)
            ignored = tuple(getattr(self, "checkpoint_ignored_keys", ()))
            state_dict = {
                key: value
                for key, value in checkpoint["state_dict"].items()
                if not any(token in key for token in ignored)
            }
            missing, unexpected, mismatched = _load_state(
                self._models[name], state_dict, strict=False
            )
            missing = [
                key
                for key in missing
                if not any(token in key for token in ignored)
            ]
            if missing or unexpected or mismatched:
                raise ValueError(
                    f"State mismatch: missing={missing[:10]}, "
                    f"unexpected={unexpected[:10]}, shape={mismatched[:10]}"
                )
            value = checkpoint.get("val_result")
            value_text = "None" if value is None else f"{value:.1f}"
            print(f"Load {path} to {name} (epoch={checkpoint['epoch']}, val_result={value_text})")

    def set_model_mode(self, mode="train", names=None):
        for name in self.get_model_names(names):
            self._models[name].train() if mode == "train" else self._models[name].eval()

    def update_lr(self, names=None):
        for name in self.get_model_names(names):
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def init_writer(self, log_dir):
        if self._writer is None:
            print(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = _SummaryWriter(log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is not None:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self, start_epoch, max_epoch):
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def before_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def after_train(self):
        pass

    def model_backward_and_update(self, loss, names=None):
        names = self.get_model_names(names)
        if len(names) != 1:
            raise NotImplementedError("AlignedNorm uses one optimizer")
        optimizer = self._optims[names[0]]
        if not bool(np.isfinite(float(loss.item()))):
            raise FloatingPointError("Loss is infinite or NaN")
        optimizer.step(loss)


class SimpleTrainer(TrainerBase):
    def __init__(self, cfg):
        super().__init__()
        self.check_cfg(cfg)
        self.device = "cuda" if cfg.USE_CUDA else "cpu"
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = -np.inf

    def check_cfg(self, cfg):
        pass

    def build_data_loader(self):
        dm = DataManager(self.cfg)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname
        self.dm = dm

    def build_model(self):
        raise NotImplementedError

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def before_train(self):
        directory = self.cfg.RESUME or self.cfg.OUTPUT_DIR
        self.start_epoch = self.resume_model_if_exist(directory)
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)
        self.time_start = time.time()

    def after_train(self):
        print("Finish training")
        if not self.cfg.TEST.NO_TEST:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test()
        elapsed = str(datetime.timedelta(seconds=round(time.time() - self.time_start)))
        print(f"Elapsed: {elapsed}")
        self.close_writer()

    def after_epoch(self):
        self._next_epoch = self.epoch + 1
        self._next_batch = 0
        last_epoch = self._next_epoch == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        checkpoint_freq = self.cfg.TRAIN.CHECKPOINT_FREQ
        meet_frequency = checkpoint_freq > 0 and self._next_epoch % checkpoint_freq == 0
        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            result = self.test(split="val")
            if result > self.best_result:
                self.best_result = result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    is_best=True,
                    val_result=result,
                    model_name="model-best.pth.tar",
                )
        if meet_frequency or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()
        split = split or self.cfg.TEST.SPLIT
        if split == "val" and self.val_loader is not None:
            loader = self.val_loader
        else:
            split = "test"
            loader = self.test_loader
        print(f"Evaluate on the *{split}* set")
        with jt.no_grad():
            for batch in tqdm(loader):
                input_image, label = self.parse_batch_test(batch)
                self.evaluator.process(self.model_inference(input_image), label)
        results = self.evaluator.evaluate()
        for key, value in results.items():
            self.write_scalar(f"{split}/{key}", value, self.epoch)
        return list(results.values())[0]

    def model_inference(self, input_image):
        return self.model(input_image)

    def parse_batch_test(self, batch):
        return batch["img"], batch["label"]

    def get_current_lr(self, names=None):
        name = self.get_model_names(names)[0]
        optimizer = self._optims[name]
        return float(optimizer.param_groups[0].get("lr", optimizer.lr))


class TrainerX(SimpleTrainer):
    def run_epoch(self):
        self.set_model_mode("train")
        if hasattr(self.train_loader_x, "set_epoch"):
            self.train_loader_x.set_epoch(self.epoch)
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)
        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            if self.epoch == self.start_epoch and self.batch_idx < self.resume_batch:
                continue
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            self.global_step += 1
            self._next_epoch = self.epoch
            self._next_batch = self.batch_idx + 1
            batch_time.update(time.time() - end)
            losses.update(loss_summary)
            meet_frequency = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            if meet_frequency or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                remaining = self.num_batches - self.batch_idx - 1
                remaining += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta = str(datetime.timedelta(seconds=int(batch_time.avg * remaining)))
                info = [
                    f"epoch [{self.epoch + 1}/{self.max_epoch}]",
                    f"batch [{self.batch_idx + 1}/{self.num_batches}]",
                    f"time {batch_time.val:.3f} ({batch_time.avg:.3f})",
                    f"data {data_time.val:.3f} ({data_time.avg:.3f})",
                    str(losses),
                    f"lr {self.get_current_lr():.4e}",
                    f"eta {eta}",
                ]
                print(" ".join(info))
            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar(f"train/{name}", meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)
            end = time.time()
        self.resume_batch = 0

    def parse_batch_train(self, batch):
        return batch["img"], batch["label"], batch["domain"]


TrainerXU = TrainerX


__all__ = [
    "TRAINER_REGISTRY",
    "build_trainer",
    "TrainerBase",
    "SimpleTrainer",
    "TrainerX",
    "TrainerXU",
]
