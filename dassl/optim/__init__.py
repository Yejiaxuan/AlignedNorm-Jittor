import math

import jittor as jt


def build_optimizer(model, optim_cfg, param_groups=None):
    if optim_cfg.STAGED_LR:
        raise NotImplementedError("Staged LR is not used by AlignedNorm")
    params = (
        param_groups
        if param_groups is not None
        else [param for param in model.parameters() if not param.is_stop_grad()]
    )
    name = optim_cfg.NAME.lower()
    common = {
        "lr": float(optim_cfg.LR),
        "weight_decay": float(optim_cfg.WEIGHT_DECAY),
    }
    if name == "adamw":
        return jt.optim.AdamW(
            params,
            betas=(float(optim_cfg.ADAM_BETA1), float(optim_cfg.ADAM_BETA2)),
            **common,
        )
    if name == "adam":
        return jt.optim.Adam(
            params,
            betas=(float(optim_cfg.ADAM_BETA1), float(optim_cfg.ADAM_BETA2)),
            **common,
        )
    if name == "sgd":
        return jt.optim.SGD(
            params,
            momentum=float(optim_cfg.MOMENTUM),
            **common,
        )
    raise NotImplementedError(f"Optimizer {optim_cfg.NAME} is not used by AlignedNorm")


class LRScheduler:
    def __init__(self, optimizer, cfg):
        self.optimizer = optimizer
        self.name = cfg.LR_SCHEDULER.lower()
        self.base_lr = float(cfg.LR)
        self.max_epoch = int(cfg.MAX_EPOCH)
        self.stepsize = tuple(cfg.STEPSIZE)
        self.gamma = float(cfg.GAMMA)
        self.warmup_epoch = max(int(cfg.WARMUP_EPOCH), 0)
        self.warmup_type = cfg.WARMUP_TYPE
        self.warmup_cons_lr = float(cfg.WARMUP_CONS_LR)
        self.warmup_min_lr = float(cfg.WARMUP_MIN_LR)
        self.last_epoch = 0
        if self.warmup_epoch > 0:
            initial = self.warmup_cons_lr if self.warmup_type == "constant" else self.warmup_min_lr
            self._set_lr(initial)

    def _set_lr(self, value):
        value = float(value)
        self.optimizer.lr = value
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def _successor_lr(self, epoch):
        if self.name == "cosine":
            return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / self.max_epoch))
        if self.name == "single_step":
            step = self.stepsize[-1]
            step = self.max_epoch if step <= 0 else step
            return self.base_lr * self.gamma ** (epoch // step)
        if self.name == "multi_step":
            return self.base_lr * self.gamma ** sum(epoch >= step for step in self.stepsize)
        raise ValueError(f"Unknown scheduler: {self.name}")

    def step(self):
        self.last_epoch += 1
        if self.warmup_epoch and self.last_epoch <= self.warmup_epoch:
            if self.last_epoch == self.warmup_epoch:
                lr = self.base_lr
            elif self.warmup_type == "constant":
                lr = self.warmup_cons_lr
            else:
                ratio = self.last_epoch / self.warmup_epoch
                lr = self.warmup_min_lr + ratio * (self.base_lr - self.warmup_min_lr)
        else:
            successor_epoch = self.last_epoch - self.warmup_epoch
            lr = self._successor_lr(successor_epoch)
        self._set_lr(lr)

    def state_dict(self):
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state):
        self.last_epoch = int(state.get("last_epoch", 0))
        if self.last_epoch == 0 and self.warmup_epoch:
            lr = self.warmup_cons_lr if self.warmup_type == "constant" else self.warmup_min_lr
        elif self.warmup_epoch and self.last_epoch <= self.warmup_epoch:
            lr = self.base_lr
        else:
            lr = self._successor_lr(self.last_epoch - self.warmup_epoch)
        self._set_lr(lr)


def build_lr_scheduler(optimizer, optim_cfg):
    return LRScheduler(optimizer, optim_cfg)


__all__ = ["build_optimizer", "build_lr_scheduler"]
