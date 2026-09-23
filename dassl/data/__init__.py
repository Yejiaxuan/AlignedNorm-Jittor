import random

import numpy as np
from PIL import Image
from jittor.dataset import Dataset
from tabulate import tabulate

from dassl.utils import read_image
from .datasets import build_dataset
from .transforms import build_transform


class DatasetWrapper(Dataset):
    def __init__(self, cfg, data_source, transform=None, is_train=False):
        super().__init__()
        self.cfg = cfg
        self.data_source = list(data_source)
        self.transform = transform
        self.is_train = is_train
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0
        self.seed = int(cfg.SEED if cfg.SEED >= 0 else 1)
        self.epoch = 0
        self.set_attrs(total_len=len(self.data_source))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._shuffle_rng = np.random.default_rng(self.seed + self.epoch)

    def __getitem__(self, index):
        item = self.data_source[index]
        output = {
            "label": np.int32(item.label),
            "domain": np.int32(item.domain),
            "impath": item.impath,
            "index": np.int32(index),
        }
        image = read_image(item.impath)
        state = random.getstate()
        random.seed(self.seed * 1000003 + self.epoch * 10007 + int(index))
        try:
            if self.transform is None:
                output["img"] = np.asarray(image)
            elif isinstance(self.transform, (list, tuple)):
                for position, operation in enumerate(self.transform):
                    key = "img" if position == 0 else f"img{position + 1}"
                    output[key] = np.asarray(operation(image), dtype=np.float32)
            else:
                images = [np.asarray(self.transform(image), dtype=np.float32) for _ in range(self.k_tfm)]
                output["img"] = images[0] if len(images) == 1 else images
        finally:
            random.setstate(state)
        if self.return_img0:
            resized = image.resize(tuple(self.cfg.INPUT.SIZE), Image.BICUBIC)
            array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
            output["img0"] = array
        return output


def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None,
):
    if sampler_type not in ["RandomSampler", "SequentialSampler"]:
        raise NotImplementedError(f"Sampler {sampler_type} is not needed by AlignedNorm")
    wrapper = (dataset_wrapper or DatasetWrapper)(
        cfg, data_source, transform=tfm, is_train=is_train
    )
    # Jittor workers return whichever batch finishes first, so seeded training
    # uses the main process to preserve sampler order across resume.
    num_workers = 0 if is_train and cfg.SEED >= 0 else int(cfg.DATALOADER.NUM_WORKERS)
    return wrapper.set_attrs(
        batch_size=int(batch_size),
        shuffle=sampler_type == "RandomSampler",
        drop_last=is_train and len(data_source) >= batch_size,
        num_workers=num_workers,
    )


class DataManager:
    def __init__(self, cfg, custom_tfm_train=None, custom_tfm_test=None, dataset_wrapper=None):
        dataset = build_dataset(cfg)
        tfm_train = custom_tfm_train or build_transform(cfg, is_train=True)
        tfm_test = custom_tfm_test or build_transform(cfg, is_train=False)
        self.train_loader_x = build_data_loader(
            cfg,
            cfg.DATALOADER.TRAIN_X.SAMPLER,
            dataset.train_x,
            cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            cfg.DATALOADER.TRAIN_X.N_INS,
            tfm_train,
            True,
            dataset_wrapper,
        )
        self.train_loader_u = None
        self.val_loader = None
        if dataset.val:
            self.val_loader = build_data_loader(
                cfg,
                cfg.DATALOADER.TEST.SAMPLER,
                dataset.val,
                cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper,
            )
        self.test_loader = build_data_loader(
            cfg,
            cfg.DATALOADER.TEST.SAMPLER,
            dataset.test,
            cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper,
        )
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self.lab2cname = dataset.lab2cname
        if cfg.VERBOSE:
            rows = [
                ["Dataset", cfg.DATASET.NAME],
                ["# classes", f"{self.num_classes:,}"],
                ["# train_x", f"{len(dataset.train_x):,}"],
            ]
            if dataset.val:
                rows.append(["# val", f"{len(dataset.val):,}"])
            rows.append(["# test", f"{len(dataset.test):,}"])
            print(tabulate(rows))


__all__ = ["DataManager", "DatasetWrapper", "build_data_loader"]
