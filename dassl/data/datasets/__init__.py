from dassl.utils import Registry, check_availability

from .base_dataset import Datum, DatasetBase


DATASET_REGISTRY = Registry("DATASET")


def build_dataset(cfg):
    available = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, available)
    if cfg.VERBOSE:
        print(f"Loading dataset: {cfg.DATASET.NAME}")
    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg)


__all__ = ["DATASET_REGISTRY", "Datum", "DatasetBase", "build_dataset"]
