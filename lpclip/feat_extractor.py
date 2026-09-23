import os, argparse
import numpy as np
import jittor as jt
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.oxford_pets import OxfordPets
from datasets.oxford_flowers import OxfordFlowers
from datasets.fgvc_aircraft import FGVCAircraft
from datasets.dtd import DescribableTextures
from datasets.eurosat import EuroSAT
from datasets.stanford_cars import StanfordCars
from datasets.food101 import Food101
from datasets.sun397 import SUN397
from datasets.caltech101 import Caltech101
from datasets.ucf101 import UCF101
from datasets.imagenet import ImageNet
from datasets.imagenetv2 import ImageNetV2
from datasets.imagenet_sketch import ImageNetSketch
from datasets.imagenet_a import ImageNetA
from datasets.imagenet_r import ImageNetR

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.data.transforms import build_transform
from dassl.data import DatasetWrapper

import clip

# import pdb; pdb.set_trace()


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.seed >= 0:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.OURS = CN()
    cfg.TRAINER.OURS.N_CTX = 10  # number of context vectors
    cfg.TRAINER.OURS.CSC = False  # class-specific context
    cfg.TRAINER.OURS.CTX_INIT = ""  # initialize context vectors with given words
    cfg.TRAINER.OURS.WEIGHT_U = 0.1  # weight for the unsupervised loss
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 64
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.INPUT.SIZE = (224, 224)
    cfg.INPUT.INTERPOLATION = "bicubic"
    cfg.INPUT.PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
    cfg.INPUT.PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
    cfg.INPUT.TRANSFORMS = ("center_crop", "normalize")
    cfg.MODEL.BACKBONE.NAME = "RN50"


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    jt.flags.use_cuda = 1 if cfg.USE_CUDA else 0

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    ######################################
    #   Setup DataLoader
    ######################################
    dataset = eval(cfg.DATASET.NAME)(cfg)

    if args.split == "train":
        dataset_input = dataset.train_x
    elif args.split == "val":
        dataset_input = dataset.val
    else:
        dataset_input = dataset.test

    tfm_train = build_transform(cfg, is_train=False)
    data_loader = DatasetWrapper(
        cfg, dataset_input, transform=tfm_train, is_train=False
    ).set_attrs(
        batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=False,
    )

    ########################################
    #   Setup Network
    ########################################
    if not args.clip_weights:
        raise ValueError(
            "Jittor RN50 weights are required; pass --clip-weights or set JITTOR_RN50_WEIGHTS"
        )
    device = "cuda" if cfg.USE_CUDA else "cpu"
    clip_model, _ = clip.load(args.clip_weights, device, jit=False)
    clip_model.eval()
    ###################################################################################################################
    # Start Feature Extractor
    feature_list = []
    label_list = []
    with jt.no_grad():
        for batch in data_loader:
            feature_list.append(clip_model.visual(batch["img"]).numpy())
            label_list.append(batch["label"].numpy())
    save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME)
    os.makedirs(save_dir, exist_ok=True)
    save_filename = f"{args.split}"
    np.savez(
        os.path.join(save_dir, save_filename),
        feature_list=np.concatenate(feature_list, axis=0),
        label_list=np.concatenate(label_list, axis=0),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--num-shot", type=int, default=1, help="number of shots")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], help="which split")
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--clip-weights",
        type=str,
        default=os.environ.get("JITTOR_RN50_WEIGHTS", ""),
        help="converted Jittor RN50 checkpoint",
    )
    args = parser.parse_args()
    main(args)
