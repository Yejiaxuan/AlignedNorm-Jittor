from yacs.config import CfgNode as CN


def get_cfg_default():
    cfg = CN()
    cfg.VERSION = 1
    cfg.OUTPUT_DIR = "./output"
    cfg.RESUME = ""
    cfg.SEED = -1
    cfg.USE_CUDA = True
    cfg.VERBOSE = True

    cfg.INPUT = CN()
    cfg.INPUT.SIZE = (224, 224)
    cfg.INPUT.INTERPOLATION = "bilinear"
    cfg.INPUT.TRANSFORMS = ()
    cfg.INPUT.NO_TRANSFORM = False
    cfg.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
    cfg.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
    cfg.INPUT.CROP_PADDING = 4
    cfg.INPUT.RRCROP_SCALE = (0.08, 1.0)
    cfg.INPUT.CUTOUT_N = 1
    cfg.INPUT.CUTOUT_LEN = 16
    cfg.INPUT.GN_MEAN = 0.0
    cfg.INPUT.GN_STD = 0.15
    cfg.INPUT.RANDAUGMENT_N = 2
    cfg.INPUT.RANDAUGMENT_M = 10
    cfg.INPUT.COLORJITTER_B = 0.4
    cfg.INPUT.COLORJITTER_C = 0.4
    cfg.INPUT.COLORJITTER_S = 0.4
    cfg.INPUT.COLORJITTER_H = 0.1
    cfg.INPUT.RGS_P = 0.2
    cfg.INPUT.GB_P = 0.5
    cfg.INPUT.GB_K = 21

    cfg.DATASET = CN()
    cfg.DATASET.ROOT = ""
    cfg.DATASET.NAME = ""
    cfg.DATASET.SOURCE_DOMAINS = ()
    cfg.DATASET.TARGET_DOMAINS = ()
    cfg.DATASET.NUM_LABELED = -1
    cfg.DATASET.NUM_SHOTS = -1
    cfg.DATASET.VAL_PERCENT = 0.1
    cfg.DATASET.STL10_FOLD = -1
    cfg.DATASET.CIFAR_C_TYPE = ""
    cfg.DATASET.CIFAR_C_LEVEL = 1
    cfg.DATASET.ALL_AS_UNLABELED = False

    cfg.DATALOADER = CN()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.K_TRANSFORMS = 1
    cfg.DATALOADER.RETURN_IMG0 = False
    cfg.DATALOADER.TRAIN_X = CN()
    cfg.DATALOADER.TRAIN_X.SAMPLER = "RandomSampler"
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 32
    cfg.DATALOADER.TRAIN_X.N_DOMAIN = 0
    cfg.DATALOADER.TRAIN_X.N_INS = 16
    cfg.DATALOADER.TRAIN_U = CN()
    cfg.DATALOADER.TRAIN_U.SAME_AS_X = True
    cfg.DATALOADER.TRAIN_U.SAMPLER = "RandomSampler"
    cfg.DATALOADER.TRAIN_U.BATCH_SIZE = 32
    cfg.DATALOADER.TRAIN_U.N_DOMAIN = 0
    cfg.DATALOADER.TRAIN_U.N_INS = 16
    cfg.DATALOADER.TEST = CN()
    cfg.DATALOADER.TEST.SAMPLER = "SequentialSampler"
    cfg.DATALOADER.TEST.BATCH_SIZE = 32

    cfg.MODEL = CN()
    cfg.MODEL.INIT_WEIGHTS = ""
    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = ""
    cfg.MODEL.BACKBONE.PRETRAINED = True
    cfg.MODEL.HEAD = CN()
    cfg.MODEL.HEAD.NAME = ""
    cfg.MODEL.HEAD.HIDDEN_LAYERS = ()
    cfg.MODEL.HEAD.ACTIVATION = "relu"
    cfg.MODEL.HEAD.BN = True
    cfg.MODEL.HEAD.DROPOUT = 0.0

    cfg.OPTIM = CN()
    cfg.OPTIM.NAME = "adam"
    cfg.OPTIM.LR = 0.0003
    cfg.OPTIM.WEIGHT_DECAY = 5e-4
    cfg.OPTIM.MOMENTUM = 0.9
    cfg.OPTIM.SGD_DAMPNING = 0
    cfg.OPTIM.SGD_NESTEROV = False
    cfg.OPTIM.RMSPROP_ALPHA = 0.99
    cfg.OPTIM.ADAM_BETA1 = 0.9
    cfg.OPTIM.ADAM_BETA2 = 0.999
    cfg.OPTIM.STAGED_LR = False
    cfg.OPTIM.NEW_LAYERS = ()
    cfg.OPTIM.BASE_LR_MULT = 0.1
    cfg.OPTIM.LR_SCHEDULER = "single_step"
    cfg.OPTIM.STEPSIZE = (-1,)
    cfg.OPTIM.GAMMA = 0.1
    cfg.OPTIM.MAX_EPOCH = 10
    cfg.OPTIM.WARMUP_EPOCH = -1
    cfg.OPTIM.WARMUP_TYPE = "linear"
    cfg.OPTIM.WARMUP_CONS_LR = 1e-5
    cfg.OPTIM.WARMUP_MIN_LR = 1e-5
    cfg.OPTIM.WARMUP_RECOUNT = True

    cfg.TRAIN = CN()
    cfg.TRAIN.CHECKPOINT_FREQ = 0
    cfg.TRAIN.PRINT_FREQ = 10
    cfg.TRAIN.COUNT_ITER = "train_x"

    cfg.TEST = CN()
    cfg.TEST.EVALUATOR = "Classification"
    cfg.TEST.PER_CLASS_RESULT = False
    cfg.TEST.COMPUTE_CMAT = False
    cfg.TEST.NO_TEST = False
    cfg.TEST.SPLIT = "test"
    cfg.TEST.FINAL_MODEL = "last_step"

    cfg.TRAINER = CN()
    cfg.TRAINER.NAME = ""
    return cfg


def clean_cfg(cfg, trainer):
    for key in list(cfg.TRAINER.keys()):
        if key not in ["NAME", trainer.upper()]:
            cfg.TRAINER.pop(key, None)
