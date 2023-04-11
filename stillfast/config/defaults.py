from pickle import NONE
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

_C.CHECKPOINT_FILE_PATH = ""
_C.CHECKPOINT_LOAD_MODEL_HEAD = True
_C.ENABLE_LOGGING = True
_C.EXPERIMENT_NAME = "default"
_C.TASK = "short_term_anticipation"
_C.WANDB_RUN = ""

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The video sampling rate of the input clip.


#_C.DATA.FAST_STILL_RATIO = 0.20
#_C.DATA.CROP_RATIO = 0.875
#_C.DATA.HORIZONTAL_FLIP = True

#_C.DATA.CROP_TYPE = "SCR" #STD or SCR (Ma & Damen. Hand-Object Interaction Reasoning)

_C.DATA.FAST = CfgNode()
_C.DATA.FAST.MEAN = [0.45, 0.45, 0.45]
_C.DATA.FAST.STD = [0.225, 0.225, 0.225]
_C.DATA.FAST.NUM_FRAMES = 16
_C.DATA.FAST.SAMPLING_RATE = 1

_C.DATA.STILL = CfgNode()
_C.DATA.STILL.MIN_SIZE = [800]
_C.DATA.STILL.MAX_SIZE = 1333
_C.DATA.STILL.MEAN = [0.485, 0.456, 0.406]
_C.DATA.STILL.STD = [0.229, 0.224, 0.225]
_C.DATA.STILL.FAST_TO_STILL_SIZE_RATIO = 0.25

# -----------------------------------------------------------------------------
# Ego4D-STA options
# -----------------------------------------------------------------------------
_C.EGO4D_STA = CfgNode()

# Path to still frames
_C.EGO4D_STA.STILL_FRAMES_PATH = "" 

# Path to fast LMDB
_C.EGO4D_STA.FAST_LMDB_PATH = "" 

# Annotations path
_C.EGO4D_STA.ANNOTATION_DIR = "" 

# Filenames of training samples list files.
_C.EGO4D_STA.TRAIN_LISTS = []

# Filenames of test samples list files.
_C.EGO4D_STA.VAL_LISTS = []

# Filenames of test samples list files.
_C.EGO4D_STA.TEST_LISTS = []

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = False

# Dataset.
_C.TRAIN.DATASET = "Ego4dShortTermAnticipationStill"

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 16

_C.TRAIN.AUGMENTATIONS = CfgNode()
_C.TRAIN.AUGMENTATIONS.RANDOM_HORIZONTAL_FLIP = False

_C.TRAIN.GROUP_BATCH_SAMPLER = False

_C.TRAIN.WEIGHTED_SAMPLER = False

# ---------------------------------------------------------------------------- #
# Testing options.
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()
_C.TEST.ENABLE = False

# Dataset.
_C.TEST.DATASET = "Ego4dShortTermAnticipationStill"

_C.TEST.BATCH_SIZE = 16
_C.TEST.OUTPUT_JSON = None
_C.TEST.GROUP_BATCH_SAMPLER = False

# -----------------------------------------------------------------------------
# Validation options.
# -----------------------------------------------------------------------------
_C.VAL = CfgNode()
_C.VAL.ENABLE = False
_C.VAL.DATASET = "Ego4dShortTermAnticipationStill"
_C.VAL.BATCH_SIZE = 16
_C.VAL.OUTPUT_JSON = None
_C.VAL.GROUP_BATCH_SAMPLER = False

# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True
 
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of devices (most likely GPUs) to use (applies to both training and testing).
_C.NUM_DEVICES = 1

_C.FAST_DEV_RUN = False
_C.OUTPUT_DIR = "./output"
_C.NUM_SHARDS = 1
_C.SAVE_TOP_K = 1
_C.AVERAGE_TOP_K_CHECKPOINTS = 1


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Learning rate policy (see optimizers/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine_warmup"

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Epoch milestones to decrease the learning rate.
_C.SOLVER.MILESTONES = []

# # Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# # Momentum.
_C.SOLVER.MOMENTUM = 0.9

# # Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# # Nesterov momentum.
_C.SOLVER.NESTEROV = True

# # L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Gradually warm up the SOLVER.BASE_LR over this number of steps.
_C.SOLVER.WARMUP_STEPS = 1000

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Which PyTorch Lightning acceleration strategy to use
_C.SOLVER.STRATEGY = "ddp"

# Which PyTorch Lightning accelerator to use
_C.SOLVER.ACCELERATOR = "gpu"

# Whether to use CUDA Benchmark mode
_C.SOLVER.BENCHMARK = False

# If samplers should be replaced when using ddp
_C.SOLVER.REPLACE_SAMPLER_DDP = False

# Training precision
_C.SOLVER.PRECISION = 32

# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# BN epsilon.
_C.BN.EPSILON = 1e-5

# BN momentum.
_C.BN.MOMENTUM = 0.1

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SplitBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm3d, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized.
_C.BN.NUM_SYNC_DEVICES = 1

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CfgNode()
_C.MODEL.NAME = "FasterRCNN"
_C.MODEL.BRANCH = "Still"
_C.MODEL.STILL = CfgNode()
_C.MODEL.STILL.BACKBONE = CfgNode()
_C.MODEL.STILL.BACKBONE.NAME = "resnet50"
_C.MODEL.STILL.BACKBONE.PRETRAINED = True
_C.MODEL.STILL.BACKBONE.TRAINABLE_LAYERS = 3
_C.MODEL.STILL.PRETRAINED = True
_C.MODEL.STILL.REPLACE_HEAD = True

_C.MODEL.STILL.RPN = CfgNode()
_C.MODEL.STILL.RPN.ANCHOR_GENERATOR = None
_C.MODEL.STILL.RPN.HEAD = None
_C.MODEL.STILL.RPN.POST_NMS_TOP_N_TEST = 1000
_C.MODEL.STILL.RPN.POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.STILL.RPN.PRE_NMS_TOP_N_TEST = 1000
_C.MODEL.STILL.RPN.PRE_NMS_TOP_N_TRAIN = 2000 

_C.MODEL.STILL.RPN.NMS_THRESH = 0.7
_C.MODEL.STILL.RPN.FG_IOU_THRESH = 0.7
_C.MODEL.STILL.RPN.BG_IOU_THRESH = 0.3
_C.MODEL.STILL.RPN.BATCH_SIZE_PER_IMAGE = 256
_C.MODEL.STILL.RPN.POSITIVE_FRACTION = 0.5
_C.MODEL.STILL.RPN.SCORE_THRESH = 0.0

_C.MODEL.FAST = CfgNode()
_C.MODEL.FAST.BACKBONE = CfgNode()
_C.MODEL.FAST.BACKBONE.NAME = "x3d_m"
_C.MODEL.FAST.BACKBONE.PRETRAINED = True
_C.MODEL.FAST.BACKBONE.TEMPORAL_CAUSAL_CONV3D = False

_C.MODEL.STILLFAST = CfgNode()
_C.MODEL.STILLFAST.FUSION = CfgNode()
_C.MODEL.STILLFAST.FUSION.FUSION_BLOCK = 'convolutional'
_C.MODEL.STILLFAST.FUSION.CONVOLUTIONAL_FUSION_BLOCK = CfgNode()
_C.MODEL.STILLFAST.FUSION.CONVOLUTIONAL_FUSION_BLOCK.POOLING = 'mean'
_C.MODEL.STILLFAST.FUSION.CONVOLUTIONAL_FUSION_BLOCK.POOLING_FRAMES = 16
_C.MODEL.STILLFAST.FUSION.CONVOLUTIONAL_FUSION_BLOCK.CONV_BLOCK_ARCHITECTURE = 'simple_convolution' #resnet_block
_C.MODEL.STILLFAST.FUSION.CONVOLUTIONAL_FUSION_BLOCK.POST_UP_CONV_BLOCK = False
_C.MODEL.STILLFAST.FUSION.CONVOLUTIONAL_FUSION_BLOCK.POST_SUM_CONV_BLOCK = True
_C.MODEL.STILLFAST.FUSION.CONVOLUTIONAL_FUSION_BLOCK.GATING_BLOCK = 'None' # channel_gating
_C.MODEL.STILLFAST.FUSION.CONVOLUTIONAL_FUSION_BLOCK.TEMPORAL_NONLOCAL_POOLING = CfgNode()
_C.MODEL.STILLFAST.FUSION.CONVOLUTIONAL_FUSION_BLOCK.TEMPORAL_NONLOCAL_POOLING.MAX_HEIGHT_BEFORE_POOLING = 16
_C.MODEL.STILLFAST.FUSION.CONVOLUTIONAL_FUSION_BLOCK.TEMPORAL_NONLOCAL_POOLING.INTER_CHANNELS = 'half'

_C.MODEL.STILLFAST.FUSION.NONLOCAL_FUSION_BLOCK = CfgNode()
_C.MODEL.STILLFAST.FUSION.NONLOCAL_FUSION_BLOCK.MAX_HEIGHT_BEFORE_SCALING_2D = 128
_C.MODEL.STILLFAST.FUSION.NONLOCAL_FUSION_BLOCK.MAX_HEIGHT_BEFORE_POOLING_3D = 16
_C.MODEL.STILLFAST.FUSION.NONLOCAL_FUSION_BLOCK.SCALING_2D_MODE = 'nearest'
_C.MODEL.STILLFAST.FUSION.NONLOCAL_FUSION_BLOCK.INTER_CHANNELS = 'half'
_C.MODEL.STILLFAST.FUSION.NONLOCAL_FUSION_BLOCK.POST_SUM_CONV_BLOCK = True

_C.MODEL.STILLFAST.FUSION.PRE_PYRAMID_FUSION = False
_C.MODEL.STILLFAST.FUSION.POST_PYRAMID_FUSION = True
_C.MODEL.STILLFAST.FUSION.LATERAL_CONNECTIONS = False

_C.MODEL.STILLFAST.ROI_HEADS = CfgNode()
_C.MODEL.STILLFAST.ROI_HEADS.VERSION = 'v1' #v2
_C.MODEL.STILLFAST.ROI_HEADS.V2_OPTIONS = CfgNode()
_C.MODEL.STILLFAST.ROI_HEADS.V2_OPTIONS.VERB_TOPK = 1
_C.MODEL.STILLFAST.ROI_HEADS.V2_OPTIONS.FUSION = 'sum' #concat, concat_residual


_C.MODEL.NOUN_CLASSES = 87
_C.MODEL.VERB_CLASSES = 74
_C.MODEL.LOSS = CfgNode()
_C.MODEL.LOSS.WEIGHTS = CfgNode()
_C.MODEL.LOSS.WEIGHTS.NOUN = 1.0
_C.MODEL.LOSS.WEIGHTS.VERB = 1.0
_C.MODEL.LOSS.WEIGHTS.TTC = 10.0
_C.MODEL.LOSS.WEIGHTS.NAO = 1
_C.MODEL.LOSS.NOUN = 'cross_entropy'
_C.MODEL.LOSS.VERB = 'cross_entropy'
_C.MODEL.LOSS.TTC = 'smooth_l1'
_C.MODEL.TTC_PREDICTOR = 'regressor'


_C.MODEL.STILL.BOX = CfgNode()
_C.MODEL.STILL.BOX.SCORE_THRESH = 0.05
_C.MODEL.STILL.BOX.NMS_THRESH = 0.5
_C.MODEL.STILL.BOX.DETECTIONS_PER_IMG = 100
_C.MODEL.STILL.BOX.FG_IOU_THRESH = 0.5
_C.MODEL.STILL.BOX.BG_IOU_THRESH = 0.5
_C.MODEL.STILL.BOX.BATCH_SIZE_PER_IMAGE = 256
_C.MODEL.STILL.BOX.POSITIVE_FRACTION = 0.25
_C.MODEL.STILL.BOX.REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
_C.MODEL.STILL.BOX.PREDICTOR_REPRESENTATION_SIZE = 1024
_C.MODEL.STILL.BOX.POOLER_SAMPLING_RATIO = 2 #0

def _assert_and_infer_cfg(cfg):
    return cfg

def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())