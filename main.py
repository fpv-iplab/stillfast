import os
from re import T
from stillfast.tasks.detection import SimpleDetectionTask
from stillfast.tasks.sta import STATask
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import Trainer
from stillfast.config.defaults import get_cfg
from stillfast.logging import StillFastLogger
import argparse
import sys
from glob import glob
from os.path import join
import wandb
from pytorch_lightning.strategies.ddp import DDPStrategy
import numpy as np

def main(cfg):

    if cfg.TASK == "simple_detection":
        TaskType = SimpleDetectionTask
    elif cfg.TASK == "sta":
        TaskType = STATask
    else:
        raise NotImplementedError(f"Task {cfg.TASK} not implemented")

    task = TaskType(cfg)
    ckp_path = cfg.CHECKPOINT_FILE_PATH
    if ckp_path!="":
        task.load_from_checkpoint_list(ckp_path)
    checkpoint_callback = ModelCheckpoint(
        monitor=task.checkpoint_metric, 
        mode="max", 
        save_last=True, 
        save_top_k=cfg.SAVE_TOP_K,
        filename='{epoch:02d}-{step:07d}-{'+task.checkpoint_metric+':.4f}'
    )
    print(f"Logging enabled: {cfg.ENABLE_LOGGING}")
    if cfg.ENABLE_LOGGING:
        args = {
            "callbacks": [
                LearningRateMonitor(), 
                checkpoint_callback
            ], 
            "logger": [
                StillFastLogger(
                    cfg, 
                    summary_metric=task.checkpoint_metric, 
                    summary_mode='max'
                    )
                ]
            }
    else:
        args = {"logger": False, "callbacks": checkpoint_callback}

    def get_strategy(strategy):
        if strategy=='ddp':
            return DDPStrategy(find_unused_parameters=False)
        else:
            raise ValueError(f"Strategy {strategy} not implemented")
    
    trainer = Trainer(
        accelerator=cfg.SOLVER.ACCELERATOR,
        devices=cfg.NUM_DEVICES,
        num_nodes=cfg.NUM_SHARDS,
        strategy=get_strategy(cfg.SOLVER.STRATEGY),
        max_epochs=cfg.SOLVER.MAX_EPOCH,
        num_sanity_val_steps=3,
        benchmark=cfg.SOLVER.BENCHMARK,
        precision=cfg.SOLVER.PRECISION,
        replace_sampler_ddp=cfg.SOLVER.REPLACE_SAMPLER_DDP,
        fast_dev_run=cfg.FAST_DEV_RUN,
        default_root_dir=join(cfg.OUTPUT_DIR, cfg.TASK),
        **args,
    )


    if cfg.TRAIN.ENABLE and cfg.TEST.ENABLE:
        trainer.fit(task)

        # Calling test without the lightning module arg automatically selects the best
        # model during training.
        return trainer.test()

    elif cfg.TRAIN.ENABLE:
        return trainer.fit(task)

    elif cfg.TEST.ENABLE:
        result = trainer.test(task)
        return result

    elif cfg.VAL.ENABLE:
        result = trainer.validate(task)[0]
        if hasattr(cfg,'TEST_DIR'):
            api = wandb.Api()
            run = api.run(cfg.WANDB_RUN)
            for k,v in result.items():
                run.summary[k+'.final']= v
            run.summary.update()
            run.update()
            api.flush()
        return result

def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        cfg (str): path to the config file.
        opts (argument): provide additional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument(
        "--num_shards", help="Number of shards using by the job", default=1, type=int
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See stillfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the model"
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Validate the model"
    )
    parser.add_argument(
        "--parallel_test",
        action="store_true",
        help="Allow testing with batch>1 - results may not be accurate"
    )
    parser.add_argument(
        '--checkpoint',
        help="Path to the checkpoint file",
        type=str,
        default=None
    )
    parser.add_argument(
        '--exp',
        help="Name of the experiment",
        type=str,
        default='default'
    )
    parser.add_argument(
        '--test_dir',
        help="Path to the directory for which to produce results",
        type=str,
        default=None
    )

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()

def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards"):
        cfg.NUM_SHARDS = args.num_shards
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir
    if hasattr(args, "fast_dev_run"):
        cfg.FAST_DEV_RUN = args.fast_dev_run
        if args.fast_dev_run:
            cfg.ENABLE_LOGGING = False
    if hasattr(args, "train"):
        cfg.TRAIN.ENABLE = args.train
    if hasattr(args, "test"):
        cfg.TEST.ENABLE = args.test
    if hasattr(args, "val"):
        cfg.VAL.ENABLE = args.val
    if args.checkpoint is not None:
        cfg.CHECKPOINT_FILE_PATH = args.checkpoint
        cfg.MODEL.STILL.PRETRAINED = False # disable pretrained model since we are loading a checkpoint
    if args.test_dir:
        cfg.TEST_DIR = args.test_dir
        if args.cfg_file is None:
            cfg.merge_from_file(join(cfg.TEST_DIR, "config.yaml"))
            if hasattr(args, "train"):
                cfg.TRAIN.ENABLE = args.train
            if hasattr(args, "test"):
                cfg.TEST.ENABLE = args.test
            if hasattr(args, "val"):
                cfg.VAL.ENABLE = args.val

        results_dir = join(cfg.TEST_DIR, "results")
        os.makedirs(results_dir, exist_ok=True)
        cfg.VAL.OUTPUT_JSON = join(results_dir, "val.json")
        cfg.TEST.OUTPUT_JSON = join(results_dir, "test.json")

        if not args.parallel_test:
            cfg.NUM_DEVICES = 1
            cfg.VAL.BATCH_SIZE = 1
            cfg.TEST.BATCH_SIZE = 1

        if args.checkpoint is None:
            checkpoints = glob(join(cfg.TEST_DIR, 'checkpoints', '*.ckpt'))
            checkpoints = [c for c in checkpoints if 'last' not in c]
            if len(checkpoints)>1:
                scores = [float(c.split('/')[-1].split('=')[-1].split('.ckp')[0]) for c in checkpoints]
                checkpoints = np.array(checkpoints)[np.argsort(scores)[::-1]]
                print(f"More than one checkpoint found in {cfg.TEST_DIR}.", end=' ')
                topk = min(len(checkpoints), cfg.AVERAGE_TOP_K_CHECKPOINTS)
                print(f"Averaging the top {topk} checkpoints:")
                checkpoints = checkpoints[:topk]
                checkpoints = [str(x) for x in checkpoints]
                # Please specify the checkpoint file using the --checkpoint argument.".format(cfg.TEST_DIR))
            elif len(checkpoints)==0:
                print('No checkpoint found. Quitting.')
                exit(1)
            else:
                assert len(checkpoints)==1
                
            cfg.CHECKPOINT_FILE_PATH = checkpoints
            cfg.MODEL.STILL.PRETRAINED = False # disable pretrained model since we are loading a checkpoint

    cfg.EXPERIMENT_NAME = cfg.MODEL.NAME + '_' + args.exp
    
    if not cfg.TRAIN.ENABLE:
        cfg.ENABLE_LOGGING = False

    return cfg

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    main(cfg)
