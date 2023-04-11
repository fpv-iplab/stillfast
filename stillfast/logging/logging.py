from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
#from pytorch_lightning.utilities import _convert_params, _flatten_dict, _sanitize_callable_params
import os
from os.path import join
import wandb

class StillFastLogger(LightningLoggerBase):

    def __init__(self, cfg, summary_metric=None, summary_mode=None, version = None, **kwargs):
        super().__init__()
        self.cfg = cfg
        self._version = version
        self._summary_metric = summary_metric
        self._summary_mode = summary_mode

        self._wandb_init = {
            "name": f"{self.name}_{self.version}",
            "dir": self.log_dir
        }

        self._wandb_init.update(**kwargs)

        _ = self.experiment

        self._write_cfg()

    @rank_zero_only
    def _write_cfg(self):
        self.cfg.WANDB_RUN = f"{self.experiment.entity}/{self.experiment.project}/{self.experiment.id}"
        with open(join(self.log_dir, "config.yaml"), "w") as f:
            f.write(str(self.cfg))
        
    @property
    def name(self):
        return self.cfg.EXPERIMENT_NAME

    @property
    def output_dir(self):
        return self.cfg.OUTPUT_DIR

    @property
    def root_dir(self) -> str:
        return os.path.join(self.output_dir, self.cfg.TASK, self.name)

    @rank_zero_only
    def _get_next_version(self):
        return self._get_current_version() + 1

    def _get_current_version(self):
        root_dir = self.root_dir
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)

        existing_versions = []
        for d in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("version_"):
                existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            return -1

        return max(existing_versions)

    @property
    def log_dir(self):
        log_dir = os.path.join(self.root_dir, self.version)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        return log_dir

    @rank_zero_only
    def log_metrics(self, metrics, step):
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        if step is not None:
            self.experiment.log({**metrics, "trainer/global_step": step})
        else:
            self.experiment.log(metrics)

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        if not hasattr(self, "_experiment") and wandb.run is not None:
            rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
        else:
            self._experiment = wandb.init(**self._wandb_init)

            # define default x-axis
            if getattr(self._experiment, "define_metric", None):
                self._experiment.define_metric("trainer/global_step")
                self._experiment.define_metric("epoch")
                self._experiment.define_metric("train/*", step_metric="trainer/global_step", step_sync=True)
                self._experiment.define_metric("val/*", step_metric="epoch", step_sync=True)
                if self._summary_metric is not None and self._summary_mode is not None:
                    self._experiment.define_metric(self._summary_metric, summary=self._summary_mode)

        return self._experiment


    @property
    def version(self) -> int:
        """Gets the version of the experiment.
        Returns:
            The version of the experiment if it is specified, else the next version.
        """
        if self._version is None:
            self._version = self._get_next_version()

        if self._version is None:
            self._version = self._get_current_version()
        
        return f"version_{self._version}"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        # params = _convert_params(params)
        # params = _flatten_dict(params)
        # params = _sanitize_callable_params(params)
        self.experiment.config.update(params, allow_val_change=True)
        pass

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
