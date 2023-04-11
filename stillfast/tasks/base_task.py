from pytorch_lightning.core import LightningModule
from ..models import build_model
from ..optimizers import lr_scheduler
from ..datasets import loader
import torch

class BaseTask(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters()
        self.model = build_model(cfg)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def training_step_end(self, training_step_outputs):
        if self.cfg.SOLVER.ACCELERATOR == "dp":
            training_step_outputs["loss"] = training_step_outputs["loss"].mean()
        return training_step_outputs

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def forward(self, inputs):
        return self.model(inputs)

    def setup(self, stage):
        self.train_loader = loader.construct_loader(self.cfg, "train")
        self.val_loader = loader.construct_loader(self.cfg, "val")
        self.test_loader = loader.construct_loader(self.cfg, "test")

    def configure_optimizers(self):
        steps_in_epoch = len(self.train_loader)
        return lr_scheduler.lr_factory(
            self.model, self.cfg, steps_in_epoch, self.cfg.SOLVER.LR_POLICY
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def load_from_checkpoint_list(self, ckps):
        if not isinstance(ckps, list):
            ckps = [ckps]
        
        
        if len(ckps)==1:
            state_dict = torch.load(ckps[0])["state_dict"]
        else:
            n = float(len(ckps))
            state_dict = None
            for ckp in ckps:
                sd = torch.load(ckp)["state_dict"]
                if state_dict is None:
                    state_dict = {k:v/n for k,v in sd.items()}
                else:
                    state_dict = {k:v+sd[k]/n for k,v in state_dict.items()}

        self.load_state_dict(state_dict)