import pytorch_lightning as pl
import torch
from dataloader.CityscapesDataModule import CityscapesDataModule
from dataloader.LIDCDataModule import LIDCDataModule
from dataloader.SimfireDataModule import SimfireDataModule
from models.ProbabilisticUNet import BaseProbabilisticUNetLightning
from models.PUNetVariants import CityscapesPUNet, LIDCPUNet, SimfirePUNet
from utils.BaseLightningCLI import CityscapesLightningCLI
from utils.utils import download_if_path_is_wandb_artifact

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    cli = CityscapesLightningCLI(
        BaseProbabilisticUNetLightning,
        pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        run=False,
    )
    cli.wandb_setup(wandb_metrics={"val_loss": "min", "train_loss_epoch": "min"})

    if cli.config.do_train:
        # If we have a checkpoint path, and do_train is set, we resume training from that checkpoint.
        ckpt = cli.config.ckpt_path
        if ckpt is not None and ckpt != "":
            ckpt = download_if_path_is_wandb_artifact(ckpt)

        cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=ckpt)

        # Switches to full sampling for every conditioning
        cli.model.training_ended = True

    # If we have trained a model, use the best checkpoint for validation and testing.
    # Without this, the model's state at the end of the training would be used, which is not necessarily the best.
    if cli.config.do_train:
        ckpt = "best"
    else:
        ckpt = cli.config.ckpt_path
        if ckpt is not None and ckpt != "":
            ckpt = download_if_path_is_wandb_artifact(ckpt)
        else:
            print(
                f"WARNING: No training is performed, but ckpt_path is also not set. Validation and/or testing are performed with untrained model."
            )

    if cli.config.do_validate:
        cli.trainer.validate(cli.model, cli.datamodule, ckpt_path=ckpt)

    if cli.config.do_test:
        cli.trainer.test(cli.model, cli.datamodule, ckpt_path=ckpt)
