import os

import wandb
from pytorch_lightning.cli import LightningCLI


class BaseLightningCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "trainer.default_root_dir", "trainer.logger.init_args.save_dir"
        )
        parser.add_argument("--do_train", type=bool, help="If True: train the model.")
        parser.add_argument(
            "--do_predict", type=bool, help="If True: compute predictions."
        )
        parser.add_argument(
            "--do_test", type=bool, help="If True: compute test metrics."
        )
        parser.add_argument(
            "--do_validate",
            type=bool,
            default=False,
            help="If True: compute val metrics.",
        )
        parser.add_argument(
            "--ckpt_path",
            type=str,
            default=None,
            help="Path to checkpoint, can be wandb URI.",
        )

    def before_fit(self):
        self.wandb_setup()

    def before_test(self):
        self.wandb_setup()

    def before_validate(self):
        self.wandb_setup()

    def define_wandb_metrics(self, wandb_metrics: dict):
        for metric_name, min_max in wandb_metrics.items():
            wandb.define_metric(metric_name, summary=min_max)

    def wandb_setup(self, wandb_metrics: dict):
        """
        Save the config used by LightningCLI to disk, then save that file to wandb.
        Using wandb.config adds some strange formating that means we'd have to do some
        processing to be able to use it again as CLI input.
        """
        if wandb.run is None:
            wandb.init()

        if self.trainer.is_global_zero:
            config_file_name = os.path.join(wandb.run.dir, "cli_config.yaml")

            cfg_string = self.parser.dump(self.config, skip_none=False)
            with open(config_file_name, "w") as f:
                f.write(cfg_string)
            wandb.save(config_file_name, policy="now", base_path=wandb.run.dir)
            self.define_wandb_metrics(wandb_metrics)


class CityscapesLightningCLI(BaseLightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.link_arguments(
            "data.init_args.binarization_mode", "model.init_args.dataset_evaluator_arg"
        )
