import importlib

import wandb
from tqdm import tqdm


def download_if_path_is_wandb_artifact(path: str, inside_a_run: bool = True):
    """_summary_ If the path does not start with "wandb://", it is assumed to be a wandb artifact path and is downloaded.

    Args:
        path (str): _description_

    Returns:
        _type_: _description_ The path to the downloaded file.
    """
    if path is not None and not path.startswith("/"):
        artifact_path = path.replace("wandb:", "")
        if inside_a_run:
            artifact = wandb.run.use_artifact(artifact_path, type="model")
        else:
            api = wandb.Api()
            artifact = api.artifact(artifact_path, type="model")
        artifact_dir = artifact.download()
        return f"{artifact_dir}/model.ckpt"
    return path


def download_val_preds_from_run(run_id: str, save_dir: str):
    """_summary_ Downloads the validation predictions from a wandb run.

    Args:
        run_id (str): _description_
        val_preds_batch_names (List[str]): _description_
    """
    api = wandb.Api()
    run = api.run(run_id)
    run_specific_save_dir = f"{save_dir}/{run_id}"
    for f in tqdm(
        [f"val_preds_batch{i}.pt" for i in range(105)],
        desc="Downloading validation predictions...",
    ):
        run.file(f).download(root=run_specific_save_dir, exist_ok=True, replace=False)

    run.file("val_distribution_eval.csv").download(
        root=run_specific_save_dir, exist_ok=True
    )

    return run_specific_save_dir


def download_val_csv(run_id: str, save_dir: str):
    """_summary_ Downloads the validation evaluation csv from a wandb run, in case we only need this.

    Args:
        run_id (str): _description_
        val_preds_batch_names (List[str]): _description_
    """
    api = wandb.Api()
    run = api.run(run_id)
    run_specific_save_dir = f"{save_dir}/{run_id}"

    run.file("val_distribution_eval.csv").download(
        root=run_specific_save_dir, exist_ok=True
    )

    return f"{run_specific_save_dir}/val_distribution_eval.csv"


def load_model_from_class_path(checkpoint_path, class_path):
    """
    Load a Lightning model given a class path string.

    Args:
        checkpoint_path: Path to the checkpoint file
        class_path: String like "my_models.vision.ResNetModel" or "models.MyModel"
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class.load_from_checkpoint(checkpoint_path, strict=False)
