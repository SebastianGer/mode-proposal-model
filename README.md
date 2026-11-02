# Deterministic Mode Proposals: An Efficient Alternative to Generative Sampling for Ambiguous Segmentation

This repository contains a cleaned-up version of the code used for the experiments in `Deterministic Mode Proposals: An Efficient Alternative to Generative Sampling for Ambiguous Segmentation`, accepted for publication at WACV 2027.

This repository continues the code of our previous paper, [Wildfire Spread Scenarios: Increasing Sample Diversity of Segmentation Diffusion Models with Training-Free Methods](https://github.com/SebastianGer/wildfire-spread-scenarios). The diffusion models and the diversity-encouraging sampling methods from that paper are also part of this repository.

## Setup 

Setting up the environment:

``` pip3 install -r requirements.txt ```

1. Download and prepare LIDC according to [https://github.com/gaozhitong/MoSE-AUSeg/](https://github.com/gaozhitong/MoSE-AUSeg/).
2. Download the Cityscapes dataset from [https://www.cityscapes-dataset.com/downloads/](https://www.cityscapes-dataset.com/downloads/) (`gtFine_trainvaltest.zip` and `leftImg8bit_trainvaltest.zip`).
3. Download the MMFire dataset from [https://doi.org/10.5281/zenodo.18037549](https://doi.org/10.5281/zenodo.18037549). In the code and configs, MMFire is called `simfire`.
4. Add the corresponding paths in `cfgs/dataset_name/dataset.yaml`. 
5. If you want to use the probabilistic UNet, follow the setup instruction 'Adding KL divergence for Independent distribution' in [https://github.com/stefanknegt/Probabilistic-Unet-Pytorch](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch).

Install the project as a local python project, to be able to import the files from `third_party/` from `src/`. If you already have installed packages with the same names as those under `third_party/`, that might lead to problems or damage of some kind, so do not do that. 

``` pip install -e . ```

## Re-running our experiments

We use wandb to log experimental results. This can be turned off by setting the environment variable `WANDB_MODE=disabled`. The results will then be logged to a local directory instead.

Experiments are parameterized via yaml files in the `cfgs` directory. Arguments are parsed via the [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html). Some experiments are done via WandB sweeps. These sweeps are parameterized via yaml files that are prefixed with `wandb_`. We omit explanations of how to use wandb sweeps to run experiments and refer the readers to the [original documentation](https://docs.wandb.ai/guides/sweeps). To run the same experiments without WandB, the parameters specified in the WandB sweep configuration file can simply be passed via the command line, similar to the base model training below. 

### Checkpoints

Configs that need a trained model contain the value `TODO` for `ckpt_path`. Replace it with the checkpoint of your own model before you run these configs. A `ckpt_path` that starts with `/` is loaded as a local file. Any other value is interpreted as a wandb artifact (`entity/project/model-<run_id>:<version>`) and downloaded.

### Training the base models

For MMFire and LIDC, we use the diffusion models from our previous paper. For Cityscapes, we use a new train/validation split: 500 random images from the official training set are the validation set, and the official validation set is the test set. Thus, we retrained the Cityscapes diffusion model on this split. The Prob. UNet results are taken from the previous paper.

Cityscapes diffusion model
```
python3 src/firespreadscenarios/run_diffusion_generic.py -c cfgs/cityscapes/base_config_image_cond.yaml --data cfgs/cityscapes/dataset.yaml --trainer cfgs/cityscapes/trainer.yaml
```

Cityscapes Prob. UNet
```
python3 src/firespreadscenarios/run_PUNet_generic.py -c cfgs/cityscapes/PUNet.yaml --data cfgs/cityscapes/dataset.yaml --trainer cfgs/cityscapes/PUNet_trainer.yaml
```

MMFire diffusion model
```
python3 src/firespreadscenarios/run_diffusion_generic.py --trainer=cfgs/simfire/trainer.yaml --data=cfgs/simfire/dataset.yaml --config=cfgs/simfire/base_config_image_cond.yaml --trainer.max_epochs=1000
```

MMFire Prob. UNet
```
python3 src/firespreadscenarios/run_PUNet_generic.py --trainer=cfgs/simfire/PUNet_trainer.yaml --data=cfgs/simfire/dataset.yaml --config=cfgs/simfire/PUNet.yaml 
```

LIDC diffusion model
```
python3 src/firespreadscenarios/run_diffusion_generic.py -c cfgs/LIDC/base_config_image_cond.yaml --data cfgs/LIDC/dataset.yaml --trainer cfgs/LIDC/trainer.yaml
```

LIDC Prob. UNet
```
python3 src/firespreadscenarios/run_PUNet_generic.py --trainer=cfgs/LIDC/PUNet_trainer.yaml --data=cfgs/LIDC/dataset.yaml --config=cfgs/LIDC/PUNet.yaml 
```

### Training the mode proposal networks

The mode proposal network is initialized from a trained diffusion model. Set `model.init_args.ckpt_path` in `cfgs/<dataset>/mode_prediction_network_*.yaml` to the checkpoint of the diffusion model of the same dataset. For each dataset, there is one config for the fully-labeled scenario (`mode_prediction_network_full.yaml`) and one for the single-label scenario (`mode_prediction_network_single.yaml`). The configs contain the hyperparameters from the paper.

Cityscapes
```
python3 src/firespreadscenarios/run_mode_prediction_generic.py -c cfgs/cityscapes/mode_prediction_network_full.yaml --data cfgs/cityscapes/dataset.yaml --trainer cfgs/cityscapes/trainer_proposal.yaml
```

MMFire
```
python3 src/firespreadscenarios/run_mode_prediction_generic.py -c cfgs/simfire/mode_prediction_network_full.yaml --data cfgs/simfire/dataset.yaml --trainer cfgs/simfire/trainer_proposal.yaml
```

LIDC
```
python3 src/firespreadscenarios/run_mode_prediction_generic.py -c cfgs/LIDC/mode_prediction_network_full.yaml --data cfgs/LIDC/dataset.yaml --trainer cfgs/LIDC/trainer_proposal.yaml
```

For the single-label scenario, replace `_full.yaml` with `_single.yaml`. For the LIDC models without selection loss ($\lambda=0$), add `--model.init_args.prob_loss_weight=0 --model.init_args.num_proposals=<16|32|64> --trainer.max_epochs=10`.

### Mode probabilities via velocity decomposition

To estimate mode probabilities with velocity decomposition, evaluate a trained mode proposal network. Set `--ckpt_path` to the checkpoint of the mode proposal network, and keep `model.init_args.ckpt_path` set to the diffusion model:

```
python3 src/firespreadscenarios/run_mode_prediction_generic.py -c cfgs/simfire/mode_prediction_network_full.yaml --data cfgs/simfire/dataset.yaml --trainer cfgs/simfire/trainer_proposal.yaml --do_train=false --ckpt_path=/path/to/mode_proposal.ckpt --model.init_args.estimate_mode_probabilities=NNLS_SUM_TO_ONE --model.init_args.estimate_mode_probabilities_hm_only=false
```

Add these arguments for the variants in the paper:
- Filter proposals by selection score: `--model.init_args.estimate_mode_probabilities_filter_selection_score=true`
- Filter duplicate proposals: `--model.init_args.estimate_mode_probabilities_iou_threshold=0.95`
- Decompose over the ground truth masks instead of the proposals: `--model.init_args.estimate_mode_probabilities_from_gt=true`
- Cityscapes, structure-aware (low-dim) estimation: `--model.init_args.estimate_cityscapes_priors=true`

### Mode probabilities of diffusion models

To evaluate the mode probabilities of a diffusion model via large-scale clustering, first write the generated samples to disk. For MMFire and Cityscapes, we sample with DPM++ (5 steps, $\rho=2$), with 2048 and 32768 samples per input to reduce the necessary runtime for this large number of samples. One run of the configs below generates 1024 samples per input, so we start 2 (MMFire) or 32 (Cityscapes) runs in parallel with the same `write_samples_dir`. The file names contain the wandb run id, so the runs do not overwrite each other. For LIDC, we use EDM's Heun solver (10 steps, $\rho=7$) with 64 samples per input in one run.

MMFire
```
python3 src/firespreadscenarios/run_diffusion_generic.py -c cfgs/simfire/base_config_image_cond_sampling_dpmpp.yaml --data cfgs/simfire/dataset.yaml --trainer cfgs/simfire/trainer.yaml --model.init_args.write_samples_dir=/path/to/saved_samples
```

Cityscapes
```
python3 src/firespreadscenarios/run_diffusion_generic.py -c cfgs/cityscapes/base_config_image_cond_sampling_dpmpp.yaml --data cfgs/cityscapes/dataset.yaml --trainer cfgs/cityscapes/trainer.yaml --model.init_args.write_samples_dir=/path/to/saved_samples
```

LIDC
```
python3 src/firespreadscenarios/run_diffusion_generic.py -c cfgs/LIDC/base_config_image_cond_sampling.yaml --data cfgs/LIDC/dataset.yaml --trainer cfgs/LIDC/trainer.yaml --model.init_args.aggregate_n_samples=64 --model.init_args.write_samples_dir=/path/to/saved_samples
```

Then cluster the samples with HDBSCAN and compute the Brier Skill Score. We use a minimum cluster size of 2 for all datasets, $\epsilon=0.05$ for MMFire and $\epsilon=0$ for Cityscapes and LIDC:

```
python -m firespreadscenarios.evaluate_diffusion_brier --dataset mmfire --data_dir /path/to/mmfire --split val --samples_dir /path/to/saved_samples --batch_size 32 --eps 0.05 --min_samples 2
python -m firespreadscenarios.evaluate_diffusion_brier --dataset cityscapes --data_dir /path/to/cityscapes --split val --samples_dir /path/to/saved_samples --batch_size 32 --eps 0 --min_samples 2
python -m firespreadscenarios.evaluate_diffusion_brier --dataset lidc --data_dir /path/to/lidc/npy --split val --samples_dir /path/to/saved_samples --batch_size 32 --eps 0 --min_samples 2
```

`--batch_size` must be the same value that was used to generate the samples to match the file names. Use `--help` to see all arguments.

## Citation

```
```
