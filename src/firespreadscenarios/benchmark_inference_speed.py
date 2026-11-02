"""
Measures inference speed of mode prediction networks, diffusion models and PUNets.

Protocol:
- One GPU, no other jobs on it.
- Dummy inputs are made on the GPU before timing: no data loading, no host-to-device copy.
- Checkpoint download and loading are not timed.
- Explicit warmup calls before timing (CUDA context, cuDNN autotune, allocator caching).
- The same fixed number of timed runs for every model (not a fixed time budget, which
  would give the slowest model the fewest runs). Each run is enclosed by
  torch.cuda.synchronize(), so it measures the full GPU work. We report median and IQR.
- Same global settings for all models (inference_mode, cudnn.benchmark, matmul precision
  as in the run_*_generic.py scripts, fixed thread count). Each model runs in its native
  precision, no AMP/compile.
- Each model is freed and the CUDA cache emptied before the next one, and peak memory is
  reset, so models do not share allocator state.
- Diffusion is timed twice: "parallel" (K samples in one sampler call with batch B*K,
  as in the eval code) and "single sample" (one sample per input, batch B).
- PUNet is timed twice: "parallel" (input copied K times, UNet at batch B*K, as in the
  eval code) and "encode once" (UNet at batch B, only fcomb at batch B*K).
- Timed part = conditioning tensor in, K continuous masks out. Post-processing
  (binarization, duplicate filtering, Hungarian matching, metrics) is not timed.

Usage (run from src/firespreadscenarios, like the other run scripts):
    python benchmark_inference_speed.py \
        --model models.SimfireDiffusionEval.SimfireModePredictionNetwork wandbuname/project/model-xxx:v0 64 64 7 \
        --model models.SimfireDiffusionEval.SimfireDiffusionEval wandbuname/project/model-yyy:v0 64 64 7 \
        --model models.PUNetVariants.SimfirePUNet wandbuname/project/model-zzz:v0 64 64 7 \
        --num-samples 16 --out speed.csv
"""

import argparse
import csv
import gc
import importlib

import torch
import time

import numpy as np

from models.BaseDiffusionEval import BaseDiffusionEval
from models.BaseModePredictionNetwork import BaseModePredictionNetwork
from models.ProbabilisticUNet import BaseProbabilisticUNetLightning
from models.sampling.sampling_strategies.ParticleGuidance import ParticleGuidance
from utils.utils import download_if_path_is_wandb_artifact

torch.set_float32_matmul_precision("medium")  # same as the run_*_generic.py scripts


def load_model(class_path, wandb_path, args):
    module_name, cls_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), cls_name)
    ckpt = download_if_path_is_wandb_artifact(wandb_path, inside_a_run=False)
    kwargs = {}
    if issubclass(cls, BaseDiffusionEval):
        # Naive sampling, as in the eval configs (particle_guidance_factor=0).
        kwargs = dict(
            sampling_strategy=ParticleGuidance(),
            timesteps=args.timesteps,
            use_second_order_correction=True,
            s_churn=0.0,
        )
    model = cls.load_from_checkpoint(ckpt, map_location="cuda", **kwargs)
    # map_location only moves checkpoint tensors; modules built in __init__ can stay on CPU.
    return model.to("cuda").eval()


def make_inference_fns(model, cond, K):
    """
    Returns (K_actual, variants). Each variant maps the fixed dummy cond to K masks per
    item: dict(mode, fn, nfe, k, batch_per_call, expected_shape), k = outputs per item.
    """
    B, _, H, W = cond.shape

    if isinstance(model, BaseModePredictionNetwork):
        K = model.hparams.num_proposals

        def fn():
            x0, _ = model(cond)
            return x0

        return K, [dict(mode="single pass", fn=fn, nfe="1", k=K, batch_per_call=B, expected_shape=(B, K, H, W))]

    cond_rep = cond.repeat_interleave(K, 0)

    if isinstance(model, BaseDiffusionEval):
        steps = model.hparams.timesteps
        nfe = 2 * steps - 1 if model.hparams.use_second_order_correction else steps

        def sample(c):
            # Latents are sampled inside the timed part, as in real use.
            latents = torch.randn((c.shape[0], 1, H, W), device=c.device)
            res = model.sampling_strategy.sample(
                model.diffusion_model, latents, c, None, model.hparams
            )
            return res["x_next"]

        # parallel: all K samples in one sampler call (batch B*K), as in compute_tvd.
        # single sample: one sample per input (batch B), i.e. the cost of one sample
        # without batching.
        return K, [
            dict(mode="parallel", fn=lambda: sample(cond_rep), nfe=str(nfe), k=K,
                 batch_per_call=B * K, expected_shape=(B * K, 1, H, W)),
            dict(mode="single sample", fn=lambda: sample(cond), nfe=str(nfe), k=1,
                 batch_per_call=B, expected_shape=(B, 1, H, W)),
        ]

    if isinstance(model, BaseProbabilisticUNetLightning):
        # Same call path as BaseProbabilisticUNetLightning.compute_tvd.
        def fn():
            model.model.forward(cond_rep, None, training=False)
            return model.model.sample(testing=True)

        # The prior net and UNet depend only on the input, so they can run once per input
        # (batch B); only the small fcomb head runs on the K latent samples (batch B*K).
        def fn_encode_once():
            punet = model.model
            punet.forward(cond, None, training=False)
            z = punet.prior_latent_space.sample((K,))  # [K, B, latent_dim]
            z = z.transpose(0, 1).reshape(B * K, -1)  # item-major, like repeat_interleave
            features = punet.unet_features.repeat_interleave(K, 0)
            return punet.fcomb.forward(features, z)

        return K, [
            dict(mode="parallel", fn=fn, nfe=f"1 UNet + {K} fcomb", k=K,
                 batch_per_call=B * K, expected_shape=(B * K, 1, H, W)),
            dict(mode="encode once", fn=fn_encode_once, nfe=f"1 UNet + {K} fcomb", k=K,
                 batch_per_call=B, expected_shape=(B * K, 1, H, W)),
        ]

    raise TypeError(f"Unsupported model class: {type(model)}")


def time_fn(fn, expected_shape, args):
    """Returns (median_ms, iqr_ms, peak_mem_MB) over args.runs runs, after warmup."""
    out = fn()
    assert tuple(out.shape) == expected_shape, (out.shape, expected_shape)
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(args.runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    q25, median, q75 = np.percentile(np.array(times) * 1e3, [25, 50, 75])
    return median, q75 - q25, torch.cuda.max_memory_allocated() / 2**20


def benchmark_one(class_path, wandb_path, H, W, C, args):
    torch.manual_seed(args.seed)
    model = load_model(class_path, wandb_path, args)
    cond = torch.randn((args.batch_size, C, H, W), device="cuda")
    n_params = sum(p.numel() for p in model.parameters())

    rows = []
    with torch.inference_mode():
        K, variants = make_inference_fns(model, cond, args.num_samples)
        if K != args.num_samples:
            print(f"WARNING: {class_path} produces K={K}, not --num-samples={args.num_samples}")

        for v in variants:
            median, iqr, peak_mb = time_fn(v["fn"], v["expected_shape"], args)
            rows.append({
                "class": class_path.rsplit(".", 1)[1],
                "ckpt": wandb_path,
                "mode": v["mode"],
                "H": H,
                "W": W,
                "C": C,
                "B": args.batch_size,
                "K": v["k"],
                "batch_per_call": v["batch_per_call"],
                "NFE": v["nfe"],
                "params_M": round(n_params / 1e6, 3),
                "median_ms": round(median, 3),
                "iqr_ms": round(iqr, 3),
                "n_runs": args.runs,
                "ms_per_sample": round(median / (args.batch_size * v["k"]), 4),
                "peak_mem_MB": round(peak_mb, 1),
            })
            print(rows[-1])

    del model, variants, cond
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model", nargs=5, action="append", required=True,
        metavar=("CLASS_PATH", "WANDB_PATH", "H", "W", "C"),
        help="Model class path, wandb artifact (or absolute ckpt path), image height, width, conditioning channels.",
    )
    p.add_argument("--num-samples", type=int, default=16, help="K samples per conditioning input (diffusion, PUNet).")
    p.add_argument("--batch-size", type=int, default=1, help="Number of conditioning inputs per call.")
    p.add_argument("--timesteps", type=int, default=10, help="Diffusion sampling steps.")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=100, help="Timed runs per model, the same for all models.")
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Optional CSV output path.")
    args = p.parse_args()

    assert torch.cuda.is_available(), "Benchmark expects a GPU."
    torch.backends.cudnn.benchmark = True  # autotuning happens during warmup
    torch.set_num_threads(args.num_threads)

    env = (
        f"GPU: {torch.cuda.get_device_name()} | torch {torch.__version__} | "
        f"CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}"
    )
    print(env)

    rows = []
    for class_path, wandb_path, H, W, C in args.model:
        rows += benchmark_one(class_path, wandb_path, int(H), int(W), int(C), args)

    print("\n" + env)
    keys = list(rows[0])
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(str(r[k]) for k in keys))

    if args.out:
        with open(args.out, "w", newline="") as f:
            f.write(f"# {env}\n")
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
