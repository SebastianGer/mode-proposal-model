"""
Convert a directory of single-tensor .pt files into compressed .npz files.

Usage::

    python -m firespreadscenarios.pt_to_npz /path/to/pt_dir /path/to/npz_dir
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def convert_dir(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_files = sorted(input_dir.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {input_dir}")
        return

    for pt_file in tqdm(pt_files):
        tensor = torch.load(pt_file, map_location="cpu", weights_only=True)
        np.savez_compressed(output_dir / f"{pt_file.stem}.npz", tensor.detach().cpu().numpy())
    print(f"Converted {len(pt_files)} file(s) to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing .pt files")
    parser.add_argument("output_dir", type=Path, help="Directory to write .npz files to")
    args = parser.parse_args()
    convert_dir(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
