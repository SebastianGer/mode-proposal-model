# Multi-modal data generation
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from simfire.sim.simulation import FireSimulation
from tqdm import tqdm

from .simfire_config_class import SamplingLatLonConfig, default_config_params

raise RuntimeError(
    "This script is only provided to document how the MMFire dataset was generated. It should not be run without receiving explicit approval from the LANDFIRE program beforehand. "
    "This code repeatedly queries real-world environmental data from the LANDFIRE product service, which is actively used by firefighters and fire responders and should thus be treated with care."
    "If you would like to generate your own version of this dataset for some reason, please download the full data at https://landfire.gov/data/FullExtentDownloads "
    "and adjust the corresponding code to query this local file instead of the critical live service. "
)


seed_offset = int(sys.argv[1])
print(f"Starting simulation with {seed_offset=}")

base_config = deepcopy(default_config_params)


def increment_seeds(seed_offset):
    base_config["operational"]["seed"] += seed_offset
    base_config["wind"]["perlin"]["speed"]["seed"] += seed_offset
    base_config["wind"]["perlin"]["direction"]["seed"] += seed_offset
    np.random.seed(base_config["operational"]["seed"])


increment_seeds(seed_offset)

for i in tqdm(range(1000)):
    datapath = None
    try:
        config = SamplingLatLonConfig(config_dict=deepcopy(base_config))
        for j in range(8):
            if j == 0:
                base_config["simulation"]["save_data"] = True
            else:
                base_config["simulation"]["save_data"] = False

            config = SamplingLatLonConfig(config_dict=deepcopy(base_config))
            sim = FireSimulation(config)

            first_img, _ = sim.run("10m")
            if j == 0:
                config.simulation.save_data = False

                # Save first image, which is the same for all different simulations
                first_img = first_img.copy()
                datapath = sim.sf_home / "data" / sim.start_time
                plt.imsave(str(datapath / f"fire_pre_{j}.png"), first_img, cmap="gray")
            # Set uniform wind direction across whole image
            new_direction = j * 45
            sim.environment.U_dir = np.full(
                sim.environment.U_dir.shape, fill_value=new_direction
            )
            sim.fire_manager.environment = sim.environment
            sim.fire_manager.U_dir = sim.environment.U_dir

            second_img, _ = sim.run("10m")

            plt.imsave(str(datapath / f"fire_post_{j}.png"), second_img, cmap="gray")
    except Exception as e:
        print("Exception caught, skipping this item: ", e)

    increment_seeds(1)
