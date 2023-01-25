# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import numpy as np

from utils.common_config import get_test_dataset, get_vis_transform
from utils.path import Path
from utils.visualize import visualize_counterfactuals


parser = argparse.ArgumentParser(description="Visualize counterfactual explanations")
parser.add_argument("--input_path", type=str, required=True)


def main():
    args = parser.parse_args()

    dirpath = os.path.join(Path.output_root_dir(), args.input_path)

    dataset = get_test_dataset(get_vis_transform(), return_image_only=True)

    counterfactuals = np.load(
        os.path.join(dirpath, "counterfactuals.npy"), allow_pickle=True
    ).item()

    dirpath_output = os.path.join(Path.output_root_dir(), "examples", args.input_path)
    os.makedirs(dirpath_output, exist_ok=True)

    for idx in np.random.choice(list(counterfactuals.keys()), 10):
        cf = counterfactuals[idx]

        visualize_counterfactuals(
            edits=cf["edits"],
            query_index=cf["query_index"],
            distractor_index=cf["distractor_index"],
            dataset=dataset,
            n_pix=7,
            fname=f"output/examples/{args.input_path}/example_{idx}.png",
        )


if __name__ == "__main__":
    main()
