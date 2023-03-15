import argparse
import os
from datetime import datetime

import numpy as np

from utils.common_config import get_test_dataset, get_vis_transform
from utils.path import Path
from utils.visualize import visualize_edits


parser = argparse.ArgumentParser(description="Visualize counterfactual explanations by merging images")
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--seed", type=int, required=False)
parser.add_argument("--samples", type=int, required=False)
parser.add_argument("--radius", type=float, required=False)



def main():
    args = parser.parse_args()

    dirpath = os.path.join(Path.output_root_dir(), args.input_path)
    samples = args.samples or 10
    radius = args.radius
    if args.seed is not None:
        np.random.seed(args.seed)

    dataset = get_test_dataset(get_vis_transform(), return_image_only=True)

    counterfactuals = np.load(
        os.path.join(dirpath, "counterfactuals.npy"), allow_pickle=True
    ).item()

    dirpath_output = os.path.join(Path.output_root_dir(), "examples", args.input_path)
    os.makedirs(dirpath_output, exist_ok=True)

    for idx in np.random.choice(list(counterfactuals.keys()), samples):
        cf = counterfactuals[idx]

        visualize_edits(
            edits=cf["edits"],
            query_index=cf["query_index"],
            distractor_index=cf["distractor_index"],
            dataset=dataset,
            n_pix=7,
            radius=radius
            fname=f"output/examples/{args.input_path}/merge_{idx}.png",
        )


if __name__ == "__main__":
    main()
