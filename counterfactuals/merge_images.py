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
parser.add_argument("--edits", type=int, required=False)
parser.add_argument("--index", type=int, required=False)
parser.add_argument("--radius", type=float, required=False)



def main():
    args = parser.parse_args()

    dirpath = os.path.join(Path.output_root_dir(), args.input_path)
    n_samples = args.samples or 10
    radius = args.radius
    n_edits = args.edits
    index = args.index

    if args.seed is not None:
        np.random.seed(args.seed)

    dataset = get_test_dataset(get_vis_transform(), return_image_only=True)

    counterfactuals = np.load(
        os.path.join(dirpath, "counterfactuals.npy"), allow_pickle=True
    ).item()

    samples = []
    if index is not None:
        samples = [index]
    else:
        samples = np.random.choice(list(counterfactuals.keys()), n_samples)

    for idx in samples:
        dirpath_output = os.path.join(Path.output_root_dir(), "examples", args.input_path, f"merge_{idx}")
        os.makedirs(dirpath_output, exist_ok=True)
        cf = counterfactuals[idx]

        visualize_edits(
            edits=cf["edits"],
            query_index=cf["query_index"],
            distractor_index=cf["distractor_index"],
            dataset=dataset,
            n_pix=7,
            radius=radius,
            n_edits=n_edits,
            fname=f"output/examples/{args.input_path}/merge_{idx}",
        )


if __name__ == "__main__":
    main()
