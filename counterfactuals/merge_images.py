import argparse
import os
import datetime

import numpy as np

from utils.common_config import get_test_dataset, get_vis_transform
from utils.path import Path
from utils.visualize import visualize_edits


parser = argparse.ArgumentParser(description="Visualize counterfactual explanations by merging images")
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--seed", type=int, required=False)
parser.add_argument("--samples", type=int, required=False)



def main():
    args = parser.parse_args()

    dirpath = os.path.join(Path.output_root_dir(), args.input_path)
    seed = args.seed or datetime.now().timestamp()
    samples = args.samples or 10

    dataset = get_test_dataset(get_vis_transform(), return_image_only=True)

    counterfactuals = np.load(
        os.path.join(dirpath, "counterfactuals.npy"), allow_pickle=True
    ).item()

    dirpath_output = os.path.join(Path.output_root_dir(), "examples", args.input_path)
    os.makedirs(dirpath_output, exist_ok=True)

    np.random.seed(seed)
    for idx in np.random.choice(list(counterfactuals.keys()), samples):
        cf = counterfactuals[idx]

        visualize_edits(
            edits=cf["edits"],
            query_index=cf["query_index"],
            distractor_index=cf["distractor_index"],
            dataset=dataset,
            n_pix=7,
            fname=f"output/merged/{args.input_path}/example_{idx}.png",
        )


if __name__ == "__main__":
    main()
