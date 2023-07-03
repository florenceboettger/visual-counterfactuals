import argparse
import csv
import os

import numpy as np

from utils.common_config import get_test_dataset, get_vis_transform
from utils.path import Path

parser = argparse.ArgumentParser(description="Save first edit for counterfactual explanations")
parser.add_argument("--input_path", type=str, required=True)

def main():
    args = parser.parse_args()

    dirpath = os.path.join(Path.output_root_dir(), args.input_path)

    counterfactuals = np.load(
        os.path.join(dirpath, "counterfactuals.npy"), allow_pickle=True
    ).item()

    result_path_edits = os.path.join(Path.output_root_dir(), "new_results", "edits", args.input_path)
    os.makedirs(result_path_edits, exist_ok=True)

    with open(os.path.join(result_path_edits, "edits.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=["query_index", "distractor_index", "query_edit", "distractor_edit"])
        writer.writeheader()
        for c in list(counterfactuals.values()):
            writer.writerow({
                "query_index": c["query_index"],
                "distractor_index": c["distractor_index"],
                "query_edit": c["edits"][0][0],
                "distractor_edit": c["edits"][0][1],
            })

if __name__ == "__main__":
    main()