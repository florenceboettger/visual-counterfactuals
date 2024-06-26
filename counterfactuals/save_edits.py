import argparse
import csv
import os

import numpy as np

from utils.common_config import get_test_dataset, get_test_transform
from utils.path import Path

parser = argparse.ArgumentParser(description="Save first edit for counterfactual explanations")
parser.add_argument("--input_path", type=str, required=True)

def main():
    args = parser.parse_args()

    dirpath = os.path.join(Path.output_root_dir(), args.input_path)

    counterfactuals = np.load(
        os.path.join(dirpath, "counterfactuals.npy"), allow_pickle=True
    ).item()

    dataset = get_test_dataset(get_test_transform(), return_image_only=False)

    result_path_edits = os.path.join(Path.output_root_dir(), "new_results", "edits", args.input_path)
    os.makedirs(result_path_edits, exist_ok=True)

    with open(os.path.join(result_path_edits, "edits.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=["query_index", "query_class", "distractor_index", "distractor_class", "query_edit", "distractor_edit", "bbox_x", "bbox_y", "bbox_width", "bbox_height"])
        writer.writeheader()
        for c in list(counterfactuals.values()):
            query_index = c["query_index"]
            bbox = dataset.__getitem__(int(query_index))["bbox"]
            query_class = dataset.__getitem__(int(query_index))["target"]
            distractor_index = c["distractor_index"]
            distractor_class = dataset.__getitem__(int(distractor_index[0]))["target"]
            writer.writerow({
                "query_index": query_index,
                "query_class": query_class,
                "distractor_index": distractor_index,
                "distractor_class": distractor_class,
                "query_edit": c["edits"][0][0],
                "distractor_edit": c["edits"][0][1],
                "bbox_x": bbox[0],
                "bbox_y": bbox[1],
                "bbox_width": bbox[2],
                "bbox_height": bbox[3],
            })

if __name__ == "__main__":
    main()