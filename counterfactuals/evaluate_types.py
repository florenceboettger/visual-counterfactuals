import argparse
import os
import csv

import numpy as np

from utils.common_config import get_test_dataset, get_test_transform
from utils.path import Path

from explainer.eval import compute_eval_metrics


parser = argparse.ArgumentParser(description="Evaluate counterfactual explanations based on subsets.")
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--type", choices=["any", "identical", "partial", "none", "full"], default="full", required=False)

def analyze_type(counterfactuals, match_type, input_path):
    sample_source = list(counterfactuals.keys())
    if match_type != "any":
        dict_path = os.path.join(Path.output_root_dir(), "new_results/edits/matches.csv")
        with open(dict_path, "r") as f:
            reader = list(csv.DictReader(f))
            sample_source = [int(row["query_index"]) for row in reader if row["match"] == match_type]

    edits_path = os.path.join(Path.outpoot_root_dir(), f"new_results/edits/{input_path}/edits.csv")
    with open(edits_path, "r") as f:
        reader = list(csv.DictReader(f))
        edits = [row for row in reader if int(row["query_index"]) in sample_source]

    bbox_x0 = [float(row["bbox_x"]) for row in edits]
    bbox_y0 = [float(row["bbox_y"]) for row in edits]
    bbox_x1 = [float(row["bbox_x"]) + float(row["bbox_width"]) for row in edits]
    bbox_y1 = [float(row["bbox_y"]) + float(row["bbox_height"]) for row in edits]

    bbox_x0_average = np.average(bbox_x0)
    bbox_y0_average = np.average(bbox_y0)
    bbox_x1_average = np.average(bbox_x1)
    bbox_y1_average = np.average(bbox_y1)

    bbox_x0_variance = np.var(bbox_x0)
    bbox_y0_variance = np.var(bbox_y0)
    bbox_x1_variance = np.var(bbox_x1)
    bbox_y1_variance = np.var(bbox_y1)

    filtered_counterfactuals = {}

    for i in sample_source:
        filtered_counterfactuals[i] = counterfactuals[i]

    
    average_num_edits = np.mean([len(res["edits"]) for res in filtered_counterfactuals.values()])
    print("Average number of edits is {:.2f}".format(average_num_edits))

    result = compute_eval_metrics(
        filtered_counterfactuals,
        dataset=get_test_dataset(transform=get_test_transform(), max_dist=0, parts_type="full"),
    )

    print("Eval results single edit: {}".format(result["single_edit"]))
    print("Eval results all edits: {}".format(result["all_edit"]))

    row_dict = {
        "type": match_type,
        "avg_edits": average_num_edits,
        "eval_single_same": result["single_edit"]["Same-KP"],
        "eval_single_near": result["single_edit"]["Near-KP"],
        "eval_all_same": result["all_edit"]["Same-KP"],
        "eval_all_near": result["all_edit"]["Near-KP"],
        "bbox_x0_average": bbox_x0_average,
        "bbox_y0_average": bbox_y0_average,
        "bbox_x1_average": bbox_x1_average,
        "bbox_y1_average": bbox_y1_average,
        "bbox_x0_variance": bbox_x0_variance,
        "bbox_y0_variance": bbox_y0_variance,
        "bbox_x1_variance": bbox_x1_variance,
        "bbox_y1_variance": bbox_y1_variance,
    }

    return row_dict

def main():
    args = parser.parse_args()

    dirpath = os.path.join(Path.output_root_dir(), args.input_path)    
    match_type = args.type

    counterfactuals = np.load(
        os.path.join(dirpath, "counterfactuals.npy"), allow_pickle=True
    ).item()

    result_dicts = []
    if match_type != "full":
        result_dicts.append(analyze_type(counterfactuals, match_type, args.input_path))
    else:
        for t in ["any", "identical", "partial", "none"]:
            result_dicts.append(analyze_type(counterfactuals, t, args.input_path))

    result_path = os.path.join(Path.output_root_dir(), "new_results", "match_analysis")
    os.makedirs(result_path, exist_ok=True)

    index = args.input_path.split('/')[1]

    with open(os.path.join(result_path, f"{match_type}_{index}.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "type",
            "avg_edits",
            "eval_single_same",
            "eval_single_near",
            "eval_all_same",
            "eval_all_near",
            "bbox_x0_average",
            "bbox_y0_average",
            "bbox_x1_average",
            "bbox_y1_average",
            "bbox_x0_variance",
            "bbox_y0_variance",
            "bbox_x1_variance",
            "bbox_y1_variance",
        ])
        writer.writeheader()
        writer.writerows(result_dicts)

if __name__ == "__main__":
    main()