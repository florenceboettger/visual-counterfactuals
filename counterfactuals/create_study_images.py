import argparse
import os
import csv

import numpy as np

from utils.common_config import get_test_dataset, get_vis_transform
from utils.path import Path
from utils.visualize import visualize_edit, save_image

parser = argparse.ArgumentParser(description="Create images for a study on counterfactual explanations")
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--seed", type=int, required=False)
parser.add_argument("--samples", type=int, required=False, default=10)

def main():
    args = parser.parse_args()

    dirpath = os.path.join(Path.output_root_dir(), args.input_path)
    n_samples = args.train_samples

    if args.seed is not None:
        np.random.seed(args.seed)

    output_path = os.path.join(Path.output_root_dir(), "study", args.input_path)

    relevant_data = []
    dict_path = os.path.join(Path.output_root_dir(), "new_results/edits/relevant_data.csv")
    with open(dict_path, "r") as f:
        reader = list(csv.DictReader(f))
        for row in reader:
            relevant_data.append(row)

    chosen_data = np.random.choice(relevant_data)
    query_class = chosen_data["query_class"]
    distractor_class = chosen_data["distractor_class"]

    print(f"chosen class: {query_class}")

    dataset = get_test_dataset(get_vis_transform(), return_image_only=True)

    counterfactuals = np.load(
        os.path.join(dirpath, "counterfactuals.npy"), allow_pickle=True
    ).item()

    query_indices = []
    distractor_indices = []

    # gather indices for query images, as well as indices for distractor images (the latter for the testing phase)
    matches_path = os.path.join(Path.output_root_dir(), "new_results/edits/matches.csv")
    with open(matches_path, "r") as f:
        reader = list(csv.DictReader(f))
        for row in reader:
            if (int(row["query_class"]) == query_class):
                query_indices.append(int(row["query_index"]))
            if (int(row["query_class"]) == distractor_class):
                distractor_indices.append(int(row["query_index"]))
                # note that after we have determined the training image pairs, we need to remove the distractor indices of those from this list

    # select ten query indices to be potentially selected for testing and ten query indices to be part of training
    (query_test, query_train) = np.random.choice(query_indices, (2, n_samples), replace=False)

    # distractor_train = [counterfactuals[i]["distractor_index"] for i in query_train]
    distractor_train = [] # TODO: fix this

    # don't reuse distractor images for testing that we use in training
    distractor_indices = [i for i in distractor_indices if i not in distractor_train]
    distractor_test = np.random.choice(distractor_indices, n_samples, replace=False)

    # 0: query (alpha), 1: distractor (bravo)
    test_choices = np.random.randint(2, size=10)
    print(f"test_choices: {test_choices}")
    # TODO: save these

    # generate the ten testing images
    for i in range(n_samples):
        index = -1
        if test_choices[i] == 0:
            index = query_test[i]
        else:
            index = distractor_test[i]
        
        img = dataset.__getitem__(index)
        img_path = os.path.join(output_path, f"test_{i}.png")
        os.makedirs(img_path, exist_ok=True)
        save_image(img, img_path)

if __name__ == "__main__":
    main()