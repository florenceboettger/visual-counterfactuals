import argparse
import os
import csv
import time

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
    n_samples = args.samples
    seed = args.seed
    if seed is None:
        seed = int(time.time())
    n_pix = 7

    np.random.seed(seed)

    print(f"chosen seed: {seed}")

    output_path = os.path.join(Path.output_root_dir(), "study", str(seed), args.input_path)
    os.makedirs(output_path, exist_ok=True)

    relevant_data = []
    dict_path = os.path.join(Path.output_root_dir(), "new_results/edits/relevant_data.csv")
    with open(dict_path, "r") as f:
        reader = list(csv.DictReader(f))
        for row in reader:
            relevant_data.append(row)

    chosen_data = np.random.choice(relevant_data)
    query_class = int(chosen_data["query_class"])
    distractor_class = int(chosen_data["distractor_class"])

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
                # note that after we have determined the training image pairs, we later need to remove the distractor indices of those from this list

    print(f"query_indices: {query_indices}")
    print(f"distractor_indices: {distractor_indices}")
    # select ten query indices to be potentially selected for testing and ten query indices to be part of training
    (query_test, query_train) = np.random.choice(query_indices, (2, n_samples), replace=False)

    distractor_train = []
    for i in query_train:
        edits = counterfactuals[i]["edits"]
        cell_index_distractor = edits[0][1]
        distractor_indices = counterfactuals[i]["distractor_index"]
        distractor_train.append(int(distractor_indices[cell_index_distractor // (n_pix**2)]))

    print(f"distractor_train: {distractor_train}")

    # don't reuse distractor images for testing that we use in training
    distractor_indices = [i for i in distractor_indices if i not in distractor_train]
    distractor_test = np.random.choice(distractor_indices, n_samples, replace=False)
    
    print(f"distractor_indices (new): {distractor_indices}")
    print(f"distractor_test: {distractor_test}")

    # 0: query (alpha), 1: distractor (beta)
    test_choices = np.random.randint(2, size=10)
    print(f"test_choices: {test_choices}")
    # TODO: save all this somewhere

    with open(os.path.join(output_path, "answers.csv", newline=''), "w") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "n_samples", "query_class", "distractor_class", "query_test", "query_train", "distractor_test", "distractor_train", "test_choices"])
        writer.writerow({
            "seed": seed,
            "n_samples": n_samples,
            "query_class": query_class,
            "distractor_class": distractor_class,
            "query_test": query_test,
            "query_train": query_train,
            "distractor_test": distractor_test,
            "distractor_train": distractor_train,
            "test_choices": test_choices
        })

    # generate the ten testing images
    for i in range(n_samples):
        index = -1
        if test_choices[i] == 0:
            index = query_test[i]
        else:
            index = distractor_test[i]
        
        img = dataset.__getitem__(index)
        img_path = os.path.join(output_path, f"test_{i}.png")
        save_image(img, img_path)

    # generate the ten training images
    for i in range(n_samples):
        query_index = query_train[i]
        distractor_index = distractor_train[i]

        query_img = dataset.__getitem__(query_index)
        distractor_img = dataset.__getitem__(distractor_index)
        
        edit = counterfactuals[query_index]["edits"][0]
        
        row_index_query = edit[0] // n_pix
        col_index_query = edit[0] % n_pix

        cell_index_distractor = edit[1] % (n_pix**2)        
        row_index_distractor = cell_index_distractor // n_pix
        col_index_distractor = cell_index_distractor % n_pix
        
        img_path = os.path.join(output_path, f"train_{i}.png")
        
        visualize_edit(query_img, col_index_query, row_index_query,
                       distractor_img, col_index_distractor, row_index_distractor,
                       n_pix, img_path, blur=True)

if __name__ == "__main__":
    main()