# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import csv

import model.auxiliary_model as auxiliary_model
import numpy as np
import optuna
import torch
import yaml

from explainer.counterfactuals import compute_counterfactual
from explainer.eval import compute_eval_metrics
from explainer.utils import get_query_distractor_pairs, process_dataset
from tqdm import tqdm
from utils.common_config import (
    get_imagenet_test_transform,
    get_model,
    get_test_dataloader,
    get_test_dataset,
    get_test_transform,
)
from utils.path import Path

parser = argparse.ArgumentParser(description="Generate counterfactual explanations")
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--index", type=str, required=True)
parser.add_argument("--mode", choices=["multiplicative", "additive"])
parser.add_argument("--lambd", type=float)
parser.add_argument("--lambd2", type=float)
parser.add_argument("--max_dist", type=float)
parser.add_argument("--parts_type", choices=["full", "minimize_head"])
parser.add_argument("--train", action="store_true")


def main():    
    assert torch.cuda.is_available()
    args = parser.parse_args()

    if args.train:
        study = optuna.create_study(
            study_name="optimize_counterfactuals_initial2",
            directions=["maximize", "minimize"],
            storage="sqlite:///optimize_counterfactuals_full.db",
            load_if_exists=True,
        )
        study.optimize(
            optimize_counterfactuals,
            n_trials=50,
            n_jobs=1,            
            callbacks=[optuna.study.MaxTrialsCallback(400, states=(optuna.trial.TrialState.COMPLETE,))],
        )
    else:        
        explain_counterfactuals(
            config_path=args.config_path,
            index=args.index,
            mode=args.mode or "additive",
            lambd=args.lambd or 0,
            lambd2=args.lambd2 or 0,
            max_dist=args.max_dist or 0,
            parts_type=args.parts_type or "full",
    )


def optimize_counterfactuals(trial):
    return explain_counterfactuals(
        config_path="visual-counterfactuals/counterfactuals/configs/counterfactuals/counterfactuals_ours_cub_vgg16.yaml",
        index=f"optimize_counterfactuals_initial2_{trial.number}",
        mode="additive",
        lambd=trial.suggest_float("lambd", 0.0, 2.0),
        lambd2=trial.suggest_float("lambd2", 0.0, 10.0),
        max_dist=trial.suggest_float("max_dist", 0.0, 3.0),
        parts_type=trial.suggest_categorical("parts_type", ["full", "minimize_head"]),
    )


def explain_counterfactuals(config_path, index, mode, lambd, lambd2, max_dist, parts_type):
    print(f"Beginning counterfactual search {index}, mode={mode}, lambd={lambd}, lambd2={lambd2}, max_dist={max_dist}, parts_type={parts_type}")
    # parse args
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    dirpath = os.path.join(Path.output_root_dir(), "batch", f"{index}")
    os.makedirs(dirpath, exist_ok=True)

    # create dataset
    dataset = get_test_dataset(transform=get_test_transform(), max_dist=max_dist, parts_type=parts_type)
    dataloader = get_test_dataloader(config, dataset)

    # device
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    # load classifier
    print("Load classification model weights")
    model = get_model(config)
    model_path = os.path.join(
        Path.output_root_dir(),
        config["counterfactuals_kwargs"]["model"],
    )
    state_dict = torch.load(model_path)["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key[len("model.") :]] = state_dict[key]
        del state_dict[key]
    model.load_state_dict(state_dict, strict=True)

    # process dataset
    print("Pre-compute classifier predictions")
    result = process_dataset(model, dataloader, device)
    features = result["features"]
    preds = result["preds"].numpy()
    targets = result["targets"].numpy()
    parts = result["parts"]
    print("Top-1 accuracy: {:.2f}".format(100 * result["top1"]))

    # compute query-distractor pairs
    print("Pre-compute query-distractor pairs")
    query_distractor_pairs = get_query_distractor_pairs(
        dataset,
        confusion_matrix=result["confusion_matrix"],
        max_num_distractors=config["counterfactuals_kwargs"][
            "max_num_distractors"
        ],  # noqa
    )

    # get classifier head
    classifier_head = model.get_classifier_head()
    classifier_head = torch.nn.DataParallel(classifier_head.cuda())
    classifier_head.eval()

    # auxiliary features for soft constraint
    if config["counterfactuals_kwargs"]["apply_soft_constraint"]:
        print("Pre-compute auxiliary features for soft constraint")
        aux_model, aux_dim, n_pix = auxiliary_model.get_auxiliary_model()
        aux_transform = get_imagenet_test_transform()
        aux_dataset = get_test_dataset(transform=aux_transform, return_image_only=True)
        aux_loader = get_test_dataloader(config, aux_dataset)

        auxiliary_features = auxiliary_model.process_dataset(
            aux_model,
            aux_dim,
            n_pix,
            aux_loader,
            device,
        ).numpy()
        use_auxiliary_features = True

    else:
        use_auxiliary_features = False

    # compute counterfactuals
    print("Compute counterfactuals")
    counterfactuals = {}

    for query_index in range(len(dataset)):
        if query_index not in query_distractor_pairs.keys():
            continue  # skips images that were classified incorrectly

        # gather query features
        query = features[query_index]  # dim x n_row x n_row
        query_parts = torch.permute(parts[query_index], (1, 2, 0)) # 7 x 7 x n_classes
        query_pred = preds[query_index]
        if query_pred != targets[query_index]:
            continue  # skip if query classified incorrect

        # gather distractor features
        distractor_target = query_distractor_pairs[query_index][
            "distractor_class"
        ]  # noqa
        distractor_index = query_distractor_pairs[query_index][
            "distractor_index"
        ]  # noqa
        if isinstance(distractor_index, int):
            if preds[distractor_index] != distractor_target:
                continue  # skip if distractor classified is incorrect
            distractor_index = [distractor_index]

        else:  # list
            distractor_index = [
                jj for jj in distractor_index if preds[jj] == distractor_target
            ]
            if len(distractor_index) == 0:
                continue  # skip if no distractors classified correct

        distractor = torch.stack([features[jj] for jj in distractor_index], dim=0)
        distractor_parts = torch.stack([parts[jj] for jj in distractor_index], dim=0).permute(0, 2, 3, 1) # N x 7 x 7 x n_classes

        # soft constraint uses auxiliary features
        if use_auxiliary_features:
            query_aux_features = torch.from_numpy(
                auxiliary_features[query_index]
            )  # aux_dim x n_row x n_row
            distractor_aux_features = torch.stack(
                [torch.from_numpy(auxiliary_features[jj]) for jj in distractor_index],
                dim=0,
            )  # n x aux_dim x n_row x n_row

        else:
            query_aux_features = None
            distractor_aux_features = None

        # compute counterfactual
        # try:
        list_of_edits = compute_counterfactual(
            query=query,
            distractor=distractor,
            classification_head=classifier_head,
            distractor_class=distractor_target,
            query_aux_features=query_aux_features,
            distractor_aux_features=distractor_aux_features,
            lambd=lambd,
            lambd2=lambd2,
            temperature=config["counterfactuals_kwargs"]["temperature"],
            topk=config["counterfactuals_kwargs"]["topk"]
            if "topk" in config["counterfactuals_kwargs"].keys()
            else None,
            query_parts = query_parts,
            distractor_parts = distractor_parts,
            mode=mode,
        )

        """except BaseException:
            print("warning - no counterfactual @ index {}".format(query_index))
            continue"""

        counterfactuals[query_index] = {
            "query_index": query_index,
            "distractor_index": distractor_index,
            "query_target": query_pred,
            "distractor_target": distractor_target,
            "edits": list_of_edits,
        }

    # save result
    np.save(os.path.join(dirpath, "counterfactuals.npy"), counterfactuals)

    # evaluation
    print("Generated {} counterfactual explanations".format(len(counterfactuals)))
    average_num_edits = np.mean([len(res["edits"]) for res in counterfactuals.values()])
    print("Average number of edits is {:.2f}".format(average_num_edits))

    result = compute_eval_metrics(
        counterfactuals,
        dataset=get_test_dataset(transform=get_test_transform(), max_dist=0, parts_type="full"),
    )

    print("Eval results single edit: {}".format(result["single_edit"]))
    print("Eval results all edits: {}".format(result["all_edit"]))

    result_path = os.path.join(Path.output_root_dir(), "new_results", f"{index}.csv")

    if index is not None:
        with open(result_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "mode", "lambd", "lambd2", "max_dist", "parts_type", "avg_edits", "eval_single", "eval_all"])
            writer.writeheader()
            writer.writerow({
                "id": index,
                "mode": mode,
                "lambd": lambd,
                "lambd2": lambd2,
                "max_dist": max_dist,
                "parts_type": parts_type,
                "avg_edits": average_num_edits,
                "eval_single": result["single_edit"],
                "eval_all": result["all_edit"],
            })

    return result["all_edit"]["Same-KP"], average_num_edits
        

if __name__ == "__main__":
    main()
