import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import stats
from functools import reduce
import numpy as np

def plot_study(study, name, is_resnet=False, print_others=True, print_pareto=True, map=lambda t: 1.0):
    trials = [t for t in study.trials if t.values is not None and (t.number > 5 or not is_resnet)]
    if print_others:
        plt.scatter([t.values[0] for t in trials], [t.values[1] for t in trials], c=[0.8 * map(t) for t in trials], cmap='Blues', linewidths=1.0, edgecolors='#000000', norm=Normalize(0.0, 1.0))
    if print_pareto:
        plt.scatter([t.values[0] for t in study.best_trials], [t.values[1] for t in study.best_trials], c=[0.8 * map(t) for t in study.best_trials], cmap='Reds', linewidths=1.0, edgecolors='#000000', norm=Normalize(0.0, 1.0))
    plt.xlabel('KP')
    plt.ylabel('Edits')
    plt.savefig(f"plots/{name}.png", dpi=500)
    plt.show()

def study_spearman(study, x_names, y_length):
    string_map = {
        "binary": 0,
        "distance": 1,
        "full": 0,
        "minimize_head": 1,
        "additive": 0,
        "multiplicative": 1,
    }
    x = [[string_map[t.params[x_name]] if isinstance(t.params[x_name], str) else t.params[x_name] for x_name in x_names] for t in study.trials if t.values is not None]
    y = [[t.values[y_index] for y_index in range(y_length)] for t in study.trials if t.values is not None]
    return stats.spearmanr(x, y)

def evaluate_results_spearman(results):
    relevant_attributes = [
        "avg_edits",
        "eval_single_near",
        "eval_single_same",
        "eval_all_near",
        "eval_all_same",
    ]
    x = [[float(r[att]) for att in relevant_attributes] for r in results]
    spearman, pvalue =  stats.spearmanr(x)
    for i, att in enumerate(relevant_attributes):
        corr_string = reduce(lambda a, b: f"{a}, {b}", spearman[i])
        print(f"{att} correlation is {corr_string}")

def evaluate_results_pearson(results):
    relevant_attributes = [
        "avg_edits",
        "eval_single_near",
        "eval_single_same",
        "eval_all_near",
        "eval_all_same",
    ]
    x = [[float(r[att]) for att in relevant_attributes] for r in results]
    pearson =  np.corrcoef(x, rowvar=False)
    for i, att in enumerate(relevant_attributes):
        corr_string = reduce(lambda a, b: f"{a}, {b}", pearson[i])
        print(f"{att} correlation is {corr_string}")

def evaluate_results_average(results):
    relevant_attributes = [
        "eval_single_near",
        "eval_single_same",
        "eval_all_near",
        "eval_all_same",
    ]
    x = [[float(r[att]) for att in relevant_attributes] for r in results]
    avg = np.average(x, axis=0)
    for i, att in enumerate(relevant_attributes):
        print(f"{att} average is {avg[i]}")

def evaluate_results_median(results):
    relevant_attributes = [
        "eval_single_near",
        "eval_single_same",
        "eval_all_near",
        "eval_all_same",
    ]
    x = [[float(r[att]) for att in relevant_attributes] for r in results]
    avg = np.median(x, axis=0)
    for i, att in enumerate(relevant_attributes):
        print(f"{att} median is {avg[i]}")