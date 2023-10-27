import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import stats
from scipy.interpolate import interp1d
from functools import reduce
import numpy as np

def plot_study(study, name, is_resnet=False, print_others=True, print_pareto=True, map=lambda t: 0.8, other_study=None, print_colorbar=True, param_name="", split_simplify=False, labels=[""]):
    trials = [t for t in study.trials if t.values is not None and (t.number > 5 or not is_resnet)]
    norm = Normalize(0.0, np.ceil(max([map(t) for t in trials])))
    scatter = None
    if print_others:
        scatter = plt.scatter([t.values[0] for t in trials], [t.values[1] for t in trials], c=[map(t) for t in trials], cmap='Blues', linewidths=1.0, edgecolors='#000000', norm=norm, label=labels[0])
    if print_pareto:
        plt.scatter([t.values[0] for t in study.best_trials], [t.values[1] for t in study.best_trials], c=[map(t) for t in study.best_trials], cmap='Reds', linewidths=1.0, edgecolors='#000000', norm=norm, label="Pareto Front")
        plt.legend(loc="upper left")
    if other_study is not None:
        other_trials = [t for t in other_study.trials if t.values is not None]
        plt.scatter([t.values[0] for t in other_trials], [t.values[1] for t in other_trials], c=[map(t) for t in other_trials], cmap='Oranges', linewidths=1.0, edgecolors='#000000', norm=norm, label=labels[1])
        plt.legend()
    if split_simplify:
        simplify_trials = [t for t in trials if t.params["parts_type"] == "minimize_head"]
        full_trials = [t for t in trials if t.params["parts_type"] != "minimize_head"]
        plt.scatter([t.values[0] for t in simplify_trials], [t.values[1] for t in simplify_trials], c=[0.8] * len(simplify_trials), cmap='Blues', norm=Normalize(0.0, 1.0), linewidths=1.0, edgecolors="black", label="Simplify")
        plt.scatter([t.values[0] for t in full_trials], [t.values[1] for t in full_trials], c="white", linewidths=1.0, edgecolors="black", label="Full")
        plt.legend(loc="upper left")
    plt.xlabel('KP')
    plt.ylabel('Edits')
    if print_colorbar:
        cb = plt.colorbar(scatter)
        cb.set_label(param_name, rotation=0, labelpad=20, y=0.5)
    plt.savefig(f"plots/{name}.png", dpi=500, bbox_inches='tight', pad_inches=0)
    plt.savefig(f"plots/{name}.pdf", dpi=500, bbox_inches='tight', pad_inches=0)
    plt.show()

def get_study_values(study, hyperparams):
    string_map = {
        "binary": 0,
        "distance": 1,
        "full": 0,
        "minimize_head": 1,
        "additive": 0,
        "multiplicative": 1,
    }
    x = [[string_map[t.params[hyperparam]] if isinstance(t.params[hyperparam], str) else t.params[hyperparam] for hyperparam in hyperparams] for t in study.trials if t.values is not None]
    y = [[t.values[y_index] for y_index in range(len(t.values))] for t in study.trials if t.values is not None]
    return x, y

def print_correlation(correlation, pvalues, hyperparams, offset=0):
    for i in range(len(hyperparams)):
        x_name = hyperparams[i]
        print(f"{x_name} correlation is: {correlation[i][offset]} on KP, {correlation[i][offset + 1]} on edits.")
        print(f"{x_name} pvalue is: {pvalues[i][offset]} on KP, {pvalues[i][offset + 1]} on edits.")

def analyze_spearman(study, hyperparams):
    spearman = study_spearman(study, hyperparams)
    print_correlation(spearman.correlation, spearman.pvalue, hyperparams, len(hyperparams))


def study_spearman(study, hyperparams):
    x, y = get_study_values(study, hyperparams)
    return stats.spearmanr(x, y)

def get_pearson(x, y):
    pearson_results = [[stats.pearsonr(np.transpose(x)[i], np.transpose(y)[j]) for j in range(np.shape(y)[1])] for i in range(np.shape(x)[1])]
    pearson = [[cell[0] for cell in row] for row in pearson_results]
    pvalue = [[cell[1] for cell in row] for row in pearson_results]
    return pearson, pvalue

def analyze_pearson(study, hyperparams):
    correlation, pvalues = study_pearson(study, hyperparams)
    print_correlation(correlation, pvalues, hyperparams)

def study_pearson(study, hyperparams):
    x, y = get_study_values(study, hyperparams)
    return get_pearson(x, y)

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
        print(f"{att} Spearman correlation is {corr_string}")

    print("")

    for i, att in enumerate(relevant_attributes):        
        pvalue_string = reduce(lambda a, b: f"{a}, {b}", pvalue[i])
        print(f"{att} Spearman pvalue is {pvalue_string}")

def evaluate_results_pearson(results):
    relevant_attributes = [
        "avg_edits",
        "eval_single_near",
        "eval_single_same",
        "eval_all_near",
        "eval_all_same",
    ]
    x = [[float(r[att]) for att in relevant_attributes] for r in results]
    pearson, pvalue = get_pearson(x, x)
    
    for i, att in enumerate(relevant_attributes):
        corr_string = reduce(lambda a, b: f"{a}, {b}", pearson[i])
        print(f"{att} Pearson correlation is {corr_string}")

    print("")

    for i, att in enumerate(relevant_attributes):        
        pvalue_string = reduce(lambda a, b: f"{a}, {b}", pvalue[i])
        print(f"{att} Pearson pvalue is {pvalue_string}")

def evaluate_results_average(results):
    relevant_attributes = [
        "avg_edits",
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
        "avg_edits",
        "eval_single_near",
        "eval_single_same",
        "eval_all_near",
        "eval_all_same",
    ]
    x = [[float(r[att]) for att in relevant_attributes] for r in results]
    avg = np.median(x, axis=0)
    for i, att in enumerate(relevant_attributes):
        print(f"{att} median is {avg[i]}")

def evaluate_performance(results):
    runtimes = [float(r["time"]) for r in results]
    print(f"Total runtime is {np.sum(runtimes)} seconds.")
    print(f"Average runtime is {np.average(runtimes)} seconds.")
    print(f"Minimum runtime is {np.min(runtimes)} seconds.")
    print(f"Maximum runtime is {np.max(runtimes)} seconds.")
    print(f"Median runtime is {np.median(runtimes)} seconds.")

def performance_edits_correlation(results):    
    relevant_attributes = [
        "avg_edits",
        "time",
    ]
    x = [[float(r[att]) for att in relevant_attributes] for r in results]
    # pearson = np.corrcoef(x, rowvar=False)
    pearson, pearson_pvalue = stats.pearsonr(np.transpose(x)[0], np.transpose(x)[1])
    print(f"Pearson correlation is {pearson}")
    print(f"Pearson pvalue is {pearson_pvalue}")
    spearman, spearman_pvalue = stats.spearmanr(x)
    print(f"Spearman correlation is {spearman}")
    print(f"Spearman pvalue is {spearman_pvalue}")

num_iterations = 4411

def average_time_per_edit(results):
    time_per_edit = [float(r["time"]) / (float(r["avg_edits"]) * num_iterations) for r in results]
    average_time = np.average(time_per_edit)
    var_time = np.var(time_per_edit)
    print(f"Average time per edit is {average_time} seconds, variance is {var_time}.")

def plot_runtime(results, name):
    x = range(len(results))
    y = [float(r["time"]) for r in results]
    plt.plot(x, y)

    fit = np.polyfit(x, y, 1)
    p = np.poly1d(fit)

    # plt.plot(x, p(x), color="red", linestyle="--", linewidth=2)

    plt.xlabel('Iteration')
    plt.ylabel('Runtime in seconds')
    
    plt.savefig(f"plots/{name}.png", dpi=500, bbox_inches='tight', pad_inches=0)
    plt.savefig(f"plots/{name}.pdf", dpi=500, bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_runtime_per_edit(results, name):
    x = range(len(results))
    y = [float(r["time"]) / (float(r["avg_edits"]) * num_iterations) for r in results]
    plt.plot(x, y)

    fit = np.polyfit(x, y, 1)
    p = np.poly1d(fit)
    # plt.plot(x, p(x), color="red", linestyle="--", linewidth=2)

    plt.xlabel('Iteration')
    plt.ylabel('Runtime per edit in seconds')
    
    plt.savefig(f"plots/{name}.png", dpi=500, bbox_inches='tight', pad_inches=0)
    plt.savefig(f"plots/{name}.pdf", dpi=500, bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_runtime_edits(results, name):
    x = [float(r["avg_edits"]) for r in results]
    y = [float(r["time"]) for r in results]

    plt.scatter(x, y, linewidths=1.0, edgecolors='#000000')

    plt.xlabel('Edits')
    plt.ylabel('Runtime in seconds')
    
    plt.savefig(f"plots/{name}.png", dpi=500, bbox_inches='tight', pad_inches=0)
    plt.savefig(f"plots/{name}.pdf", dpi=500, bbox_inches='tight', pad_inches=0)
    plt.show()