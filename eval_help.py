import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import stats

def plot_study(study, name, is_resnet=False, print_others=True, print_pareto=True, map=lambda t: .8):
    trials = [t for t in study.trials if t.values is not None and (t.number > 5 or not is_resnet)]
    if print_others:
        plt.scatter([t.values[0] for t in trials], [t.values[1] for t in trials], c=[map(t) for t in trials], cmap='Blues', linewidths=1.0, edgecolors='#000000', norm=Normalize(0.0, 1.0))
    if print_pareto:
        plt.scatter([t.values[0] for t in study.best_trials], [t.values[1] for t in study.best_trials], c=[map(t) for t in study.best_trials], cmap='Reds', linewidths=1.0, edgecolors='#000000', norm=Normalize(0.0, 1.0))
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