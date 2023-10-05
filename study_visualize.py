import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np

def visualize_responses(responses, truth, study_name):
    x = np.arange(40) // 4
    y = np.tile(np.array([-1, -1/3, 1/3, 1]), 10)
    
    sizes = np.array([np.count_nonzero([r["main_responses"][x[i]] for r in responses] == y[i]) for i in range(len(x))]) / len(responses) * 70

    fig, ax = plt.subplots()

    plt.suptitle(study_name)

    truth_backgrounds = [Rectangle([i - 0.5, t * 0.5 - 0.5], 1, 1) for i, t in enumerate(truth)]
    ax.add_collection(PatchCollection(truth_backgrounds, color="lightgrey", zorder=-1, linewidth=0))

    ax.scatter(x, y, s=sizes)
    plt.yticks([-1, -1/3, 1/3, 1], labels=['Beta (Certain)', 'Beta (Uncertain)', 'Alpha (Uncertain)', 'Alpha (Certain)'])
    plt.xticks(np.arange(10))

    plt.show()

def has_valid_intro(response):
    return all(i == 0 for i in response["initial_responses"])

def visualize_main_results(responses, truth, study_name):
    valid_responses = [r for r in responses if has_valid_intro(r)]
    print(f"Number of valid responses: {len(valid_responses)} out of {len(responses)}")

    visualize_responses(responses, truth, study_name)
    visualize_responses(valid_responses, truth, f"{study_name}_valid_intro")

    for i, r in enumerate(responses):
        visualize_responses([r], truth, f"{study_name}_{i} ({'Valid' if has_valid_intro(r) else 'Not Valid'})")
        for j, e in enumerate(r["main_explanations"]):
            print(f"{j}: {e}")
        print(r["mental_model"])