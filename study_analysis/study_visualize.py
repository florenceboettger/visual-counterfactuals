import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np

from study import Study

def visualize_responses(study):
    x = np.arange(40) // 4
    y = np.tile(np.array([-1, -1/3, 1/3, 1]), 10)
    
    sizes = np.array([np.count_nonzero([r["main_responses"][x[i]] for r in study.responses] == y[i]) for i in range(len(x))]) / len(study.responses) * 70

    fig, ax = plt.subplots()

    plt.suptitle(study.name)

    truth_backgrounds = [Rectangle([i - 0.5, t * 0.5 - 0.5], 1, 1) for i, t in enumerate(study.truth)]
    ax.add_collection(PatchCollection(truth_backgrounds, color="lightgrey", zorder=-1, linewidth=0))

    ax.scatter(x, y, s=sizes)
    plt.yticks([-1, -1/3, 1/3, 1], labels=['Beta (Certain)', 'Beta (Uncertain)', 'Alpha (Uncertain)', 'Alpha (Certain)'])
    plt.xticks(np.arange(10))

    plt.show()

def visualize_main_results(study, show_individual):
    valid_study = study.create_valid_responses()
    print(f"Number of valid responses: {len(valid_study.responses)} out of {len(study.responses)}")

    visualize_responses(study)
    visualize_responses(valid_study)

    if show_individual:
        for i, r in enumerate(study.responses):
            visualize_responses(study.create_individual_response(i, f"{study.name}_{i} ({'Valid' if Study.has_valid_intro(r) else 'Not Valid'})"))
            for j, e in enumerate(r["main_explanations"]):
                print(f"{j}: {e}")
            print(r["mental_model"])