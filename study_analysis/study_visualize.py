import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np

from study import Study

def visualize_responses(study: Study):
    x = np.arange(40) // 4
    y = np.tile(np.array([-1, -1/3, 1/3, 1]), 10)
    
    sizes = np.array([np.count_nonzero([r.main_testing[x[i]] for r in study.responses] == y[i]) for i in range(len(x))]) / len(study.responses) * 70

    fig, ax = plt.subplots()

    plt.suptitle(f"{study.name} (Accuracy)")

    truth_backgrounds = [Rectangle([i - 0.5, t * 0.5 - 0.5], 1, 1) for i, t in enumerate(study.truth)]
    ax.add_collection(PatchCollection(truth_backgrounds, color="lightgrey", zorder=-1, linewidth=0))

    ax.scatter(x, y, s=sizes)
    plt.xticks(np.arange(10))
    plt.yticks([-1, -1/3, 1/3, 1], labels=['Beta (Certain)', 'Beta (Uncertain)', 'Alpha (Uncertain)', 'Alpha (Certain)'])

    plt.show()

def visualize_main_results(study: Study, show_individual: bool):
    valid_study = study.create_valid_responses()
    print(f"Number of valid responses: {len(valid_study.responses)} out of {len(study.responses)}")

    visualize_responses(study)
    visualize_responses(valid_study)

    if show_individual:
        for i, r in enumerate(study.responses):
            # visualize_responses(study.create_individual_response(i, f"{study.name}_{i} ({'Valid' if Study.has_valid_initial_test(r) else 'Not Valid'})"))
            visualize_responses(study.create_individual_response(i, f"{study.name}_{i} ({'Valid' if r.has_valid_initial_test else 'Not Valid'})"))
            for j, e in enumerate(r.main_testing):
                print(f"{j}: {e}")
            print(r["mental_model"])

def visualize_familiarity(study: Study):
    x = [r.familiarity for r in study.responses]
    y = [r.average_accuracy() for r in study.responses]

    fig, ax = plt.subplots()

    ax.scatter(x, y)
    plt.suptitle(f"{study.name} (Familiarity)")

    plt.xticks(np.arange(1, 6))
    plt.xlim(0.8, 5.2)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.ylim(-0.05, 1.05)

    plt.show()

def visualize_familiarity_correlation(studies: list[Study]):
    x = np.tile(np.arange(1, 6), 2)
    y = np.concatenate((np.full(5, -1), np.full(5, 1)))

    sizes = np.array([np.zeros(10) for _ in range(3)])

    num_responses = 0

    for s in studies:
        for r in s.responses:
            num_responses += 1
            for i, b in enumerate(r.intro_responses):
                """if i == 2:
                    print(b)"""
                sizes[i, (r.familiarity - 1) + 5 * int(b)] += 1

    sizes *= 70 / num_responses

    fig, axs = plt.subplots(3, constrained_layout=True)
    # fig.tight_layout()

    for i, ax in enumerate(axs):
        ax.set_title(f"Familiarity for Question {i + 1}")
        ax.set_xlim(0.8, 5.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xticks(np.arange(1, 6))
        ax.set_yticks([-1, 1])
        ax.set_yticklabels(['Incorrect', 'Correct'])

        ax.scatter(x, y, sizes=sizes[i])



    print(sizes)