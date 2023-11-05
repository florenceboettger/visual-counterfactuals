import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import numpy as np
import re
import scipy.stats as stats

from study import Study, Response, Referral

def visualize_responses(study: Study):
    x = (np.arange(40) // 4)
    y = np.tile(np.array([-1, -1/3, 1/3, 1]), 10)
    
    sizes = np.array([np.count_nonzero([r.main_testing[x[i]] for r in study.responses] == y[i]) for i in range(len(x))]) / len(study.responses) * 70
    colors = np.array([np.count_nonzero([r.main_testing[x[i]] for r in study.responses] == y[i]) for i in range(len(x))])

    x = x + 1

    fig, ax = plt.subplots()

    # ax.set_box_aspect(1)

    print(f"{study.name} (Accuracy)")

    truth_backgrounds = [Rectangle([i + 0.5, t * 0.5 - 0.5], 1, 1) for i, t in enumerate(study.truth)]
    ax.add_collection(PatchCollection(truth_backgrounds, color="lightgrey", zorder=-1, linewidth=0))

    ax.scatter(x, y, s=sizes, edgecolors="black")
    # ax.scatter(x, y, edgecolors="black", c=colors, cmap='Blues', norm=Normalize(0.0, max(colors)))
    plt.xticks(np.arange(1, 11))
    plt.yticks([-1, -1/3, 1/3, 1], labels=['Beta (Certain)', 'Beta (Uncertain)', 'Alpha (Uncertain)', 'Alpha (Certain)'])
    plt.xlabel("Question", fontsize="xx-large")
    # plt.ylabel("Answer")
    
    plt.savefig(f"../plots/user_study/accuracy_{study.name}.png", dpi=500, bbox_inches='tight', pad_inches=0)
    plt.savefig(f"../plots/user_study/accuracy_{study.name}.pdf", dpi=500, bbox_inches='tight', pad_inches=0)

    plt.show()

def visualize_main_results(study: Study, show_individual: bool):
    valid_study = study.create_valid_responses()
    print(f"Number of valid responses: {len(valid_study.responses)} out of {len(study.responses)}")

    visualize_responses(study)
    visualize_responses(valid_study)

    if show_individual:
        for i, r in enumerate(study.responses):
            visualize_responses(study.create_individual_response(i, f"{study.name}_{i} ({'Valid' if r.has_valid_initial_test else 'Not Valid'})"))
            for j, e in enumerate(r.main_explanations):
                print(f"{j}: {e}")
            print(r.mental_model)

def visualize_familiarity(study: Study):
    x = [r.familiarity for r in study.responses]
    y = [r.average_accuracy() for r in study.responses]

    sizes = np.array([np.count_nonzero([k == i and l == j for k, l in zip(x, y)]) for i, j in zip(x, y)]) * 25

    fig, ax = plt.subplots()

    ax.scatter(x, y, sizes=sizes)
    plt.suptitle(f"{study.name} (Familiarity)")

    plt.xticks(np.arange(1, 6))
    plt.xlim(0.8, 5.2)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.ylim(-0.05, 1.05)

    plt.show()

def visualize_familiarity_accuracy(study: Study, legend: str = None):
    x = range(1, 6)

    heights = []
    colors = ["#4dac26", "#b8e186", "#f1b6da", "#d01c8b"]
    
    for v in [1, 1/3, -1/3, -1]:
        heights.append([np.sum([np.count_nonzero([r.main_testing[i] * r.truth[i] == v and r.familiarity == f for r in study.responses]) for i in range(10)]) for f in range(1, 6)])

    print(heights)

    corr_x = [r.familiarity for r in study.responses]
    corr_y = [r.average_accuracy() for r in study.responses]

    spearman, pvalue = stats.spearmanr(corr_x, corr_y)

    print(f"Spearman correlation is {spearman}, pvalue = {pvalue}")

    for i, height in enumerate(heights):
        bottom = np.zeros(5)
        for j in range(i):
            bottom += heights[j]
        plt.bar(x, height, color=colors[i], bottom=bottom, edgecolor="black", width=0.75)

    max_height = max(np.sum(heights, axis=0))
    
    if legend:
        plt.legend(labels=["Correct (Certain)", "Correct (Uncertain)", "Incorrect (Uncertain)", "Incorrect (Certain)"], loc=legend, fontsize="x-large")

    plt.xlabel("Familiarity", fontsize="xx-large")
    plt.xticks(x)
    plt.ylabel("# Responses", fontsize="xx-large")

    plt.ylim(0, max_height + 1)

    plt.savefig(f"../plots/user_study/familiarity_accuracy_{study.name}.png", dpi=500, bbox_inches='tight', pad_inches=0)
    plt.savefig(f"../plots/user_study/familiarity_accuracy_{study.name}.pdf", dpi=500, bbox_inches='tight', pad_inches=0)

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
    
def visualize_familiarity_correlation_bar(studies: list[Study]):
    responses: list[Response] = []
    for s in studies:
        responses += s.responses
        
    x = range(1, 6)

    heights = [[[np.count_nonzero([r.intro_responses[i] == b and r.familiarity == f for r in responses]) for f in range(1, 6)] for b in [True, False]] for i in range(3)]

    print(heights)

    colors = ["#4dac26", "#d01c8b"]

    fig, axs = plt.subplots(3, constrained_layout=True, figsize=(6.4, 6.4), sharex=True)

    for i, ax in enumerate(axs):
        # ax.set_aspect(0.5)
        ax.set_title(f"Question {i + 1}")
        # ax.set_xlim(0.8, 5.2)
        # ax.set_ylim(0, len(responses) + 0.5)
        ax.set_xticks(range(1, 6))
        ax.set_yticks([0, 5, 10])
        if i == 2:
            ax.set_xlabel("Familiarity", fontsize="xx-large")

        ax.set_ylabel("# Responses", fontsize="xx-large")

        for j, h in enumerate(heights[i]):
            bottom = np.zeros(5)
            for k in range(j):
                bottom += heights[i][k]

            ax.bar(x, h, color=colors[j], bottom=bottom, edgecolor="black")        

        if i == 0:
            ax.legend(labels=["Correct", "Incorrect"], fontsize="x-large")
    
    plt.savefig(f"../plots/user_study/familiarity_correlation_bar.png", dpi=500, bbox_inches='tight', pad_inches=0)
    plt.savefig(f"../plots/user_study/familiarity_correlation_bar.pdf", dpi=500, bbox_inches='tight', pad_inches=0)

def visualize_familiarity_count(studies: list[Study]):
    responses: list[Response] = []
    for s in studies:
        print(f"Study {s.name} received {len(s.responses)} responses.")
        responses += s.responses

    print(f"Total responses received: {len(responses)}")
    
    heights = {}

    x = range(1, 6)

    types = [t.value for t in Referral if t != Referral.BIOLOGY]

    for t in Referral:
        if t != Referral.BIOLOGY:
            heights[t.value] = np.empty(0)
            for f in x:
                heights[t.value] = np.append(heights[t.value], np.count_nonzero([r.familiarity == f and r.referral_type == t for r in responses]))

    print(heights)

    colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    for i in range(len(types)):
        bottom = np.zeros(5)
        for j in range(i):
            bottom += heights[types[j]]
        plt.bar(x, heights[types[i]], color=colors[i], bottom=bottom, edgecolor="black")
    

    plt.legend(labels=types, fontsize="x-large")

    plt.xlabel("Familiarity", fontsize="xx-large")
    plt.ylabel("# Responses", fontsize="xx-large")
    
    plt.savefig(f"../plots/user_study/familiarity_count.png", dpi=500, bbox_inches='tight', pad_inches=0)
    plt.savefig(f"../plots/user_study/familiarity_count.pdf", dpi=500, bbox_inches='tight', pad_inches=0)

def count_answers(study: Study, query: str = "\S"):
    count = 0
    for r in study.responses:
        for e in r.main_explanations:
            if re.search(query, e, flags=re.IGNORECASE):
                count += 1
                break

    print(f"Study {study.name} has {count} matches for query {query}.")

def visualize_accuracy(study: Study, max_count, legend: str):
    colors = ["#4dac26", "#b8e186", "#f1b6da", "#d01c8b"]

    x = range(1, 11)
    heights = []

    for v in [1, 1/3, -1/3, -1]:
        heights.append([np.count_nonzero([r.main_testing[i] * r.truth[i] == v for r in study.responses]) for i in range(10)])

    for i, height in enumerate(heights):
        bottom = np.zeros(10)
        for j in range(i):
            bottom += heights[j]
        plt.bar(x, height, color=colors[i], bottom=bottom, edgecolor="black", width=0.75)

    if legend:
        plt.legend(labels=["Correct (Certain)", "Correct (Uncertain)", "Incorrect (Uncertain)", "Incorrect (Certain)"], loc=legend, fontsize="x-large")

    plt.xlabel("Question", fontsize="xx-large")
    plt.xticks(x)
    plt.ylabel("# Responses", fontsize="xx-large")
    plt.yticks(range(0, len(study.responses) + 1))

    plt.ylim(0, max_count + 0.5)

    plt.savefig(f"../plots/user_study/accuracy_bar_{study.name}.png", dpi=500, bbox_inches='tight', pad_inches=0)
    plt.savefig(f"../plots/user_study/accuracy_bar_{study.name}.pdf", dpi=500, bbox_inches='tight', pad_inches=0)

    plt.show()

class WordCount():
    word: str
    count: str

    def __init__(self, word, count):
        self.word = word
        self.count = count

def compare(str1: str, str2: str):
    if (str.lower(str1) == str.lower(str2)):
        return 1
    if re.search(str1, str2, flags=re.IGNORECASE):
        return 1
    if re.search(str2, str1, flags=re.IGNORECASE):
        return 2
    return 0

def compare_answers(responses: list[Response]):
    words: list[WordCount] = []
    for r in responses:
        answer_words: list[str] = []
        for answer in r.main_explanations:
            modified_answer = re.sub("colour", "color", answer, flags=re.IGNORECASE)
            modified_answer = re.sub("grey", "gray", modified_answer, flags=re.IGNORECASE)
            modified_answer = re.sub("schnabel", "bill", modified_answer, flags=re.IGNORECASE)
            answer_words += re.findall(r"\b\w{3,}\b", modified_answer, flags=re.IGNORECASE)

        answer_words = [str.lower(w) for w in answer_words if str.lower(w) not in ["the", "and"]]

        remove = set([])

        for w in answer_words:
            for v in answer_words:
                if re.search(w, v, flags=re.IGNORECASE) and w != v:
                    remove.add(v)#

        answer_words = set(answer_words)

        for w in remove:
            answer_words.remove(w)#

        remove = []
            
        for wc in words:
            for i, v in enumerate(answer_words):
                comp = compare(wc.word, v)
                if comp == 1:
                    wc.count += 1
                    remove.append(i)
                if comp == 2:
                    wc.count += 1
                    wc.word = str.lower(v)
                    remove.append(i)

        # print([f"{wc.word}, {wc.count}" for wc in words])

        for i, v in enumerate(answer_words):
            if not i in remove:
                words.append(WordCount(str.lower(v), 1))

        # print([f"{wc.word}, {wc.count}" for wc in words])

    sorted_words = sorted(words, key=lambda wc: wc.count, reverse=True)

    for wc in sorted_words:
        if wc.count > 1:
            print(f"{wc.word} appeared {wc.count} time(s).")