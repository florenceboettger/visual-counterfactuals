import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import numpy as np
import re
import scipy.stats as stats

from study import Study, Response, Referral

colors_qualitative = ["#1b9e77", "#d95f02", "#7570b3"]
colors_diverging_4 = ["#4dac26", "#b8e186", "#f1b6da", "#d01c8b"]
colors_diverging_2 = [colors_diverging_4[0], colors_diverging_4[3]]

def set_dims():    
    plt.figure(figsize=(6.4, 4.8), dpi=100)

def visualize_familiarity_accuracy(study: Study, legend: str = None):
    set_dims()

    x = range(1, 6)

    heights = []
    
    for v in [1, 1/3, -1/3, -1]:
        heights.append([np.sum([np.count_nonzero([r.main_testing[i] * r.truth[i] == v and r.familiarity == f for r in study.responses]) for i in range(10)]) for f in range(1, 6)])

    corr_x = [r.familiarity for r in study.responses]
    corr_y = [r.average_accuracy() for r in study.responses]

    spearman, pvalue = stats.spearmanr(corr_x, corr_y)

    print(f"Spearman correlation is {spearman}, pvalue = {pvalue}")
    bars = []
    labels = ["Correct (Certain)", "Correct (Uncertain)", "Incorrect (Uncertain)", "Incorrect (Certain)"]

    for i, height in enumerate(heights):
        bottom = np.zeros(5)
        for j in range(i):
            bottom += heights[j]
        bars.append(plt.bar(x, height, color=colors_diverging_4[i], bottom=bottom, edgecolor="black", width=0.75, label=labels[i]))
    bars.reverse()

    max_height = max(np.sum(heights, axis=0))
    
    if legend:
        plt.legend(handles=bars, loc=legend, fontsize="x-large")

    plt.xlabel("Familiarity", fontsize="x-large")
    plt.xticks(x)
    plt.ylabel("# Responses", fontsize="x-large")

    plt.ylim(0, 31)

    plt.savefig(f"../plots/user_study/familiarity_accuracy_{study.name}.png", dpi=500, bbox_inches='tight', pad_inches=0)
    plt.savefig(f"../plots/user_study/familiarity_accuracy_{study.name}.pdf", dpi=500, bbox_inches='tight', pad_inches=0)

    plt.show()

    return spearman, pvalue
    
def visualize_familiarity_correlation_bar(studies: list[Study]):
    responses: list[Response] = []
    for s in studies:
        responses += s.responses
        
    x = range(1, 6)

    heights = [[[np.count_nonzero([r.intro_responses[i] == b and r.familiarity == f for r in responses]) for f in range(1, 6)] for b in [True, False]] for i in range(3)]

    print(heights)

    corr_x = [r.familiarity for r in responses]
    corr_y = [r.intro_responses.count(True)/len(r.intro_responses) for r in responses]

    fig, axs = plt.subplots(3, constrained_layout=True, figsize=(6.4, 6.4), sharex=True)
    fig.dpi = 100
    fig.set_figwidth(6.4)
    fig.set_figheight(7.2)

    for i, ax in enumerate(axs):
        # ax.set_aspect(0.5)
        ax.set_title(f"Question {i + 1}", fontsize="x-large")
        # ax.set_xlim(0.8, 5.2)
        # ax.set_ylim(0, len(responses) + 0.5)
        ax.set_xticks(range(1, 6))
        ax.set_yticks([0, 5, 10])
        if i == 2:
            ax.set_xlabel("Familiarity", fontsize="x-large")

        ax.set_ylabel("# Responses", fontsize="x-large")

        bars = []
        labels = ["Correct", "Incorrect"]

        for j, h in enumerate(heights[i]):
            bottom = np.zeros(5)
            for k in range(j):
                bottom += heights[i][k]

            bars.append(ax.bar(x, h, color=colors_diverging_2[j], bottom=bottom, edgecolor="black", label=labels[j]))   
        bars.reverse()     

        if i == 0:
            ax.legend(handles=bars, fontsize="x-large")
    
    plt.savefig(f"../plots/user_study/familiarity_correlation_bar.png", dpi=500, bbox_inches='tight', pad_inches=0)
    plt.savefig(f"../plots/user_study/familiarity_correlation_bar.pdf", dpi=500, bbox_inches='tight', pad_inches=0)

def visualize_familiarity_count(studies: list[Study]):
    set_dims()
    
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
    bars = []

    for i in range(len(types)):
        bottom = np.zeros(5)
        for j in range(i):
            bottom += heights[types[j]]
        bars.append(plt.bar(x, heights[types[i]], color=colors_qualitative[i], bottom=bottom, edgecolor="black", label="Personal Contact" if types[i] == "Other" else types[i]))
    bars.reverse()

    plt.legend(handles=bars, fontsize="x-large")

    plt.xlabel("Familiarity", fontsize="x-large")
    plt.ylabel("# Responses", fontsize="x-large")
    
    plt.savefig(f"../plots/user_study/familiarity_count.png", dpi=500, bbox_inches='tight', pad_inches=0)
    plt.savefig(f"../plots/user_study/familiarity_count.pdf", dpi=500, bbox_inches='tight', pad_inches=0)

def count_answers(study: Study, query: str = "\S"):
    count = 0
    for r in study.responses:
        if re.search(query, r.mental_model, flags=re.IGNORECASE):
            count += 1
            continue
        for e in r.main_explanations:
            if re.search(query, e, flags=re.IGNORECASE):
                count += 1
                break

    print(f"Study {study.name} has {count} matches for query {query}.")

def count_mental_models(study: Study, query: str = "\S"):
    count = 0
    for r in study.responses:
        if re.search(query, r.mental_model, flags=re.IGNORECASE):
            count += 1

    print(f"Study {study.name} has {count} matches in the mental models for query {query}.")

def visualize_accuracy(study: Study, max_count, legend: str):
    set_dims()

    x = range(1, 11)
    heights = []

    for v in [1, 1/3, -1/3, -1]:
        heights.append([np.count_nonzero([r.main_testing[i] * r.truth[i] == v for r in study.responses]) for i in range(10)])

    bars = []
    labels = ["Correct (Certain)", "Correct (Uncertain)", "Incorrect (Uncertain)", "Incorrect (Certain)"]

    for i, height in enumerate(heights):
        bottom = np.zeros(10)
        for j in range(i):
            bottom += heights[j]
        bars.append(plt.bar(x, height, color=colors_diverging_4[i], bottom=bottom, edgecolor="black", width=0.75, label=labels[i]))
    bars.reverse()

    if legend:
        plt.legend(handles=bars, loc=legend, fontsize="x-large")

    plt.xlabel("Question", fontsize="x-large")
    plt.xticks(x)
    plt.ylabel("# Responses", fontsize="x-large")
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
        for answer in r.main_explanations + [r.mental_model]:
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