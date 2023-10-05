import csv
from pathlib import Path
from study_visualize import visualize_main_results

intro_truth = [
    "Ruby-throated Hummingbird",
    "Ivory Gull",
    "Hooded Oriole"
]

answer_map = {
    "Alpha (Certain)": 1,
    "Alpha (Uncertain)": 1/3,
    "Don't Know": 0,
    "Beta (Uncertain)": -1/3,
    "Beta (Certain)": -1
}

truth_map = {
    "Alpha": 1,
    "Beta": -1
}

def analyze_response(response):
    intro_responses = []
    for i in range(len(intro_truth)):
        intro_responses.append(intro_truth[i] == response[f"initial_{i}"])

    initial_responses = []
    main_responses = []
    main_explanations = []
    for i in range(10):
        initial_responses.append(answer_map[response[f"training_initial_{i}"]])
        main_responses.append(answer_map[response[f"training_later_{i}"]])
        main_explanations.append(response[f"training_explanation_{i}"])

    return {
        "intro_responses": intro_responses,
        "initial_responses": initial_responses,
        "main_responses": main_responses,
        "main_explanations": main_explanations,
        "mental_model": response["mental_model"]
    }


def load_studies():
    results_path = Path.cwd() / 'study_data'
    study_dirs = [str(results_path / f) for f in results_path.iterdir() if (results_path / f).is_dir()]

    for dir in study_dirs:
        dir_path = Path(dir)
        study_name = dir_path.stem

        study_responses = []
        with (dir_path / 'main.csv').open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                study_responses.append(row)

        correct_answers = None
        with (dir_path / 'answers.csv').open('r') as f:
            reader = csv.DictReader(f)
            correct_answers = reader.__next__()
        
        assert correct_answers["seed"] == study_name.split('_')[0]

        main_truth = []
        for i in range(10):
            main_truth.append(truth_map[correct_answers[f"test_choice_{i}"]])

        response_infos = []
        for response in study_responses:
            response_infos.append(analyze_response(response))

        visualize_main_results(response_infos, main_truth, study_name)
