from pathlib import Path
import csv
import numpy as np

class Study():
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

    @staticmethod
    def from_directory(dir):
        dir_path = Path(dir)
        name = dir_path.stem
        raw_responses = []
        responses = []
        truth = []        

        with (dir_path / 'main.csv').open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_responses.append(row)

        correct_answers = None
        with (dir_path / 'answers.csv').open('r') as f:
            reader = csv.DictReader(f)
            correct_answers = reader.__next__()
        
        assert correct_answers["seed"] == name.split('_')[0]

        for i in range(10):
            truth.append(Study.truth_map[correct_answers[f"test_choice_{i}"]])

        for response in raw_responses:
            responses.append(Study._filter_response(response))

        return Study(truth, name, responses)

    def __init__(self, truth, name, responses):
        self.truth = truth
        self.name = name
        self.responses = responses

    def create_individual_response(self, index=0, name=None):
        name = name or f"{self.name}_{index}"
        return Study(self.truth, name, [self.responses[index]])
    
    def create_valid_responses(self, name=None):
        name = name or f"{self.name} (Valid Intro)"
        return Study(self.truth, name, [r for r in self.responses if Study.has_valid_intro(r)])

    def evaluate(self):
        self._evaluate_accuracy()

    def _evaluate_accuracy(self):
        accuracies = []
        for i, response in enumerate(self.responses):
            accuracy = np.count_nonzero(np.array([r * t for r, t in zip(response["main_responses"], self.truth)]) > 0) / len(response["main_responses"])
            print(f"Response {i} of study {self.name} has an accuracy of {accuracy}")
            
            accuracies.append(accuracy)
        
        print(f"Average accuracy for study {self.name} is {np.average(accuracies)}")
        return
    

    @staticmethod
    def _filter_response(response):
        intro_responses = []
        for i in range(len(Study.intro_truth)):
            intro_responses.append(Study.intro_truth[i] == response[f"initial_{i}"])

        initial_responses = []
        main_responses = []
        main_explanations = []
        for i in range(10):
            initial_responses.append(Study.answer_map[response[f"testing_initial_{i}"]])
            main_responses.append(Study.answer_map[response[f"testing_later_{i}"]])
            main_explanations.append(response[f"testing_explanation_{i}"])

        return {
            "intro_responses": intro_responses,
            "initial_responses": initial_responses,
            "main_responses": main_responses,
            "main_explanations": main_explanations,
            "mental_model": response["mental_model"]
        }
    
    @staticmethod
    def has_valid_intro(response):
        return all(i == 0 for i in response["initial_responses"])