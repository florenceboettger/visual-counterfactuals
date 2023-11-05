from pathlib import Path
import csv
import numpy as np
from enum import Enum
    
class Referral(Enum):
    UNIVERSITY = "University E-mail"
    BIOLOGY = "Biology E-mail"
    BIRD_FORUMS = "Bird Forums"
    OTHER = "Other"

class Response:
    intro_responses: list[bool]
    initial_testing: list[float]
    main_testing: list[float]
    main_explanations: list[str]
    mental_model: str
    familiarity: int
    referral_text: str
    referral_type: Referral
    truth: list[float]

    referral_calculation = {
        Referral.UNIVERSITY: lambda s: s == "HPI Infoschleuder",
        Referral.BIOLOGY: lambda s: s == "Uni Potsdam FSR",
        Referral.BIRD_FORUMS: lambda s: s in ["/r/whatsthisbird", "Reddit birding", "Reddit - Ornithology", "Reddit - General", "Reddit - What’s this bird"]
}

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

    def __init__(self, raw_response, truth):
        intro_responses = []
        for i in range(len(Response.intro_truth)):
            intro_responses.append(Response.intro_truth[i] == raw_response[f"initial_{i}"])

        initial_testing = []
        main_testing = []
        main_explanations = []
        for i in range(10):
            initial_testing.append(Response.answer_map[raw_response[f"testing_initial_{i}"]])
            main_testing.append(Response.answer_map[raw_response[f"testing_later_{i}"]])
            main_explanations.append(raw_response[f"testing_explanation_{i}"])

        self.intro_responses = intro_responses
        self.initial_testing = initial_testing
        self.main_testing = main_testing
        self.main_explanations = main_explanations
        self.mental_model = raw_response["mental_model"]
        self.familiarity = int(raw_response["bird_familiarity"])
        self.referral_text = raw_response["referral"]
        self.referral_type = Referral.OTHER
        for type, c in Response.referral_calculation.items():
            if c(self.referral_text): 
                self.referral_type = type
        self.truth = truth

    def has_valid_initial_test(self):
        return all(i == 0 for i in self.initial_testing)
    
    def average_accuracy(self):
        return np.count_nonzero(np.array([r * t for r, t in zip(self.main_testing, self.truth)]) > 0) / len(self.main_testing)
    
    def print_explanations(self):
        for i, s in enumerate(self.main_explanations):
            print(f"Question {i + 1}: Correct: {self.truth[i]}; Answer: {self.main_testing[i]}, {s}")

        print(f"Mental Model: {self.mental_model}")

class Study():
    responses: list[Response]
    truth: list[float]
    name: str

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

        for raw_response in raw_responses:
            responses.append(Response(raw_response, truth))

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
        return Study(self.truth, name, [r for r in self.responses if r.has_valid_initial_test()])
    
    def evaluate(self):
        self._evaluate_accuracy()

    def _evaluate_accuracy(self):
        accuracies = []
        for i, response in enumerate(self.responses):
            accuracy = response.average_accuracy()
            print(f"Response {i} of study {self.name} has an accuracy of {accuracy}")
            
            accuracies.append(accuracy)
        
        print(f"Average accuracy for study {self.name} is {np.average(accuracies)}")

    def print_explanations(self):
        print(f"Study {self.name}")
        for i, r in enumerate(self.responses):
            print(f"Response #{i + 1}")
            r.print_explanations()