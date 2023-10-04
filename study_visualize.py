import matplotlib.pyplot as plt
import numpy as np

def visualize_main_results(responses, truth):
    x = np.arange(40) // 4
    y = np.tile(np.array([-1, -0.5, 0.5, 1]), 10)
    
    size = [np.count_nonzero([r["main_responses"][x[i]] for r in responses] == y[i]) for i in range(len(x))]
    print(size)