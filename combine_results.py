import os
import csv
import re

def combine_results(name, results_path="//store-01.hpi.uni-potsdam.de/fg/doellner/florence.boettger/counterfactuals/output/new_results"):
    files = [os.path.join(results_path, f) for f in os.listdir(results_path) if re.search(f"^{name}_\d*.csv$", f)]

    results = []
    for filename in files:
        with open(filename, "r") as f:
            reader = list(csv.DictReader(f))
            raw_dict = reader[0]
            eval_single = re.findall("\d*\.\d*", raw_dict["eval_single"])
            eval_all = re.findall("\d*\.\d*", raw_dict["eval_all"])
            results.append({
                "id": raw_dict["id"],
                "mode": raw_dict["mode"],
                "lambd": raw_dict["lambd"],
                "lambd2": raw_dict["lambd2"],
                "max_dist": raw_dict["max_dist"],
                "parts_type": raw_dict["parts_type"],
                "avg_edits": raw_dict["avg_edits"],
                "eval_single_near": eval_single[0],
                "eval_single_same": eval_single[1],
                "eval_all_near": eval_all[0],
                "eval_all_same": eval_all[1],
            })

    with open(os.path.join(results_path, "combined", f"{name}.csv"), "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    return results