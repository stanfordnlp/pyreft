from collections import Counter, defaultdict
import glob
import json
from tqdm import tqdm

math_datasets = ["AddSub", "AQuA", "gsm8k", "mawps", "MultiArith", "SingleEq", "SVAMP"]

def analyse_math():
    counts = defaultdict(Counter)
    for dataset in math_datasets:
        print(dataset)
        for file in tqdm(list(glob.glob(f"official_results/*/{dataset}_outputs.json"))):
            run_name = file.split("/")[1]
            with open(file, "r") as f:
                data = json.load(f)
            for item in data:
                try:
                    prefix = item["raw_generation"].split("### Response:\n")[1].split()[0]
                except:
                    prefix = "ERROR!!!"
                counts[run_name][prefix] += 1
    for run_name, count in sorted(list(counts.items()), key=lambda x: -x[1]['1.']):
        print(run_name)
        print(count.most_common())
        print('----')

def main():
    analyse_math()

if __name__ == "__main__":
    main()