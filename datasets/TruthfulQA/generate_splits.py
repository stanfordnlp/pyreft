import csv
import random
import json

with open('TruthfulQA.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# convert to our format
converted = []
for d in data:
    converted.append({
        "instruction": d["Question"],
        "input": "",
        "output": d["Best Answer"],
        "answer": d["Best Answer"],
        "type": d["\ufeffType"],
        "category": d["Category"],
    })

# split into 2 sets (for ITI cross-validation)
random.shuffle(converted)
split = len(converted) // 2
split1 = converted[:split]
split2 = converted[split:]

# write
with open('TruthfulQA_split1.json', 'w') as file:
    json.dump(split1, file, indent=4)
with open('TruthfulQA_split2.json', 'w') as file:
    json.dump(split2, file, indent=4)