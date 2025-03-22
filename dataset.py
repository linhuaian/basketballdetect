from sklearn.model_selection import train_test_split
import os
import json
import random


# Define paths, these paths are ignored in git repository
make_folder = "make/"
miss_folder = "miss/"

dataset = {}
for video in os.listdir(make_folder):
    dataset[os.path.join(make_folder, video)] = 1  

for video in os.listdir(miss_folder):
    dataset[os.path.join(miss_folder, video)] = 0  

data = list(dataset.items())
pos = [d for d in data if d[1] == 1]
neg = [d for d in data if d[1] == 0]

test_size = min(len(pos), len(neg)) // 5 

train_pos, test_pos = train_test_split(pos, test_size=test_size, random_state=42)
train_neg, test_neg = train_test_split(neg, test_size=test_size, random_state=42)

train_data = train_pos + train_neg
test_data = test_pos + test_neg

random.shuffle(train_data)

with open("train_data.json", "w") as f:
    json.dump(dict(train_data), f, indent=4)

with open("test_data.json", "w") as f:
    json.dump(dict(test_data), f, indent=4)

print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
