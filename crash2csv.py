import pandas as pd
from tqdm import tqdm
import re
import os 

if not os.path.exists("./data"):
    os.mkdir("./data")

### Cleaning training file

train = open("./raw/train.crash").readlines()
id_locations = []
label_locations = []
for idx, line in tqdm(enumerate(train)):
    line = line.strip()
    if line.startswith("train_"):
        id_locations.append(idx)
    elif line == "0" or line == "1":
        label_locations.append(idx)
data = []

for id_loc, l_loc in tqdm(zip(id_locations, label_locations)):
    line_id = train[id_loc].strip()
    label = train[l_loc].strip()
    text = re.sub('\s+', ' ', ' '.join(train[id_loc + 1: l_loc])).strip()[1:-1].strip()
    data.append(f"{line_id}\t{text}\t{label}")

with open("./data/train.csv", "wt") as f:
    f.write("id\ttext\tlabel\n")
    f.write("\n".join(data))

### Cleaning test file

test = open("./raw/test.crash").readlines()
id_locations = []
for idx, line in tqdm(enumerate(test)):
    line = line.strip()
    if line.startswith("test_"):
        id_locations.append(idx)
data = []

for i, id_loc in tqdm(enumerate(id_locations)):
    if i >= len(id_locations) - 1:
        end = len(test)
    else:
        end = id_locations[i + 1]
    line_id = test[id_loc].strip()
    text = re.sub('\s+', ' ', ' '.join(test[id_loc + 1:end])).strip()[1:-1].strip()
    data.append(f"{line_id}\t{text}")

with open("./data/test.csv", "wt") as f:
    f.write("id\ttext\n")
    f.write("\n".join(data))
