# This filters the reviews by a certain date

import json
from tqdm import tqdm

output = []

with open("output_final.json") as infile:
    data = json.load(infile)
    for item in tqdm(data):
        if int(item["date"][0:4]) > 2015:
            output.append(item)

with open("output_date.json", "w") as outfile:
    json.dump(output, outfile, indent=2)
