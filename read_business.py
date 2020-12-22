import json

with open("output_date.json") as infile:
    data = json.load(infile)

print(len(data))
