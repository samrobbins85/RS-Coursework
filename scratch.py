import json

with open("output_desc.json") as infile:
    data = json.load(infile)

with open("output_desc_format.json", "w") as outfile:
    json.dump(data, outfile, indent=2)
