# This gives all reviews for companies provided in output_business.json

import json
from tqdm import tqdm

data = []
business = open("output_business.json", "r")
business_data = json.load(business)

with open("output.json") as infile:
    data1 = json.load(infile)
    for myline in tqdm(data1):
        if myline["business_id"] in business_data:
            data.append(myline)

with open("output_final.json", "w") as outfile:
    json.dump(data, outfile, indent=2)