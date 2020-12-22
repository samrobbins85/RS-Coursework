# This takes the list of all businesses and returns italian restaurants

import json
from tqdm import tqdm

data = []

for line in tqdm(open("yelp_academic_dataset_business.json", "r")):
    myline = json.loads(line)
    newdict = {
        "business_id": myline["business_id"],
        "categories": (
            myline["categories"].split(", ") if myline["categories"] else []
        ),
    }
    if "Restaurants" in newdict["categories"] and "Italian" in newdict["categories"]:
        data.append(newdict["business_id"])


with open("output_business.json", "w") as outfile:
    json.dump(data, outfile, indent=2)