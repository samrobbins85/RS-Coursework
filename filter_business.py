# This takes the list of all businesses and returns italian restaurants

import json
from tqdm import tqdm

data = []

for line in tqdm(open("yelp_academic_dataset_business.json", "r")):
    myline = json.loads(line)
    newdict = {
        "business_id": myline["business_id"],
        "name": myline["name"],
    }
    categories = myline["categories"].split(", ") if myline["categories"] else []
    if "Restaurants" in categories and "Italian" in categories:
        data.append(newdict)


with open("output_business_names.json", "w") as outfile:
    json.dump(data, outfile, indent=2)