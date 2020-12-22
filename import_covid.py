# This takes the covid dataset and makes it easier to import and loop through

import json

data = []


for line in open("yelp_academic_dataset_covid_features.json", "r"):
    myline = json.loads(line)
    data.append(myline)

with open("covid_usable.json", "w") as outfile:
    json.dump(data, outfile)