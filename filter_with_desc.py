import json
from tqdm import tqdm

data = []

business = open("output_business.json", "r")
business_data = json.load(business)

for line in tqdm(open("./Dataset/yelp_dataset/yelp_academic_dataset_review.json", "r")):
    myline = json.loads(line)
    if int(myline["date"][0:4]) >= 2019:
        if myline["business_id"] in business_data:
            data.append(myline)

with open("output_desc.json", "w") as outfile:
    json.dump(data, outfile)